#import "SecretManager.h"
#import <Security/Security.h>
#import <CommonCrypto/CommonCryptor.h>

static NSString *const kServiceIdentifier = @"com.yourapp.aeskeys"; // Replace with your app's unique identifier

@implementation SecretManager

#pragma mark - Singleton

+ (instancetype)sharedManager {
    static SecretManager *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[SecretManager alloc] init];
    });
    return sharedInstance;
}

#pragma mark - Key Retrieval

- (NSArray<NSString *> *)getAllStoredKeys {
    NSMutableArray<NSString *> *keys = [NSMutableArray array];
    
    // Query the Keychain for all stored keys
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: kServiceIdentifier,
        (__bridge id)kSecReturnAttributes: @YES,
        (__bridge id)kSecReturnData: @YES,
        (__bridge id)kSecMatchLimit: (__bridge id)kSecMatchLimitAll
    };
    
    CFTypeRef result = NULL;
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, &result);
    if (status == errSecSuccess) {
        NSArray *items = (__bridge_transfer NSArray *)result;
        for (NSDictionary *item in items) {
            NSData *keyData = item[(__bridge id)kSecValueData];
            if (keyData) {
                NSString *key = [[NSString alloc] initWithData:keyData encoding:NSUTF8StringEncoding];
                if (key) {
                    [keys addObject:key];
                }
            }
        }
    } else if (status != errSecItemNotFound) {
        NSLog(@"Failed to retrieve keys from Keychain: %d", (int)status);
    }
    
    return [keys copy];
}


- (BOOL)saveEncryptionKey:(NSString *)key error:(NSError **)error {
    if (!key) {
        if (error) {
            *error = [NSError errorWithDomain:@"SecretManagerErrorDomain"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Encryption key cannot be nil."}];
        }
        return NO;
    }

    NSData *keyData = [key dataUsingEncoding:NSUTF8StringEncoding];

    // Define a fixed identifier for the encryption key
    NSString *encryptionKeyIdentifier = @"encryption_key";

    // Delete any existing encryption key before saving a new one
    [self deleteKeyWithIdentifier:encryptionKeyIdentifier error:nil];

    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: kServiceIdentifier,
        (__bridge id)kSecAttrAccount: encryptionKeyIdentifier,
        (__bridge id)kSecValueData: keyData,
        (__bridge id)kSecAttrAccessible: (__bridge id)kSecAttrAccessibleWhenUnlocked
    };

    OSStatus status = SecItemAdd((__bridge CFDictionaryRef)query, NULL);
    if (status != errSecSuccess) {
        NSLog(@"Failed to save encryption key: %d", (int)status);
        if (error) {
            *error = [NSError errorWithDomain:@"SecretManagerErrorDomain"
                                         code:status
                                     userInfo:@{NSLocalizedDescriptionKey: [self errorMessageForStatus:status]}];
        }
        return NO;
    }

    NSLog(@"Successfully saved encryption key.");
    return YES;
}

- (NSArray<NSString *> *)getAllDecryptionKeys {
    NSMutableArray<NSString *> *keys = [NSMutableArray array];

    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: kServiceIdentifier,
        (__bridge id)kSecReturnAttributes: @YES,
        (__bridge id)kSecReturnData: @YES,
        (__bridge id)kSecMatchLimit: (__bridge id)kSecMatchLimitAll
    };

    CFTypeRef result = NULL;
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, &result);
    if (status == errSecSuccess) {
        NSArray *items = (__bridge_transfer NSArray *)result;
        for (NSDictionary *item in items) {
            NSString *identifier = item[(__bridge id)kSecAttrAccount];
            if (![identifier isEqualToString:@"encryption_key"]) { // Ignore the encryption key
                NSData *keyData = item[(__bridge id)kSecValueData];
                NSString *key = [[NSString alloc] initWithData:keyData encoding:NSUTF8StringEncoding];
                if (key) {
                    [keys addObject:key];
                }
            }
        }
    }

    return [keys copy];
}


- (BOOL)saveDecryptionKey:(NSString *)key withIdentifier:(NSString *)identifier error:(NSError **)error {
    if (!key || !identifier) {
        if (error) {
            *error = [NSError errorWithDomain:@"SecretManagerErrorDomain"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Decryption key and identifier cannot be nil."}];
        }
        return NO;
    }

    NSData *keyData = [key dataUsingEncoding:NSUTF8StringEncoding];

    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: kServiceIdentifier,
        (__bridge id)kSecAttrAccount: identifier,
        (__bridge id)kSecValueData: keyData,
        (__bridge id)kSecAttrAccessible: (__bridge id)kSecAttrAccessibleWhenUnlocked
    };

    // Ensure we don't overwrite an existing key with the same identifier
    SecItemDelete((__bridge CFDictionaryRef)query);

    OSStatus status = SecItemAdd((__bridge CFDictionaryRef)query, NULL);
    if (status != errSecSuccess) {
        NSLog(@"Failed to save decryption key: %d", (int)status);
        if (error) {
            *error = [NSError errorWithDomain:@"SecretManagerErrorDomain"
                                         code:status
                                     userInfo:@{NSLocalizedDescriptionKey: [self errorMessageForStatus:status]}];
        }
        return NO;
    }

    NSLog(@"Successfully saved decryption key with identifier: %@", identifier);
    return YES;
}


- (NSString *)getEncryptionKey {
    NSString *encryptionKeyIdentifier = @"encryption_key";

    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: kServiceIdentifier,
        (__bridge id)kSecAttrAccount: encryptionKeyIdentifier,
        (__bridge id)kSecReturnData: @YES,
        (__bridge id)kSecMatchLimit: (__bridge id)kSecMatchLimitOne
    };

    CFTypeRef result = NULL;
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, &result);

    if (status == errSecSuccess) {
        NSData *keyData = (__bridge_transfer NSData *)result;
        return [[NSString alloc] initWithData:keyData encoding:NSUTF8StringEncoding];
    }

    NSLog(@"Encryption key not found.");
    return nil;
}


#pragma mark - Key Deletion

- (BOOL)deleteKeyWithIdentifier:(NSString *)identifier error:(NSError **)error {
    if (!identifier) {
        if (error) {
            *error = [NSError errorWithDomain:@"SecretManagerErrorDomain"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Identifier cannot be nil."}];
        }
        return NO;
    }
    
    // Keychain query to delete the key
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: kServiceIdentifier,
        (__bridge id)kSecAttrAccount: identifier
    };
    
    OSStatus status = SecItemDelete((__bridge CFDictionaryRef)query);
    if (status != errSecSuccess && status != errSecItemNotFound) {
        NSLog(@"Failed to delete key from Keychain: %d", (int)status);
        if (error) {
            *error = [NSError errorWithDomain:@"SecretManagerErrorDomain"
                                         code:status
                                     userInfo:@{NSLocalizedDescriptionKey: [self errorMessageForStatus:status]}];
        }
        return NO;
    }
    
    NSLog(@"Successfully deleted key from Keychain with identifier: %@", identifier);
    return YES;
}

- (BOOL)deleteAllKeysWithError:(NSError **)error {
    // Keychain query to delete all keys for this service
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: kServiceIdentifier
    };
    
    OSStatus status = SecItemDelete((__bridge CFDictionaryRef)query);
    if (status != errSecSuccess && status != errSecItemNotFound) {
        NSLog(@"Failed to delete all keys from Keychain: %d", (int)status);
        if (error) {
            *error = [NSError errorWithDomain:@"SecretManagerErrorDomain"
                                         code:status
                                     userInfo:@{NSLocalizedDescriptionKey: [self errorMessageForStatus:status]}];
        }
        return NO;
    }
    
    NSLog(@"Successfully deleted all keys from Keychain.");
    return YES;
}

#pragma mark - Helper Methods

- (NSString *)errorMessageForStatus:(OSStatus)status {
    switch (status) {
        case errSecDuplicateItem:
            return @"A key with this identifier already exists.";
        case errSecItemNotFound:
            return @"The requested key was not found.";
        case errSecAuthFailed:
            return @"Authentication failed.";
        default:
            return [NSString stringWithFormat:@"Keychain error %d occurred.", (int)status];
    }
}

#define MAGIC_NUMBER 0x4D41474943ULL // "MAGIC" in ASCII as a 64-bit value
#define HEADER_SIZE (sizeof(uint64_t)) // Size of the magic number (8 bytes)

- (NSData *)decryptData:(NSData *)encryptedData withKey:(NSString *)key {
    // Step 1: Decrypt the entire data first (we need at least the header)
    char keyPtr[kCCKeySizeAES256 + 1]; // Buffer for key
    bzero(keyPtr, sizeof(keyPtr)); // Zero out buffer
    [key getCString:keyPtr maxLength:sizeof(keyPtr) encoding:NSUTF8StringEncoding];
    
    size_t bufferSize = encryptedData.length + kCCBlockSizeAES128;
    void *buffer = malloc(bufferSize);
    
    size_t numBytesDecrypted = 0;
    CCCryptorStatus status = CCCrypt(kCCDecrypt,
                                     kCCAlgorithmAES,
                                     kCCOptionPKCS7Padding,
                                     keyPtr,
                                     kCCKeySizeAES256,
                                     NULL, // IV (NULL for no IV)
                                     encryptedData.bytes,
                                     encryptedData.length,
                                     buffer,
                                     bufferSize,
                                     &numBytesDecrypted);
    
    if (status != kCCSuccess) {
        free(buffer);
        return nil;
    }
    
    // Step 2: Check if the decrypted data is long enough to contain the header
    if (numBytesDecrypted < HEADER_SIZE) {
        free(buffer);
        return nil; // Data is too short to contain a valid header
    }
    
    // Step 3: Extract and verify the magic number from the header
    uint64_t decryptedMagicNumber;
    memcpy(&decryptedMagicNumber, buffer, HEADER_SIZE);
    if (decryptedMagicNumber != MAGIC_NUMBER) {
        free(buffer);
        return nil; // Wrong key or corrupted data
    }
    
    // Step 4: Extract the original data (skip the header)
    size_t originalDataLength = numBytesDecrypted - HEADER_SIZE;
    void *originalDataBuffer = malloc(originalDataLength);
    memcpy(originalDataBuffer, buffer + HEADER_SIZE, originalDataLength);
    
    // Step 5: Return the original data
    NSData *result = [NSData dataWithBytesNoCopy:originalDataBuffer length:originalDataLength];
    free(buffer); // Free the full decrypted buffer
    return result;
}

@end
