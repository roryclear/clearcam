#import "pgp.h"
#import <CommonCrypto/CommonCrypto.h>
#import <Security/Security.h>

@interface PGP ()

@property (nonatomic) SecKeyRef privateKey;
@property (nonatomic) SecKeyRef publicKey;

@end

@implementation PGP

- (instancetype)init {
    self = [super init];
    if (self) {
        // Try to load existing keys from Keychain
        if ([self loadKeysFromKeychain]) {
            NSLog(@"Loaded existing keys from Keychain");
        } else {
            // Generate new keys if they don't exist
            NSArray *keyPair = [self generateRSAKeyPair:3072];
            if (keyPair) {
                _privateKey = (__bridge_retained SecKeyRef)keyPair[0];
                _publicKey = (__bridge_retained SecKeyRef)keyPair[1];
                
                // Save the new keys to Keychain
                if ([self saveKeysToKeychain]) {
                    NSLog(@"Generated and saved new keys to Keychain");
                } else {
                    NSLog(@"Failed to save new keys to Keychain");
                    return nil;
                }
            } else {
                NSLog(@"Failed to generate keys");
                return nil;
            }
        }
        
        //todo dont need
        CFErrorRef error = NULL;
        CFDataRef publicKeyData = SecKeyCopyExternalRepresentation(self.publicKey, &error);
        NSData *keyData = (__bridge_transfer NSData *)publicKeyData;
        NSString *base64Key = [keyData base64EncodedStringWithOptions:0];
        NSLog(@"Public Key (Base64): %@", base64Key);
        //
        
        /*
        NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
        NSString *documentsDirectory = [paths firstObject];
        NSString *imagePath = [documentsDirectory stringByAppendingPathComponent:@"image.jpg"];
        NSString *encryptedImagePath = [documentsDirectory stringByAppendingPathComponent:@"image.pgp"];
        [self encryptImageWithPublicKey:imagePath];
        [self decryptImageWithPrivateKey:encryptedImagePath];
         */
        //[self deleteKeysFromKeychain];
    }
    return self;
}

- (void)dealloc {
    if (_privateKey) CFRelease(_privateKey);
    if (_publicKey) CFRelease(_publicKey);
}

//only way to delete them
- (void)deleteKeysFromKeychain {
    NSString *privateKeyIdentifier = @"com.clearcam.pgp.privateKey";
    NSString *publicKeyIdentifier = @"com.clearcam.pgp.publicKey";

    NSDictionary *privateDeleteQuery = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassKey,
        (__bridge id)kSecAttrApplicationTag: privateKeyIdentifier
    };
    SecItemDelete((__bridge CFDictionaryRef)privateDeleteQuery);

    NSDictionary *publicDeleteQuery = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassKey,
        (__bridge id)kSecAttrApplicationTag: publicKeyIdentifier
    };
    SecItemDelete((__bridge CFDictionaryRef)publicDeleteQuery);
}

// Method to save keys to Keychain
- (BOOL)saveKeysToKeychain {
    // Keychain identifiers
    NSString *privateKeyIdentifier = @"com.clearcam.pgp.privateKey";
    NSString *publicKeyIdentifier = @"com.clearcam.pgp.publicKey";

    // Export private key
    CFErrorRef error = NULL;
    CFDataRef privateKeyData = SecKeyCopyExternalRepresentation(self.privateKey, &error);
    if (!privateKeyData) {
        NSLog(@"Failed to export private key: %@", CFBridgingRelease(error));
        return NO;
    }

    // Export public key
    error = NULL;
    CFDataRef publicKeyData = SecKeyCopyExternalRepresentation(self.publicKey, &error);
    if (!publicKeyData) {
        NSLog(@"Failed to export public key: %@", CFBridgingRelease(error));
        if (privateKeyData) CFRelease(privateKeyData);
        return NO;
    }
    
    // Save private key to Keychain
    NSDictionary *privateKeyQuery = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassKey,
        (__bridge id)kSecAttrApplicationTag: privateKeyIdentifier,
        (__bridge id)kSecAttrKeyType: (__bridge id)kSecAttrKeyTypeRSA,
        (__bridge id)kSecAttrKeyClass: (__bridge id)kSecAttrKeyClassPrivate,
        (__bridge id)kSecValueData: (__bridge id)privateKeyData,
        (__bridge id)kSecAttrIsPermanent: @YES,
        (__bridge id)kSecAttrAccessible: (__bridge id)kSecAttrAccessibleWhenUnlocked
    };
    OSStatus privateSaveStatus = SecItemAdd((__bridge CFDictionaryRef)privateKeyQuery, NULL);
    if (privateSaveStatus != errSecSuccess) {
        NSLog(@"Failed to save private key to Keychain: %d", (int)privateSaveStatus);
        if (privateKeyData) CFRelease(privateKeyData);
        if (publicKeyData) CFRelease(publicKeyData);
        return NO;
    }

    // Save public key to Keychain
    NSDictionary *publicKeyQuery = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassKey,
        (__bridge id)kSecAttrApplicationTag: publicKeyIdentifier,
        (__bridge id)kSecAttrKeyType: (__bridge id)kSecAttrKeyTypeRSA,
        (__bridge id)kSecAttrKeyClass: (__bridge id)kSecAttrKeyClassPublic,
        (__bridge id)kSecValueData: (__bridge id)publicKeyData,
        (__bridge id)kSecAttrIsPermanent: @YES,
        (__bridge id)kSecAttrAccessible: (__bridge id)kSecAttrAccessibleWhenUnlocked
    };
    OSStatus publicSaveStatus = SecItemAdd((__bridge CFDictionaryRef)publicKeyQuery, NULL);
    if (publicSaveStatus != errSecSuccess) {
        NSLog(@"Failed to save public key to Keychain: %d", (int)publicSaveStatus);
        // Clean up private key if public key save fails
        NSDictionary *privateDeleteQuery = @{
            (__bridge id)kSecClass: (__bridge id)kSecClassKey,
            (__bridge id)kSecAttrApplicationTag: privateKeyIdentifier
        };
        SecItemDelete((__bridge CFDictionaryRef)privateDeleteQuery);
        if (privateKeyData) CFRelease(privateKeyData);
        if (publicKeyData) CFRelease(publicKeyData);
        return NO;
    }

    if (privateKeyData) CFRelease(privateKeyData);
    if (publicKeyData) CFRelease(publicKeyData);
    return YES;
}

// Method to load keys from Keychain
- (BOOL)loadKeysFromKeychain {
    // Keychain identifiers
    NSString *privateKeyIdentifier = @"com.clearcam.pgp.privateKey";
    NSString *publicKeyIdentifier = @"com.clearcam.pgp.publicKey";

    // Query for private key
    NSDictionary *privateKeyQuery = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassKey,
        (__bridge id)kSecAttrApplicationTag: privateKeyIdentifier,
        (__bridge id)kSecAttrKeyType: (__bridge id)kSecAttrKeyTypeRSA,
        (__bridge id)kSecAttrKeyClass: (__bridge id)kSecAttrKeyClassPrivate,
        (__bridge id)kSecReturnRef: @YES
    };
    SecKeyRef privateKey = NULL;
    OSStatus privateStatus = SecItemCopyMatching((__bridge CFDictionaryRef)privateKeyQuery, (CFTypeRef *)&privateKey);
    if (privateStatus == errSecItemNotFound) {
        // Key doesn't exist yet, which is fine for first run
        return NO;
    } else if (privateStatus != errSecSuccess) {
        NSLog(@"Failed to load private key from Keychain: %d", (int)privateStatus);
        return NO;
    }

    // Query for public key
    NSDictionary *publicKeyQuery = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassKey,
        (__bridge id)kSecAttrApplicationTag: publicKeyIdentifier,
        (__bridge id)kSecAttrKeyType: (__bridge id)kSecAttrKeyTypeRSA,
        (__bridge id)kSecAttrKeyClass: (__bridge id)kSecAttrKeyClassPublic,
        (__bridge id)kSecReturnRef: @YES
    };
    SecKeyRef publicKey = NULL;
    OSStatus publicStatus = SecItemCopyMatching((__bridge CFDictionaryRef)publicKeyQuery, (CFTypeRef *)&publicKey);
    if (publicStatus == errSecItemNotFound) {
        // Key doesn't exist yet, which is fine for first run
        if (privateKey) CFRelease(privateKey);
        return NO;
    } else if (publicStatus != errSecSuccess) {
        NSLog(@"Failed to load public key from Keychain: %d", (int)publicStatus);
        if (privateKey) CFRelease(privateKey);
        return NO;
    }

    // If we successfully loaded both keys, set the properties
    if (_privateKey) CFRelease(_privateKey);
    if (_publicKey) CFRelease(_publicKey);
    _privateKey = privateKey;
    _publicKey = publicKey;

    return YES;
}

// Modified to ensure keys are extractable
- (NSArray *)generateRSAKeyPair:(int)keySize {
    NSMutableDictionary *privateKeyAttr = [NSMutableDictionary dictionary];
    NSMutableDictionary *publicKeyAttr = [NSMutableDictionary dictionary];
    NSMutableDictionary *keyPairAttr = [NSMutableDictionary dictionary];

    // Set key type and size
    keyPairAttr[(__bridge id)kSecAttrKeyType] = (__bridge id)kSecAttrKeyTypeRSA;
    keyPairAttr[(__bridge id)kSecAttrKeySizeInBits] = @(keySize);
    
    // Ensure keys can be exported and stored in Keychain
    keyPairAttr[(__bridge id)kSecAttrIsPermanent] = @YES;
    privateKeyAttr[(__bridge id)kSecAttrIsPermanent] = @YES;
    publicKeyAttr[(__bridge id)kSecAttrIsPermanent] = @YES;
    
    // Add application tags to help identify keys in Keychain
    NSString *privateKeyIdentifier = @"com.clearcam.pgp.privateKey";
    NSString *publicKeyIdentifier = @"com.clearcam.pgp.publicKey";
    privateKeyAttr[(__bridge id)kSecAttrApplicationTag] = privateKeyIdentifier;
    publicKeyAttr[(__bridge id)kSecAttrApplicationTag] = publicKeyIdentifier;

    keyPairAttr[(__bridge id)kSecPrivateKeyAttrs] = privateKeyAttr;
    keyPairAttr[(__bridge id)kSecPublicKeyAttrs] = publicKeyAttr;

    SecKeyRef publicKey, privateKey;
    OSStatus status = SecKeyGeneratePair((__bridge CFDictionaryRef)keyPairAttr, &publicKey, &privateKey);
    if (status != errSecSuccess) {
        NSLog(@"Error generating RSA key pair: %d", (int)status);
        return nil;
    }

    return @[(__bridge_transfer id)privateKey, (__bridge_transfer id)publicKey];
}

// Existing methods remain unchanged
- (void)encryptImageWithPublicKey:(NSString *)filePath {
    NSString *encryptedFilePath = [[filePath stringByDeletingPathExtension] stringByAppendingPathExtension:@"pgp"];
    NSData *imageData = [NSData dataWithContentsOfFile:filePath];
    
    NSMutableData *symmetricKey = [NSMutableData dataWithLength:32];
    NSData *encryptedSymmetricKey = [self rsaEncryptData:symmetricKey publicKey:self.publicKey];

    NSMutableData *iv = [NSMutableData dataWithLength:kCCBlockSizeAES128];
    NSData *encryptedImageData = [self aesEncryptData:imageData key:symmetricKey iv:iv];

    NSMutableData *encryptedFileData = [NSMutableData data];
    [encryptedFileData appendData:encryptedSymmetricKey];
    [encryptedFileData appendData:iv];
    [encryptedFileData appendData:encryptedImageData];

    if ([encryptedFileData writeToFile:encryptedFilePath atomically:YES]) {
        NSLog(@"Encryption complete: %@", encryptedFilePath);
    } else {
        NSLog(@"Failed to write encrypted file");
    }
}

- (void)decryptImageWithPrivateKey:(NSString *)encryptedFilePath {
    NSString *decryptedFilePath = [[encryptedFilePath stringByDeletingPathExtension] stringByAppendingPathExtension:@"jpg"];
    NSData *loadedEncryptedFileData = [NSData dataWithContentsOfFile:encryptedFilePath];

    NSData *loadedEncryptedSymmetricKey = [loadedEncryptedFileData subdataWithRange:NSMakeRange(0, 384)];
    NSData *loadedIV = [loadedEncryptedFileData subdataWithRange:NSMakeRange(384, kCCBlockSizeAES128)];
    NSData *loadedEncryptedImageData = [loadedEncryptedFileData subdataWithRange:NSMakeRange(384 + kCCBlockSizeAES128, loadedEncryptedFileData.length - 384 - kCCBlockSizeAES128)];

    NSData *decryptedSymmetricKey = [self rsaDecryptData:loadedEncryptedSymmetricKey privateKey:self.privateKey];
    NSData *decryptedImageData = [self aesDecryptData:loadedEncryptedImageData key:decryptedSymmetricKey iv:loadedIV];

    if ([decryptedImageData writeToFile:decryptedFilePath atomically:YES]) {
        NSLog(@"Decryption complete: %@", decryptedFilePath);
    } else {
        NSLog(@"Failed to write decrypted file");
    }
}

- (NSData *)rsaEncryptData:(NSData *)data publicKey:(SecKeyRef)publicKey {
    size_t cipherBufferSize = SecKeyGetBlockSize(publicKey);
    uint8_t *cipherBuffer = malloc(cipherBufferSize);
    OSStatus status = SecKeyEncrypt(publicKey, kSecPaddingOAEP, data.bytes, data.length, cipherBuffer, &cipherBufferSize);

    if (status != errSecSuccess) {
        NSLog(@"Error encrypting data: %d", (int)status);
        free(cipherBuffer);
        return nil;
    }

    NSData *encryptedData = [NSData dataWithBytes:cipherBuffer length:cipherBufferSize];
    free(cipherBuffer);
    return encryptedData;
}

- (NSData *)rsaDecryptData:(NSData *)data privateKey:(SecKeyRef)privateKey {
    size_t plainBufferSize = SecKeyGetBlockSize(privateKey);
    uint8_t *plainBuffer = malloc(plainBufferSize);
    OSStatus status = SecKeyDecrypt(privateKey, kSecPaddingOAEP, data.bytes, data.length, plainBuffer, &plainBufferSize);

    if (status != errSecSuccess) {
        NSLog(@"Error decrypting data: %d", (int)status);
        free(plainBuffer);
        return nil;
    }

    NSData *decryptedData = [NSData dataWithBytes:plainBuffer length:plainBufferSize];
    free(plainBuffer);
    return decryptedData;
}

- (NSData *)aesEncryptData:(NSData *)data key:(NSData *)key iv:(NSData *)iv {
    size_t bufferSize = data.length + kCCBlockSizeAES128;
    void *buffer = malloc(bufferSize);
    size_t numBytesEncrypted = 0;

    CCCryptorStatus status = CCCrypt(kCCEncrypt, kCCAlgorithmAES, kCCOptionPKCS7Padding, key.bytes, key.length, iv.bytes, data.bytes, data.length, buffer, bufferSize, &numBytesEncrypted);

    if (status != kCCSuccess) {
        NSLog(@"AES encryption failed: %d", (int)status);
        free(buffer);
        return nil;
    }

    return [NSData dataWithBytesNoCopy:buffer length:numBytesEncrypted freeWhenDone:YES];
}

- (NSData *)aesDecryptData:(NSData *)data key:(NSData *)key iv:(NSData *)iv {
    size_t bufferSize = data.length + kCCBlockSizeAES128;
    void *buffer = malloc(bufferSize);
    size_t numBytesDecrypted = 0;

    CCCryptorStatus status = CCCrypt(kCCDecrypt, kCCAlgorithmAES, kCCOptionPKCS7Padding, key.bytes, key.length, iv.bytes, data.bytes, data.length, buffer, bufferSize, &numBytesDecrypted);

    if (status != kCCSuccess) {
        NSLog(@"AES decryption failed: %d", (int)status);
        free(buffer);
        return nil;
    }

    return [NSData dataWithBytesNoCopy:buffer length:numBytesDecrypted freeWhenDone:YES];
}

@end
