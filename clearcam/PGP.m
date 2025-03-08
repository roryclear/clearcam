#import "pgp.h"
#import <Security/Security.h>
#import <CommonCrypto/CommonCrypto.h>

@implementation PGP

- (instancetype)init {
    self = [super init];
    if (self) {
        // Generate RSA Key Pair
        NSArray *keyPair = [self generateRSAKeyPair:3072];

        SecKeyRef privateKey = (__bridge SecKeyRef)keyPair[0];
        SecKeyRef publicKey = (__bridge SecKeyRef)keyPair[1];

        NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
        NSString *imagePath = [documentsDirectory stringByAppendingPathComponent:@"image.jpg"];
        [self encryptImageWithPublicKey:publicKey filePath:imagePath];

        NSString *encryptedImagePath = [documentsDirectory stringByAppendingPathComponent:@"image.pgp"];
        [self decryptImageWithPrivateKey:privateKey filePath:encryptedImagePath];
    }
    return self;
}

- (void)encryptImageWithPublicKey:(SecKeyRef)publicKey filePath:(NSString *)filePath {
    NSString *encryptedFilePath = [[filePath stringByDeletingPathExtension] stringByAppendingPathExtension:@"pgp"];
    NSData *imageData = [NSData dataWithContentsOfFile:filePath];

    NSMutableData *symmetricKey = [NSMutableData dataWithLength:32];
    NSData *encryptedSymmetricKey = [self rsaEncryptData:symmetricKey publicKey:publicKey];

    NSMutableData *iv = [NSMutableData dataWithLength:kCCBlockSizeAES128];
    NSData *encryptedImageData = [self aesEncryptData:imageData key:symmetricKey iv:iv];

    NSMutableData *encryptedFileData = [NSMutableData data];
    [encryptedFileData appendData:encryptedSymmetricKey];
    [encryptedFileData appendData:iv];
    [encryptedFileData appendData:encryptedImageData];

    if ([encryptedFileData writeToFile:encryptedFilePath atomically:YES]) {
        NSLog(@"Encryption complete: %@", encryptedFilePath);
    } else {
        NSLog(@"Failed to write encrypted file to %@", encryptedFilePath);
    }
}

- (void)decryptImageWithPrivateKey:(SecKeyRef)privateKey filePath:(NSString *)encryptedFilePath {
    NSString *decryptedFilePath = [[encryptedFilePath stringByDeletingPathExtension] stringByAppendingPathExtension:@"jpg"];
    NSData *loadedEncryptedFileData = [NSData dataWithContentsOfFile:encryptedFilePath];

    NSData *loadedEncryptedSymmetricKey = [loadedEncryptedFileData subdataWithRange:NSMakeRange(0, 384)];
    NSData *loadedIV = [loadedEncryptedFileData subdataWithRange:NSMakeRange(384, kCCBlockSizeAES128)];
    NSData *loadedEncryptedImageData = [loadedEncryptedFileData subdataWithRange:NSMakeRange(384 + kCCBlockSizeAES128, loadedEncryptedFileData.length - 384 - kCCBlockSizeAES128)];

    NSData *decryptedSymmetricKey = [self rsaDecryptData:loadedEncryptedSymmetricKey privateKey:privateKey];
    NSData *decryptedImageData = [self aesDecryptData:loadedEncryptedImageData key:decryptedSymmetricKey iv:loadedIV];

    if ([decryptedImageData writeToFile:decryptedFilePath atomically:YES]) {
        NSLog(@"Decryption complete: %@", decryptedFilePath);
    } else {
        NSLog(@"Failed to write decrypted file to %@", decryptedFilePath);
    }
}

- (NSArray *)generateRSAKeyPair:(int)keySize {
    NSMutableDictionary *keyPairAttributes = [NSMutableDictionary dictionary];
    keyPairAttributes[(__bridge id)kSecAttrKeyType] = (__bridge id)kSecAttrKeyTypeRSA;
    keyPairAttributes[(__bridge id)kSecAttrKeySizeInBits] = @(keySize);

    SecKeyRef publicKey, privateKey;
    OSStatus status = SecKeyGeneratePair((__bridge CFDictionaryRef)keyPairAttributes, &publicKey, &privateKey);
    if (status != errSecSuccess) {
        NSLog(@"Error generating RSA key pair: %d", (int)status);
        return nil;
    }

    return @[(__bridge id)privateKey, (__bridge id)publicKey];
}

- (NSData *)rsaEncryptData:(NSData *)data publicKey:(SecKeyRef)publicKey {
    size_t cipherBufferSize = SecKeyGetBlockSize(publicKey);
    uint8_t *cipherBuffer = malloc(cipherBufferSize);

    OSStatus status = SecKeyEncrypt(
        publicKey,
        kSecPaddingOAEP,
        data.bytes,
        data.length,
        cipherBuffer,
        &cipherBufferSize
    );

    if (status != errSecSuccess) {
        NSLog(@"Error encrypting data with RSA: %d", (int)status);
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

    OSStatus status = SecKeyDecrypt(
        privateKey,
        kSecPaddingOAEP,
        data.bytes,
        data.length,
        plainBuffer,
        &plainBufferSize
    );

    if (status != errSecSuccess) {
        NSLog(@"Error decrypting data with RSA: %d", (int)status);
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

    CCCryptorStatus status = CCCrypt(
        kCCEncrypt,
        kCCAlgorithmAES,
        kCCOptionPKCS7Padding,
        key.bytes,
        key.length,
        iv.bytes,
        data.bytes,
        data.length,
        buffer,
        bufferSize,
        &numBytesEncrypted
    );

    if (status != kCCSuccess) {
        NSLog(@"Error encrypting data with AES: %d", (int)status);
        free(buffer);
        return nil;
    }

    return [NSData dataWithBytesNoCopy:buffer length:numBytesEncrypted freeWhenDone:YES];
}

- (NSData *)aesDecryptData:(NSData *)data key:(NSData *)key iv:(NSData *)iv {
    size_t bufferSize = data.length + kCCBlockSizeAES128;
    void *buffer = malloc(bufferSize);
    size_t numBytesDecrypted = 0;

    CCCryptorStatus status = CCCrypt(
        kCCDecrypt,
        kCCAlgorithmAES,
        kCCOptionPKCS7Padding,
        key.bytes,
        key.length,
        iv.bytes,
        data.bytes,
        data.length,
        buffer,
        bufferSize,
        &numBytesDecrypted
    );

    if (status != kCCSuccess) {
        NSLog(@"Error decrypting data with AES: %d", (int)status);
        free(buffer);
        return nil;
    }

    return [NSData dataWithBytesNoCopy:buffer length:numBytesDecrypted freeWhenDone:YES];
}

@end
