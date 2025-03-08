#import "pgp.h"
#import <CommonCrypto/CommonCrypto.h>

@interface PGP ()

@property (nonatomic) SecKeyRef privateKey;
@property (nonatomic) SecKeyRef publicKey;

@end

@implementation PGP

- (instancetype)init {
    self = [super init];
    if (self) {
        NSArray *keyPair = [self generateRSAKeyPair:3072];
        _privateKey = (__bridge_retained SecKeyRef)keyPair[0];
        _publicKey = (__bridge_retained SecKeyRef)keyPair[1];
    }
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths firstObject];
    NSString *imagePath = [documentsDirectory stringByAppendingPathComponent:@"image.jpg"];
    NSString *encryptedImagePath = [documentsDirectory stringByAppendingPathComponent:@"image.pgp"];
    [self encryptImageWithPublicKey:imagePath];
    [self decryptImageWithPrivateKey:encryptedImagePath];
    return self;
}

- (void)dealloc {
    if (_privateKey) CFRelease(_privateKey);
    if (_publicKey) CFRelease(_publicKey);
}

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

    return @[(__bridge_transfer id)privateKey, (__bridge_transfer id)publicKey];
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

