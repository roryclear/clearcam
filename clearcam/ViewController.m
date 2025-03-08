#import "ViewController.h"
#import <Security/Security.h>
#import <CommonCrypto/CommonCrypto.h>

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    // Generate RSA Key Pair
    NSArray *keyPair = [self generateRSAKeyPair:3072];

    SecKeyRef privateKey = (__bridge SecKeyRef)keyPair[0];
    SecKeyRef publicKey = (__bridge SecKeyRef)keyPair[1];

    // Encrypt Image
    [self encryptImageWithPublicKey:publicKey];

    // Decrypt Image
    [self decryptImageWithPrivateKey:privateKey];
}

- (void)encryptImageWithPublicKey:(SecKeyRef)publicKey {
    NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
    NSString *imagePath = [documentsDirectory stringByAppendingPathComponent:@"image.jpg"];
    NSString *encryptedFilePath = [documentsDirectory stringByAppendingPathComponent:@"image_encrypted.pgp"];

    NSData *imageData = [NSData dataWithContentsOfFile:imagePath];
    if (!imageData) {
        NSLog(@"Failed to read image file");
        return;
    }

    // Generate a random symmetric key
    NSMutableData *symmetricKey = [NSMutableData dataWithLength:32];
    if (SecRandomCopyBytes(kSecRandomDefault, symmetricKey.length, symmetricKey.mutableBytes) != errSecSuccess) {
        NSLog(@"Failed to generate symmetric key");
        return;
    }

    // Encrypt the symmetric key using RSA
    NSData *encryptedSymmetricKey = [self rsaEncryptData:symmetricKey publicKey:publicKey];
    if (!encryptedSymmetricKey) {
        NSLog(@"Failed to encrypt symmetric key");
        return;
    }

    // Generate IV
    NSMutableData *iv = [NSMutableData dataWithLength:kCCBlockSizeAES128];
    if (SecRandomCopyBytes(kSecRandomDefault, iv.length, iv.mutableBytes) != errSecSuccess) {
        NSLog(@"Failed to generate IV");
        return;
    }

    // Encrypt the image data using AES
    NSData *encryptedImageData = [self aesEncryptData:imageData key:symmetricKey iv:iv];
    if (!encryptedImageData) {
        NSLog(@"Failed to encrypt image data");
        return;
    }

    // Save encrypted data
    NSMutableData *encryptedFileData = [NSMutableData data];
    [encryptedFileData appendData:encryptedSymmetricKey];
    [encryptedFileData appendData:iv];
    [encryptedFileData appendData:encryptedImageData];

    [encryptedFileData writeToFile:encryptedFilePath atomically:YES];
    NSLog(@"Encryption complete: %@", encryptedFilePath);
}

- (void)decryptImageWithPrivateKey:(SecKeyRef)privateKey {
    NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
    NSString *encryptedFilePath = [documentsDirectory stringByAppendingPathComponent:@"image_encrypted.pgp"];
    NSString *decryptedFilePath = [documentsDirectory stringByAppendingPathComponent:@"image_decrypted.jpg"];

    NSData *loadedEncryptedFileData = [NSData dataWithContentsOfFile:encryptedFilePath];
    if (!loadedEncryptedFileData) {
        NSLog(@"Failed to read encrypted file");
        return;
    }

    // Extract the encrypted symmetric key, IV, and encrypted image data
    NSData *loadedEncryptedSymmetricKey = [loadedEncryptedFileData subdataWithRange:NSMakeRange(0, 384)];
    NSData *loadedIV = [loadedEncryptedFileData subdataWithRange:NSMakeRange(384, kCCBlockSizeAES128)];
    NSData *loadedEncryptedImageData = [loadedEncryptedFileData subdataWithRange:NSMakeRange(384 + kCCBlockSizeAES128, loadedEncryptedFileData.length - 384 - kCCBlockSizeAES128)];

    // Decrypt the symmetric key using RSA
    NSData *decryptedSymmetricKey = [self rsaDecryptData:loadedEncryptedSymmetricKey privateKey:privateKey];
    if (!decryptedSymmetricKey) {
        NSLog(@"Failed to decrypt symmetric key");
        return;
    }

    // Decrypt the image data using AES
    NSData *decryptedImageData = [self aesDecryptData:loadedEncryptedImageData key:decryptedSymmetricKey iv:loadedIV];
    if (!decryptedImageData) {
        NSLog(@"Failed to decrypt image data");
        return;
    }

    // Save the decrypted image
    [decryptedImageData writeToFile:decryptedFilePath atomically:YES];
    NSLog(@"Decryption complete: %@", decryptedFilePath);
}


// Helper function to generate RSA key pair
- (NSArray *)generateRSAKeyPair:(int)keySize {
    NSMutableDictionary *privateKeyAttributes = [NSMutableDictionary dictionary];
    NSMutableDictionary *publicKeyAttributes = [NSMutableDictionary dictionary];
    NSMutableDictionary *keyPairAttributes = [NSMutableDictionary dictionary];

    // Set key size and type
    keyPairAttributes[(__bridge id)kSecAttrKeyType] = (__bridge id)kSecAttrKeyTypeRSA;
    keyPairAttributes[(__bridge id)kSecAttrKeySizeInBits] = @(keySize);

    // Generate key pair
    SecKeyRef publicKey, privateKey;
    OSStatus status = SecKeyGeneratePair((__bridge CFDictionaryRef)keyPairAttributes, &publicKey, &privateKey);
    if (status != errSecSuccess) {
        NSLog(@"Error generating RSA key pair: %d", (int)status);
        return nil;
    }

    return @[(__bridge id)privateKey, (__bridge id)publicKey];
}

// Helper function to encrypt data using RSA public key
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

// Helper function to decrypt data using RSA private key
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

// Helper function to encrypt data using AES
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

// Helper function to decrypt data using AES
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

