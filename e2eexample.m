#import "ViewController.h"
#import <Security/Security.h>
#import <CommonCrypto/CommonCrypto.h>

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    [self runEncryptionDecryption];
}

- (void)runEncryptionDecryption {
    // Generate RSA key pair
    NSArray *keyPair = [self generateRSAKeyPair:3072];
    if (!keyPair) {
        NSLog(@"Failed to generate RSA key pair");
        return;
    }

    SecKeyRef privateKey = (__bridge SecKeyRef)keyPair[0];
    SecKeyRef publicKey = (__bridge SecKeyRef)keyPair[1];

    NSLog(@"RSA key pair generation complete");

    // Paths for the image and encrypted/decrypted files
    NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
    NSString *imagePath = [documentsDirectory stringByAppendingPathComponent:@"image.jpg"];
    NSString *encryptedFilePath = [documentsDirectory stringByAppendingPathComponent:@"image_encrypted.pgp"];
    NSString *decryptedFilePath = [documentsDirectory stringByAppendingPathComponent:@"image_decrypted.jpg"];

    // Read the image file
    NSData *imageData = [NSData dataWithContentsOfFile:imagePath];
    if (!imageData) {
        NSLog(@"Failed to read image file");
        return;
    }

    // Generate a random symmetric key for AES encryption
    NSMutableData *symmetricKey = [NSMutableData dataWithLength:32];
    int result = SecRandomCopyBytes(kSecRandomDefault, symmetricKey.length, symmetricKey.mutableBytes);
    if (result != errSecSuccess) {
        NSLog(@"Failed to generate symmetric key");
        return;
    }

    // Encrypt the symmetric key with the RSA public key
    NSData *encryptedSymmetricKey = [self rsaEncryptData:symmetricKey publicKey:publicKey];
    if (!encryptedSymmetricKey) {
        NSLog(@"Failed to encrypt symmetric key");
        return;
    }

    // Encrypt the image data using the symmetric key
    NSMutableData *iv = [NSMutableData dataWithLength:kCCBlockSizeAES128];
    result = SecRandomCopyBytes(kSecRandomDefault, iv.length, iv.mutableBytes);
    if (result != errSecSuccess) {
        NSLog(@"Failed to generate IV");
        return;
    }

    NSData *encryptedImageData = [self aesEncryptData:imageData key:symmetricKey iv:iv];
    if (!encryptedImageData) {
        NSLog(@"Failed to encrypt image data");
        return;
    }

    // Save the encrypted symmetric key and encrypted image data
    NSMutableData *encryptedFileData = [NSMutableData data];
    [encryptedFileData appendData:encryptedSymmetricKey];
    [encryptedFileData appendData:iv];
    [encryptedFileData appendData:encryptedImageData];

    [encryptedFileData writeToFile:encryptedFilePath atomically:YES];
    NSLog(@"Encryption complete: %@", encryptedFilePath);

    // Read the encrypted symmetric key and encrypted image data
    NSData *loadedEncryptedFileData = [NSData dataWithContentsOfFile:encryptedFilePath];
    if (!loadedEncryptedFileData) {
        NSLog(@"Failed to read encrypted file");
        return;
    }

    NSData *loadedEncryptedSymmetricKey = [loadedEncryptedFileData subdataWithRange:NSMakeRange(0, 384)]; // RSA key size / 8 (3072 bits / 8 = 384 bytes)
    NSData *loadedIV = [loadedEncryptedFileData subdataWithRange:NSMakeRange(384, kCCBlockSizeAES128)];
    NSData *loadedEncryptedImageData = [loadedEncryptedFileData subdataWithRange:NSMakeRange(384 + kCCBlockSizeAES128, loadedEncryptedFileData.length - 384 - kCCBlockSizeAES128)];

    // Decrypt the symmetric key with the RSA private key
    NSData *decryptedSymmetricKey = [self rsaDecryptData:loadedEncryptedSymmetricKey privateKey:privateKey];
    if (!decryptedSymmetricKey) {
        NSLog(@"Failed to decrypt symmetric key");
        return;
    }

    // Decrypt the image data using the symmetric key
    NSData *decryptedImageData = [self aesDecryptData:loadedEncryptedImageData key:decryptedSymmetricKey iv:loadedIV];
    if (!decryptedImageData) {
        NSLog(@"Failed to decrypt image data");
        return;
    }

    // Save the decrypted image data
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

