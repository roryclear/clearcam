#import <Foundation/Foundation.h>
#import <Security/Security.h>

@interface PGP : NSObject

- (instancetype)init;
- (void)encryptImageWithPublicKey:(SecKeyRef)publicKey filePath:(NSString *)filePath;
- (void)decryptImageWithPrivateKey:(SecKeyRef)privateKey filePath:(NSString *)encryptedFilePath;

@end
