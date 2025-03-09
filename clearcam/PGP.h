#import <Foundation/Foundation.h>
#import <Security/Security.h>

@interface PGP : NSObject

@property (nonatomic, readonly) SecKeyRef privateKey;
@property (nonatomic, readonly) SecKeyRef publicKey;

- (instancetype)init;
- (void)encryptImageWithPublicKey:(NSString *)filePath;
- (void)decryptImageWithPrivateKey:(NSString *)encryptedFilePath;

@end
