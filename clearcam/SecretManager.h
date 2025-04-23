#ifndef SecretManager_h
#define SecretManager_h

#import <Foundation/Foundation.h>

@interface SecretManager : NSObject

// Singleton instance
+ (instancetype)sharedManager;

// Encryption Key Management (Single Key)
- (BOOL)saveEncryptionKey:(NSString *)key error:(NSError **)error;
- (NSString *)getEncryptionKey;

// Decryption Key Management (Multiple Keys)
- (BOOL)saveDecryptionKey:(NSString *)key withIdentifier:(NSString *)identifier error:(NSError **)error;
- (NSArray<NSString *> *)getAllDecryptionKeys;

// Generic Key Management
- (BOOL)deleteKeyWithIdentifier:(NSString *)identifier error:(NSError **)error;
- (BOOL)deleteAllKeysWithError:(NSError **)error;

- (NSData *)decryptData:(NSData *)encryptedData withKey:(NSString *)key;
- (NSData *)encryptData:(NSData *)data withKey:(NSString *)key;
//TODO remove these once they work
- (NSData *)decryptLiveSegment:(NSData *)encryptedDataWithIv withKey:(NSString *)key;
- (NSData *)encryptLiveSegment:(NSData *)data withKey:(NSString *)key;

@end

#endif /* SecretManager_h */

