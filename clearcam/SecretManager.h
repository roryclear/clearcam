#ifndef SecretManager_h
#define SecretManager_h

#import <Foundation/Foundation.h>

@interface SecretManager : NSObject

// Singleton instance
+ (instancetype)sharedManager;

// Retrieve all stored keys from the Keychain
- (NSArray<NSString *> *)getAllStoredKeys;

// Save a key to the Keychain
- (BOOL)saveKey:(NSString *)key error:(NSError **)error;

// Delete a specific key from the Keychain using its identifier
- (BOOL)deleteKeyWithIdentifier:(NSString *)identifier error:(NSError **)error;

// Delete all keys from the Keychain
- (BOOL)deleteAllKeysWithError:(NSError **)error;

@end

#endif /* SecretManager_h */
