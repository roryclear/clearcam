#import <Foundation/Foundation.h>

@interface Email : NSObject

+ (instancetype)sharedInstance;
- (void)uploadImageAtPath:(NSString *)imagePath;
- (void)sendNotification;

@end
