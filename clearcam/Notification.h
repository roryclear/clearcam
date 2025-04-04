#import <Foundation/Foundation.h>

@interface notification : NSObject

+ (instancetype)sharedInstance;
- (void)uploadImageAtPath:(NSString *)imagePath;
- (void)sendNotification;

@end
