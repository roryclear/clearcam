#import <Foundation/Foundation.h>

@interface Email : NSObject

+ (instancetype)sharedInstance;
- (void)sendEmailWithImageAtPath:(NSString *)imagePath;

@end
