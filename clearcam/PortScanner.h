#import <Foundation/Foundation.h>

@interface PortScanner : NSObject

- (void)scanNetworkForPort:(int)port completion:(void (^)(void))completion;
- (NSString *)getDeviceIPAddress;

@end

