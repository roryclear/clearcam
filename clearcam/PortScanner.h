#import <Foundation/Foundation.h>

@interface PortScanner : NSObject

- (void)scanNetworkForPort:(int)port completion:(void (^)(NSArray<NSString *> *openPorts))completion;
- (NSString *)getDeviceIPAddress;

@end

