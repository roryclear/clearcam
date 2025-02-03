#import <Foundation/Foundation.h>

@interface PortScanner : NSObject

- (void)scanNetworkForPort:(int)port completion:(void (^)(NSArray<NSString *> *openPorts))completion;
- (NSString *)getDeviceIPAddress;
- (void)updateCachedOpenPortsForPort:(int)port;

@property (nonatomic, strong) NSMutableArray<NSString *> *cachedOpenPorts;
@property (nonatomic, strong) dispatch_queue_t scanQueue;
@end

