#import <Foundation/Foundation.h>

@interface PortScanner : NSObject

- (NSString *)getDeviceIPAddress;
- (void)updateCachedOpenPortsForPort:(int)port;

@property (nonatomic, strong) NSMutableArray<NSString *> *cachedOpenPorts;
@property (nonatomic, strong) dispatch_queue_t scanQueue;
@end

