#import "PortScanner.h"
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

@interface PortScanner ()
@property (nonatomic, strong) NSTimer *ipCheckTimer;
@property (nonatomic, strong) NSString *lastKnownIPAddress;
@end

@implementation PortScanner

- (instancetype)init {
    self = [super init];
    if (self) {
        [self startPeriodicIPCheck];
    }
    return self;
}

- (void)startPeriodicIPCheck {
    [self checkAndUpdateIPAddress];
    self.ipCheckTimer = [NSTimer scheduledTimerWithTimeInterval:5.0
                                                         target:self
                                                       selector:@selector(checkAndUpdateIPAddress)
                                                       userInfo:nil
                                                        repeats:YES];
}

- (void)checkAndUpdateIPAddress {
    NSString *currentIP = [self getDeviceIPAddress];
    
    if (currentIP && ![currentIP isEqualToString:self.lastKnownIPAddress]) {
        self.lastKnownIPAddress = currentIP;
        [[NSUserDefaults standardUserDefaults] setObject:currentIP forKey:@"DeviceIPAddress"];
        [[NSUserDefaults standardUserDefaults] synchronize];
        [[NSNotificationCenter defaultCenter] postNotificationName:@"DeviceIPAddressDidChangeNotification"
                                                            object:nil];
    }
}

- (NSString *)getDeviceIPAddress {
    struct ifaddrs *interfaces = NULL;
    struct ifaddrs *temp_addr = NULL;
    NSString *ipAddress = nil;

    if (getifaddrs(&interfaces) == 0) {
        temp_addr = interfaces;
        while (temp_addr != NULL) {
            if (temp_addr->ifa_addr->sa_family == AF_INET) {
                if ([[NSString stringWithUTF8String:temp_addr->ifa_name] isEqualToString:@"en0"]) {
                    ipAddress = [NSString stringWithUTF8String:
                        inet_ntoa(((struct sockaddr_in *)temp_addr->ifa_addr)->sin_addr)];
                    break;
                }
            }
            temp_addr = temp_addr->ifa_next;
        }
    }
    freeifaddrs(interfaces);
    if (ipAddress && ipAddress.length > 0) {
        return [NSString stringWithFormat:@"http://%@", ipAddress];
    } else {
        return @"Waiting for Wi-Fi";
    }
}

- (void)dealloc {
    [self.ipCheckTimer invalidate];
    self.ipCheckTimer = nil;
}
@end
