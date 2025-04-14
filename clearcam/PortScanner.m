#import "PortScanner.h"
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

@implementation PortScanner

// Initialize cached list and scan queue
- (instancetype)init {
    self = [super init];
    if (self) {
        [self getDeviceIPAddress];
    }
    return self;
}

- (NSString *)getDeviceIPAddress {
    struct ifaddrs *interfaces = NULL;
    struct ifaddrs *temp_addr = NULL;
    NSString *ipAddress = nil;

    if (getifaddrs(&interfaces) == 0) {
        temp_addr = interfaces;
        while (temp_addr != NULL) {
            if (temp_addr->ifa_addr->sa_family == AF_INET) {
                // Check for Wi-Fi (en0)
                if ([[NSString stringWithUTF8String:temp_addr->ifa_name] isEqualToString:@"en0"]) {
                    ipAddress = [NSString stringWithUTF8String:inet_ntoa(((struct sockaddr_in *)temp_addr->ifa_addr)->sin_addr)];
                    break; // Found the IP, exit loop
                }
            }
            temp_addr = temp_addr->ifa_next;
        }
    }
    freeifaddrs(interfaces);
    
    if (ipAddress) {
        [[NSUserDefaults standardUserDefaults] setObject:ipAddress forKey:@"DeviceIPAddress"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
    return ipAddress ? ipAddress : @"No IP found";
}

@end
