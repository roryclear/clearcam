#import "PortScanner.h"
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

@implementation PortScanner

// Get the local IP address and subnet mask
- (NSDictionary *)getLocalIPInfo {
    struct ifaddrs *interfaces = NULL;
    struct ifaddrs *temp_addr = NULL;
    NSString *localIP = nil;
    NSString *subnetMask = nil;

    if (getifaddrs(&interfaces) == 0) {
        temp_addr = interfaces;
        while (temp_addr != NULL) {
            if (temp_addr->ifa_addr->sa_family == AF_INET) {
                if ([[NSString stringWithUTF8String:temp_addr->ifa_name] isEqualToString:@"en0"]) {
                    struct sockaddr_in *addr = (struct sockaddr_in *)temp_addr->ifa_addr;
                    struct sockaddr_in *netmask = (struct sockaddr_in *)temp_addr->ifa_netmask;
                    localIP = [NSString stringWithUTF8String:inet_ntoa(addr->sin_addr)];
                    subnetMask = [NSString stringWithUTF8String:inet_ntoa(netmask->sin_addr)];
                }
            }
            temp_addr = temp_addr->ifa_next;
        }
    }
    freeifaddrs(interfaces);

    if (localIP && subnetMask) {
        return @{@"ip": localIP, @"subnet": subnetMask};
    }
    return nil;
}

// Convert IP and subnet mask to a range of possible IPs
- (NSArray<NSString *> *)getIPRangeFromIP:(NSString *)ip subnetMask:(NSString *)mask {
    struct in_addr ipAddr, subnetAddr;
    inet_aton([ip UTF8String], &ipAddr);
    inet_aton([mask UTF8String], &subnetAddr);

    uint32_t network = ipAddr.s_addr & subnetAddr.s_addr;
    uint32_t broadcast = network | ~subnetAddr.s_addr;

    NSMutableArray *ipList = [NSMutableArray array];
    for (uint32_t current = ntohl(network) + 1; current < ntohl(broadcast); current++) {
        struct in_addr addr;
        addr.s_addr = htonl(current);
        [ipList addObject:[NSString stringWithUTF8String:inet_ntoa(addr)]];
    }
    return ipList;
}

// Check if port is open on a given IP
- (BOOL)isPortOpen:(NSString *)ipAddress port:(int)port {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return NO;

    struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    inet_pton(AF_INET, [ipAddress UTF8String], &server.sin_addr);

    // Set socket to non-blocking mode
    fcntl(sock, F_SETFL, O_NONBLOCK);

    int result = connect(sock, (struct sockaddr *)&server, sizeof(server));
    
    if (result == 0) {
        close(sock);
        return YES; // Port is open
    } else if (errno == EINPROGRESS) {
        fd_set fdset;
        struct timeval timeout;
        FD_ZERO(&fdset);
        FD_SET(sock, &fdset);
        timeout.tv_sec = 1;  // 1 second timeout
        timeout.tv_usec = 0;

        if (select(sock + 1, NULL, &fdset, NULL, &timeout) > 0) {
            int so_error;
            socklen_t len = sizeof(so_error);
            getsockopt(sock, SOL_SOCKET, SO_ERROR, &so_error, &len);
            if (so_error == 0) {
                close(sock);
                return YES; // Port is open
            }
        }
    }

    close(sock);
    return NO; // Port is closed
}


- (void)scanNetworkForPort:(int)port completion:(void (^)(void))completion {
    NSDictionary *ipInfo = [self getLocalIPInfo];
    if (!ipInfo) {
        NSLog(@"Failed to get local IP info.");
        if (completion) completion();
        return;
    }

    NSArray<NSString *> *ipList = [self getIPRangeFromIP:ipInfo[@"ip"] subnetMask:ipInfo[@"subnet"]];
    NSLog(@"Scanning %lu IPs for open port %d...", (unsigned long)ipList.count, port);

    BOOL found = NO;
    for (NSString *ip in ipList) {
        NSLog(@"Checking IP: %@", ip);

        if ([self isPortOpen:ip port:port]) {
            NSLog(@"[+] Open port %d found at %@", port, ip);
            found = YES;
        }
    }

    NSLog(@"Scan complete. Found: %@", found ? @"YES" : @"NO");

    if (completion) {
        NSLog(@"Calling completion block...");
        completion();
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
    
    return ipAddress ? ipAddress : @"No IP found";
}


@end

