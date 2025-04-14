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
        _cachedOpenPorts = [NSMutableArray array];
        _scanQueue = dispatch_queue_create("com.example.PortScannerQueue", DISPATCH_QUEUE_SERIAL);
    }
    [self updateCachedOpenPortsForPort:80];
    return self;
}

- (void)updateCachedOpenPortsForPort:(int)port {
    dispatch_async(self.scanQueue, ^{
        [self throttledScanNetworkForPort:port withBatchSize:10 completion:^(NSArray<NSString *> *openPorts) {
            @synchronized (self.cachedOpenPorts) {
                [self.cachedOpenPorts removeAllObjects];
                [self.cachedOpenPorts addObjectsFromArray:openPorts];
            }
        }];
    });
}

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

- (void)throttledScanNetworkForPort:(int)port withBatchSize:(int)batchSize completion:(void (^)(NSArray<NSString *> *openPorts))completion {
    NSDictionary *ipInfo = [self getLocalIPInfo];
    if (!ipInfo) {
        if (completion) completion(@[]);
        return;
    }

    NSString *deviceIP = [self getDeviceIPAddress];
    NSMutableArray<NSString *> *foundIPs = [NSMutableArray array];
    
    // Check device IP first
    if (deviceIP && [self isAppRunningAtIP:deviceIP port:port]) {
        [foundIPs addObject:deviceIP];
    }

    NSArray<NSString *> *ipList = [self getIPRangeFromIP:ipInfo[@"ip"] subnetMask:ipInfo[@"subnet"]];
    dispatch_group_t scanGroup = dispatch_group_create();
    dispatch_queue_t scanQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0);
    
    for (NSInteger i = 0; i < ipList.count; i += batchSize) {
        NSRange range = NSMakeRange(i, MIN(batchSize, ipList.count - i));
        NSArray *batch = [ipList subarrayWithRange:range];
        
        for (NSString *ip in batch) {
            // Skip device IP since we already checked it
            if ([ip isEqualToString:deviceIP]) {
                continue;
            }
            
            dispatch_group_enter(scanGroup);
            dispatch_async(scanQueue, ^{
                // First check if port is open
                if ([self isPortOpen:ip port:port]) {
                    // Then verify if it's our app
                    if ([self isAppRunningAtIP:ip port:port]) {
                        @synchronized(self) {
                            [foundIPs addObject:ip];
                        }
                    }
                }
                dispatch_group_leave(scanGroup);
            });
        }
        
        // Throttle by waiting for the batch to complete before starting the next batch
        dispatch_group_wait(scanGroup, DISPATCH_TIME_FOREVER);
    }
    
    dispatch_group_notify(scanGroup, dispatch_get_main_queue(), ^{
        if (completion) {
            completion([foundIPs copy]);
        }
    });
}

// Helper method to verify if our app is running at the IP:port
- (BOOL)isAppRunningAtIP:(NSString *)ip port:(int)port {
    NSString *urlString = [NSString stringWithFormat:@"http://%@:%d/get-devices", ip, port];
    NSURL *url = [NSURL URLWithString:urlString];
    
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block BOOL isOurApp = NO;
    
    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithURL:url completionHandler:^(NSData * _Nullable data, NSURLResponse * _Nullable response, NSError * _Nullable error) {
        if (!error && data) {
            // Check if the response matches what your app would return
            id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
            if ([json isKindOfClass:[NSArray class]]) {
                isOurApp = YES;
            }
        }
        dispatch_semaphore_signal(semaphore);
    }];
    
    [task resume];
    
    // Wait with a 2-second timeout
    dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC);
    dispatch_semaphore_wait(semaphore, timeout);
    
    return isOurApp;
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
        // Store in NSUserDefaults
        [[NSUserDefaults standardUserDefaults] setObject:ipAddress forKey:@"DeviceIPAddress"];
        [[NSUserDefaults standardUserDefaults] synchronize]; // Ensure it's saved immediately
    }
    
    return ipAddress ? ipAddress : @"No IP found";
}

@end
