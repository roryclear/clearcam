#import "FileServer.h"
#import <sys/types.h>
#import <sys/socket.h>
#import <netinet/in.h>
#import <arpa/inet.h>
#import <unistd.h>
#import <signal.h>
#import <errno.h>
#import <AVFoundation/AVFoundation.h>
#import "SettingsManager.h"
#import "PortScanner.h"
#import "AppDelegate.h"
#import "StoreManager.h"
#import <CoreData/CoreData.h>
#import "SecretManager.h"

@interface FileServer ()
@property (nonatomic, strong) NSString *basePath;
@property (nonatomic, strong) NSMutableDictionary *durationCache;
@property (nonatomic, assign) int serverSocket;//todo
@property (nonatomic, assign) BOOL isServerRunning;
@end

@implementation FileServer

+ (instancetype)sharedInstance {
    static FileServer *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _serverSocket = -1;
        _isServerRunning = NO;
        
        // Add KVO observer for stream_via_wifi_enabled
        [[NSUserDefaults standardUserDefaults] addObserver:self
                                               forKeyPath:@"stream_via_wifi_enabled"
                                                  options:NSKeyValueObservingOptionNew
                                                  context:nil];
    }
    return self;
}

- (void)start {
    AppDelegate *appDelegate = (AppDelegate *)[[UIApplication sharedApplication] delegate];
    self.context = appDelegate.persistentContainer.viewContext;
    self.segment_length = 60;
    self.scanner = [[PortScanner alloc] init];
    self.last_req_time = [NSDate now];
    self.basePath = [self getDocumentsDirectory];
    self.durationCache = [[NSMutableDictionary alloc] init];
    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"stream_via_wifi_enabled"]) [self startServer];
    if ([[NSUserDefaults standardUserDefaults] objectForKey:@"stream_via_wifi_enabled"] == nil) {
        [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"stream_via_wifi_enabled"];
    }
}

- (void)dealloc {
    [self stopServer];
    [[NSUserDefaults standardUserDefaults] removeObserver:self
                                              forKeyPath:@"stream_via_wifi_enabled"];
}

- (void)startServer {
    if (self.isServerRunning) return;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            signal(SIGPIPE, SIG_IGN);
            self.serverSocket = socket(AF_INET, SOCK_STREAM, 0);
            if (self.serverSocket == -1) return;
            struct sockaddr_in serverAddr;
            memset(&serverAddr, 0, sizeof(serverAddr));
            serverAddr.sin_family = AF_INET;
            serverAddr.sin_addr.s_addr = INADDR_ANY;
            serverAddr.sin_port = htons(80);

            int opt = 1;
            setsockopt(self.serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            
            if (bind(self.serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1) {
                close(self.serverSocket);
                self.serverSocket = -1;
                return;
            }
            
            if (listen(self.serverSocket, 5) == -1) {
                close(self.serverSocket);
                self.serverSocket = -1;
                return;
            }
            
            self.isServerRunning = YES;
            while (self.isServerRunning) {
                int clientSocket = accept(self.serverSocket, NULL, NULL);
                if (clientSocket != -1) {
                    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                        [self handleClientRequest:clientSocket withBasePath:self.basePath];
                        close(clientSocket);
                    });
                }
            }
            
        } @catch (NSException *exception) {
            self.isServerRunning = NO;
            if (self.serverSocket != -1) {
                close(self.serverSocket);
                self.serverSocket = -1;
            }
        }
    });
}

- (void)stopServer {
    if (!self.isServerRunning) {
        return;  // Server already stopped
    }
    
    self.isServerRunning = NO;
    if (self.serverSocket != -1) {
        close(self.serverSocket);
        self.serverSocket = -1;
    }
}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary *)change
                       context:(void *)context {
    if ([keyPath isEqualToString:@"stream_via_wifi_enabled"]) {
        BOOL streamViaWiFiEnabled = [change[NSKeyValueChangeNewKey] boolValue];
        if (streamViaWiFiEnabled) {
            [self startServer];
        } else {
            [self stopServer];
        }
    }
}

- (NSString *)getDocumentsDirectory {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    return [paths firstObject];
}

- (NSArray *)fetchAndProcessSegmentsFromCoreDataForDateParam:(NSString *)dateParam
                                                      start:(double)startTime
                                                    context:(NSManagedObjectContext *)context {
    if (!context) return @[];

    __block NSArray *processedSegments = @[];

    [context performBlockAndWait:^{
        NSError *error = nil;

        // Fetch the DayEntity to get its ID
        NSFetchRequest *dayFetchRequest = [[NSFetchRequest alloc] initWithEntityName:@"DayEntity"];
        dayFetchRequest.predicate = [NSPredicate predicateWithFormat:@"date == %@", dateParam];
        dayFetchRequest.fetchLimit = 1; // We only need one DayEntity

        NSArray *fetchedDays = [context executeFetchRequest:dayFetchRequest error:&error];
        if (error || fetchedDays.count == 0) return;

        NSManagedObject *dayEntity = fetchedDays.firstObject;

        // Fetch segments with a filter based on timeStamp
        NSFetchRequest *segmentFetchRequest = [[NSFetchRequest alloc] initWithEntityName:@"SegmentEntity"];
        segmentFetchRequest.predicate = [NSPredicate predicateWithFormat:@"day == %@ AND timeStamp >= %f", dayEntity, startTime];
        segmentFetchRequest.sortDescriptors = @[[NSSortDescriptor sortDescriptorWithKey:@"timeStamp" ascending:YES]]; // Consistent ordering
        segmentFetchRequest.propertiesToFetch = @[@"url", @"timeStamp", @"duration", @"orientation"];
        segmentFetchRequest.resultType = NSDictionaryResultType; // Return dictionaries directly

        NSArray *fetchedSegments = [context executeFetchRequest:segmentFetchRequest error:&error];
        if (error) return;

        processedSegments = fetchedSegments;
    }];

    return processedSegments;
}
- (NSArray *)fetchFramesWithURLsFromCoreDataForDateParam:(NSString *)dateParam
                                                   start:(NSInteger)start
                                                 context:(NSManagedObjectContext *)context {
    if (!context) return @[];

    __block NSArray *copiedSegments = @[];

    [context performBlockAndWait:^{
        NSError *error = nil;

        // Fetch the DayEntity for the given date
        NSFetchRequest *dayFetchRequest = [[NSFetchRequest alloc] initWithEntityName:@"DayEntity"];
        dayFetchRequest.predicate = [NSPredicate predicateWithFormat:@"date == %@", dateParam];

        NSArray *fetchedDays = [context executeFetchRequest:dayFetchRequest error:&error];

        if (error) return;
        if (fetchedDays.count == 0) return;

        NSManagedObject *dayEntity = fetchedDays.firstObject;
        NSOrderedSet *segments = [dayEntity valueForKey:@"segments"];

        // Slice the segments based on the 'start' parameter to avoid fetching everything
        if (start >= segments.count) return;

        copiedSegments = [[segments array] subarrayWithRange:NSMakeRange(start, segments.count - start)];
    }];

    // Process outside of performBlockAndWait to avoid blocking Core Data
    NSMutableArray *framesWithURLs = [NSMutableArray array];

    for (NSManagedObject *segment in copiedSegments) {
        NSString *url = [segment valueForKey:@"url"];
        NSArray *frames = [[segment valueForKey:@"frames"] array]; // Copy frames

        for (NSManagedObject *frame in frames) {
            double frameTimeStamp = [[frame valueForKey:@"frame_timeStamp"] doubleValue];
            double aspectRatio = [[frame valueForKey:@"aspect_ratio"] doubleValue];
            int res = [[frame valueForKey:@"res"] intValue];

            NSMutableArray *squareDicts = [NSMutableArray array];
            NSArray *squares = [[frame valueForKey:@"squares"] array]; // Copy squares

            for (NSManagedObject *square in squares) {
                double originX = [[square valueForKey:@"originX"] doubleValue];
                double originY = [[square valueForKey:@"originY"] doubleValue];
                double bottomRightX = [[square valueForKey:@"bottomRightX"] doubleValue];
                double bottomRightY = [[square valueForKey:@"bottomRightY"] doubleValue];
                int classIndex = [[square valueForKey:@"classIndex"] intValue];

                NSDictionary *squareDict = @{
                    @"originX": @(originX),
                    @"originY": @(originY),
                    @"bottomRightX": @(bottomRightX),
                    @"bottomRightY": @(bottomRightY),
                    @"classIndex": @(classIndex)
                };

                [squareDicts addObject:squareDict];
            }

            NSDictionary *frameDict = @{
                @"url": url,  // Include the segment's URL
                @"frame_timeStamp": @(frameTimeStamp),
                @"aspect_ratio": @(aspectRatio),
                @"res": @(res),
                @"squares": squareDicts
            };

            [framesWithURLs addObject:frameDict];
        }
    }
    return framesWithURLs;
}

- (void)sendJson200:(NSArray *)array toClient:(int)clientSocket {
    NSError *error;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:array options:0 error:&error];
    NSString *httpHeader = [NSString stringWithFormat:@"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %lu\r\n\r\n", (unsigned long)jsonData.length];
    send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
    send(clientSocket, jsonData.bytes, jsonData.length, 0);
}

- (void)handleClientRequest:(int)clientSocket withBasePath:(NSString *)basePath {
    char requestBuffer[1024];
    ssize_t bytesRead = recv(clientSocket, requestBuffer, sizeof(requestBuffer) - 1, 0);
    if (bytesRead < 0) return;
    requestBuffer[bytesRead] = '\0';
    NSString *request = [NSString stringWithUTF8String:requestBuffer];
    NSRange range = [request rangeOfString:@"GET /"];
    if (range.location == NSNotFound) {
        dprintf(clientSocket, "HTTP/1.1 400 Bad Request\r\n\r\n");
        return;
    }
    NSString *filePath = [[request substringFromIndex:NSMaxRange(range)] componentsSeparatedByString:@" "][0];
    filePath = [filePath stringByRemovingPercentEncoding];
    if ([filePath isEqualToString:@"/"]) filePath = @"";
    
    NSMutableDictionary<NSString *, NSString *> *queryParams = [NSMutableDictionary dictionary];
    NSRange queryRange = [filePath rangeOfString:@"?"];
    if (queryRange.location != NSNotFound) {
        NSString *queryString = [filePath substringFromIndex:queryRange.location + 1];
        NSArray<NSString *> *queryItems = [queryString componentsSeparatedByString:@"&"];
        
        for (NSString *item in queryItems) {
            NSArray<NSString *> *keyValue = [item componentsSeparatedByString:@"="];
            if (keyValue.count == 2) {
                queryParams[keyValue[0]] = keyValue[1];
            }
        }
    }
    
    if ([filePath hasPrefix:@"multicamera"] || [filePath hasPrefix:@"events"]) {
        NSString *fileName = [filePath hasPrefix:@"multicamera"] ? @"multicamera.html" : @"events.html";
        NSString *playerFilePath = [[NSBundle mainBundle] pathForResource:[fileName stringByDeletingPathExtension] ofType:@"html"];
        
        if (playerFilePath && [[NSFileManager defaultManager] fileExistsAtPath:playerFilePath]) {
            FILE *file = fopen([playerFilePath UTF8String], "rb");
            if (file) {
                fseek(file, 0, SEEK_END);
                NSUInteger fileSize = ftell(file);
                fseek(file, 0, SEEK_SET);
                dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
                dprintf(clientSocket, "Content-Type: text/html\r\n");
                dprintf(clientSocket, "Content-Length: %lu\r\n", fileSize);
                dprintf(clientSocket, "\r\n");
                [self sendFileData:file toSocket:clientSocket withContentLength:fileSize];
                fclose(file);
                return;
            }
        }
        dprintf(clientSocket, "HTTP/1.1 500 Internal Server Error\r\n\r\n");
        return;
    }
    if ([filePath hasPrefix:@"get-segments"]) {
        self.last_req_time = [NSDate now];
        if (self.segment_length == 60) {
            self.segment_length = 1;
            sleep(3);//todo hack, not good because clips wait too
        }
                
        if (!queryParams[@"date"]) {
            NSString *httpHeader = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n";
            NSString *errorMessage = @"{\"error\": \"Missing or invalid date parameter\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [errorMessage UTF8String], errorMessage.length, 0);
            return;
        }
        
        double startTime = queryParams[@"start"] ? [queryParams[@"start"] doubleValue] : 0;
        NSArray *segmentsForDate = [self fetchAndProcessSegmentsFromCoreDataForDateParam:queryParams[@"date"]
                                                                                  start:startTime
                                                                                context:self.context];
        
        [self sendJson200:segmentsForDate toClient:clientSocket];
    }

    if ([filePath hasPrefix:@"download"]) {
        if (!queryParams[@"start"] || !queryParams[@"end"]) {
            NSString *errorResponse = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"Missing or invalid start/end parameter\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        NSTimeInterval startTimeStamp = [queryParams[@"start"] doubleValue];
        NSTimeInterval endTimeStamp = [queryParams[@"end"] doubleValue];
        
        NSString *outputPath = [self processVideoDownloadWithLowRes:NO
                                                        startTime:startTimeStamp
                                                          endTime:endTimeStamp
                                                         context:self.context];
        
        if (!outputPath) {
            NSString *errorResponse = @"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"No video found for the specified time range\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }
        
        // Send the file
        FILE *mergedFile = fopen([outputPath UTF8String], "rb");
        if (!mergedFile) {
            NSString *errorResponse = @"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"Failed to open video file\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        NSFileManager *fileManager = [NSFileManager defaultManager];
        NSDictionary *fileAttributes = [fileManager attributesOfItemAtPath:outputPath error:nil];
        NSUInteger fileSize = [fileAttributes[NSFileSize] unsignedIntegerValue];

        dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
        dprintf(clientSocket, "Content-Type: video/mp4\r\n");
        dprintf(clientSocket, "Content-Disposition: attachment; filename=\"%s-%s.mp4\"\r\n",
                [[^{ NSDateFormatter *f = [NSDateFormatter new]; f.dateFormat = @"yyyy-MM-dd_HH-mm-ss"; return f; }() stringFromDate:[NSDate dateWithTimeIntervalSince1970:startTimeStamp]] UTF8String],
                [[^{ NSDateFormatter *f = [NSDateFormatter new]; f.dateFormat = @"yyyy-MM-dd_HH-mm-ss"; return f; }() stringFromDate:[NSDate dateWithTimeIntervalSince1970:endTimeStamp]] UTF8String]);
        dprintf(clientSocket, "Content-Length: %lu\r\n", (unsigned long)fileSize);
        dprintf(clientSocket, "Accept-Ranges: bytes\r\n");
        dprintf(clientSocket, "\r\n");
        
        char buffer[64 * 1024];
        size_t bytesRead;
        while ((bytesRead = fread(buffer, 1, sizeof(buffer), mergedFile)) > 0) {
            ssize_t bytesSent = send(clientSocket, buffer, bytesRead, 0);
            if (bytesSent < 0) break;
        }
        fclose(mergedFile);
        
        // Clean up
        [[NSFileManager defaultManager] removeItemAtPath:outputPath error:nil];
    }
    
    if ([filePath hasPrefix:@"get-frames"]) {
        if (!queryParams[@"url"]) {
            NSString *httpHeader = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n";
            NSString *errorMessage = @"{\"error\": \"Missing or invalid url parameter\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [errorMessage UTF8String], errorMessage.length, 0);
            return;
        }        
        [self sendJson200:[self fetchFramesForURL:queryParams[@"url"] context:self.context] toClient:clientSocket];
    }
    
    if ([filePath hasPrefix:@"get-events"]) [self sendJson200: [self fetchEventDataFromCoreData:self.context] toClient:clientSocket];

    if ([filePath hasPrefix:@"delete-event"]) {
        NSURLComponents *components = [NSURLComponents componentsWithString:[NSString stringWithFormat:@"http://localhost/%@", filePath]];
        NSString *eventTimeStamp = nil;
        NSString *eventTimeStamp_end = nil;
        
        for (NSURLQueryItem *item in components.queryItems) {
            if ([item.name isEqualToString:@"timeStamp"]) {
                eventTimeStamp = item.value;
            }
            if ([item.name isEqualToString:@"end"]) {
                eventTimeStamp_end = item.value;
            }
        }
        
        if(!eventTimeStamp_end) eventTimeStamp_end = eventTimeStamp;
        
        if (!eventTimeStamp) {
            NSString *httpHeader = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n";
            NSString *errorMessage = @"{\"error\": \"Missing timeStamp parameter\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [errorMessage UTF8String], errorMessage.length, 0);
            return;
        }
        
        
        BOOL success = [self deleteEventsBetweenStartTimeStamp:[eventTimeStamp doubleValue] andEndTimeStamp:[eventTimeStamp_end doubleValue] ];
        
        if (success) {
            NSString *httpHeader = @"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n";
            NSString *successMessage = @"{\"success\": \"Event and associated image deleted\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [successMessage UTF8String], successMessage.length, 0);
        } else {
            NSString *httpHeader = @"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n";
            NSString *errorMessage = @"{\"error\": \"Event not found\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [errorMessage UTF8String], errorMessage.length, 0);
        }
    }
    
    NSString *fullPath = [basePath stringByAppendingPathComponent:filePath];
    queryRange = [fullPath rangeOfString:@"?"];
    if (queryRange.location != NSNotFound) {
        fullPath = [fullPath substringToIndex:queryRange.location];
    }
    
    BOOL isDirectory = YES;
    if (![[NSFileManager defaultManager] fileExistsAtPath:fullPath isDirectory:&isDirectory]) {
        dprintf(clientSocket, "HTTP/1.1 404 Not Found\r\n\r\n");
        return;
    }
    
    
    if (isDirectory) { //todo, change url maybe
        NSString *playerFilePath = [[NSBundle mainBundle] pathForResource:@"player" ofType:@"html"];
        if (playerFilePath && [[NSFileManager defaultManager] fileExistsAtPath:playerFilePath]) {
            FILE *file = fopen([playerFilePath UTF8String], "rb");
            if (file) {
                fseek(file, 0, SEEK_END);
                NSUInteger fileSize = ftell(file);
                fseek(file, 0, SEEK_SET);
                dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
                dprintf(clientSocket, "Content-Type: text/html\r\n");
                dprintf(clientSocket, "Content-Length: %lu\r\n", fileSize);
                dprintf(clientSocket, "\r\n");
                [self sendFileData:file toSocket:clientSocket withContentLength:fileSize];
                fclose(file);
                return;
            }
        }
        dprintf(clientSocket, "HTTP/1.1 500 Internal Server Error\r\n\r\n");
        return;
    }
    
    FILE *file = fopen([fullPath UTF8String], "rb");
    if (!file) {
        dprintf(clientSocket, "HTTP/1.1 500 Internal Server Error\r\n\r\n");
        return;
    }
    fseek(file, 0, SEEK_END);
    NSUInteger fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    NSString *fileExtension = [[fullPath pathExtension] lowercaseString];
    
    // Check if the file exists in the app bundle
    NSString *bundlePath = [[NSBundle mainBundle] pathForResource:[fullPath stringByDeletingPathExtension] ofType:fileExtension];
    if (bundlePath) {
        fullPath = bundlePath;
    }
    
    if ([fileExtension isEqualToString:@"mp4"]) {
        range = [request rangeOfString:@"Range: bytes="];
        if (range.location != NSNotFound) {
            range = [request rangeOfString:@"bytes="];
            NSString *byteRange = [request substringFromIndex:NSMaxRange(range)];
            byteRange = [byteRange stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            NSArray<NSString *> *rangeParts = [byteRange componentsSeparatedByString:@"-"];
            NSUInteger start = [rangeParts[0] integerValue];
            NSUInteger end = (rangeParts.count > 1 && rangeParts[1].length > 0) ? [rangeParts[1] integerValue] : fileSize - 1;

            if (start >= fileSize || end >= fileSize || start > end) {
                dprintf(clientSocket, "HTTP/1.1 416 Requested Range Not Satisfiable\r\n");
                dprintf(clientSocket, "Content-Range: bytes */%lu\r\n", fileSize);
                dprintf(clientSocket, "\r\n");
            } else {
                fseek(file, start, SEEK_SET);
                NSUInteger contentLength = end - start + 1;
                dprintf(clientSocket, "HTTP/1.1 206 Partial Content\r\n");
                dprintf(clientSocket, "Content-Type: video/mp4\r\n");
                dprintf(clientSocket, "Content-Range: bytes %lu-%lu/%lu\r\n", start, end, fileSize);
                dprintf(clientSocket, "Content-Length: %lu\r\n", contentLength);
                dprintf(clientSocket, "Accept-Ranges: bytes\r\n");
                dprintf(clientSocket, "\r\n");
                [self sendFileData:file toSocket:clientSocket withContentLength:contentLength];
            }
        }
    } else if ([fileExtension isEqualToString:@"html"]) {
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
        dprintf(clientSocket, "Content-Type: text/html\r\n");
        dprintf(clientSocket, "Content-Length: %lu\r\n", fileSize);
        dprintf(clientSocket, "\r\n");
        [self sendFileData:file toSocket:clientSocket withContentLength:fileSize];
    } else if ([fileExtension isEqualToString:@"txt"]) {
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
        dprintf(clientSocket, "Content-Type: text/plain\r\n");
        dprintf(clientSocket, "Content-Length: %lu\r\n", fileSize);
        dprintf(clientSocket, "\r\n");
        [self sendFileData:file toSocket:clientSocket withContentLength:fileSize];
    } else if ([fileExtension isEqualToString:@"jpg"]) {
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
        dprintf(clientSocket, "Content-Type: image/jpg\r\n");
        dprintf(clientSocket, "Content-Length: %lu\r\n", fileSize);
        dprintf(clientSocket, "\r\n");
        [self sendFileData:file toSocket:clientSocket withContentLength:fileSize];
    }

    fclose(file);
}

- (BOOL)deleteEventsBetweenStartTimeStamp:(NSTimeInterval)startTimeStamp andEndTimeStamp:(NSTimeInterval)endTimeStamp {
    __block BOOL success = NO;
    const int maxRetries = 3;
    int attempt = 0;
    
    if (!self.context) return NO;
    
    // Verify entity exists in model
    NSEntityDescription *entity = [NSEntityDescription entityForName:@"EventEntity"
                                             inManagedObjectContext:self.context];
    if (!entity) return NO;
    
    // Validate timestamp range
    if (startTimeStamp > endTimeStamp) return NO;
    
    // Get the app's Documents directory and images folder
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths firstObject];
    NSString *imagesDirectory = [documentsDirectory stringByAppendingPathComponent:@"images"];
    
    while (attempt < maxRetries && !success) {
        attempt++;
        [self.context performBlockAndWait:^{
            NSFetchRequest *fetchRequest = [[NSFetchRequest alloc] init];
            if (!fetchRequest) {
                success = NO;
                return;
            }
            
            [fetchRequest setEntity:entity];
            double epsilon = 1.0; // 1-second buffer on both ends
            NSPredicate *predicate = [NSPredicate predicateWithFormat:@"(timeStamp >= %lf) AND (timeStamp <= %lf)",
                                    startTimeStamp - epsilon, endTimeStamp + epsilon];
            [fetchRequest setPredicate:predicate];
            
            // Execute fetch with error handling
            NSError *fetchError = nil;
            NSArray *events = nil;
            @try {
                events = [self.context executeFetchRequest:fetchRequest error:&fetchError];
            }
            @catch (NSException *exception) {
                success = NO;
                return;
            }
            
            if (fetchError) {
                success = NO;
                return;
            }
            
            if (!events) {
                success = YES; // Treat as success if no objects to delete
                return;
            }
            
            if (events.count == 0) {
                success = YES; // No events in range, still a success
                return;
            }
            
            // Delete events and their associated images
            NSFileManager *fileManager = [NSFileManager defaultManager];
            for (NSManagedObject *event in events) {
                NSNumber *timeStampNumber = [event valueForKey:@"timeStamp"];
                if (!timeStampNumber) {
                    [self.context deleteObject:event];
                    continue;
                }
                
                NSTimeInterval timeStamp = [timeStampNumber doubleValue];
                long long roundedTimestamp = (long long)floor(timeStamp); // Floor to integer
                NSString *imageFileName = [NSString stringWithFormat:@"%lld", roundedTimestamp];
                NSString *smallImageFilePath = [imagesDirectory stringByAppendingPathComponent:[imageFileName stringByAppendingString:@"_small.jpg"]];

                NSError *fileError = nil;
                if ([fileManager fileExistsAtPath:smallImageFilePath]) [fileManager removeItemAtPath:smallImageFilePath error:&fileError];
                [self.context deleteObject:event];
            }
            
            // Save changes
            if ([self.context hasChanges]) {
                NSError *saveError = nil;
                @try {
                    success = [self.context save:&saveError];
                }
                @catch (NSException *exception) {
                    success = NO;
                }
            } else {
                success = YES; // No changes to save, but this shouldn't happen if events were deleted
            }
        }];
        if (!success) [NSThread sleepForTimeInterval:0.1];
    }
    return success;
}

- (NSArray *)fetchFramesForURL:(NSString *)url context:(NSManagedObjectContext *)context {
    if (!context) return @[];
    __block NSArray *framesForURL = @[];

    [context performBlockAndWait:^{
        NSError *error = nil;

        // Fetch the single segment with frames and squares pre-fetched
        NSFetchRequest *segmentFetchRequest = [[NSFetchRequest alloc] initWithEntityName:@"SegmentEntity"];
        segmentFetchRequest.predicate = [NSPredicate predicateWithFormat:@"url == %@", url];
        segmentFetchRequest.fetchLimit = 1; // Only one segment exists per URL
        segmentFetchRequest.relationshipKeyPathsForPrefetching = @[@"frames", @"frames.squares"];
        segmentFetchRequest.returnsObjectsAsFaults = NO;

        NSArray *segments = [context executeFetchRequest:segmentFetchRequest error:&error];
        if (error) return;
        if (segments.count == 0) return;
        NSManagedObject *segment = segments.firstObject;
        NSArray *frames = [[segment valueForKey:@"frames"] array];
        NSMutableArray *tempFrames = [NSMutableArray arrayWithCapacity:frames.count];

        // Process frames and squares
        for (NSManagedObject *frame in frames) {
            double frameTimeStamp = [[frame valueForKey:@"frame_timeStamp"] doubleValue];
            double aspectRatio = [[frame valueForKey:@"aspect_ratio"] doubleValue];
            int res = [[frame valueForKey:@"res"] intValue];

            NSArray *squares = [[frame valueForKey:@"squares"] array];
            NSMutableArray *squareDicts = [NSMutableArray arrayWithCapacity:squares.count];

            for (NSManagedObject *square in squares) {
                NSNumber *originXNum = [square valueForKey:@"originX"];
                NSNumber *originYNum = [square valueForKey:@"originY"];
                NSNumber *bottomRightXNum = [square valueForKey:@"bottomRightX"];
                NSNumber *bottomRightYNum = [square valueForKey:@"bottomRightY"];
                NSNumber *classIndexNum = [square valueForKey:@"classIndex"];

                [squareDicts addObject:@{
                    @"originX": @(originXNum ? [originXNum doubleValue] : 0.0),
                    @"originY": @(originYNum ? [originYNum doubleValue] : 0.0),
                    @"bottomRightX": @(bottomRightXNum ? [bottomRightXNum doubleValue] : 0.0),
                    @"bottomRightY": @(bottomRightYNum ? [bottomRightYNum doubleValue] : 0.0),
                    @"classIndex": @(classIndexNum ? [classIndexNum intValue] : 0)
                }];
            }

            [tempFrames addObject:@{
                @"url": url,
                @"frame_timeStamp": @(frameTimeStamp),
                @"aspect_ratio": @(aspectRatio),
                @"res": @(res),
                @"squares": squareDicts
            }];
        }

        framesForURL = [tempFrames copy];
    }];
    return framesForURL;
}

+ (void)sendDeviceTokenToServer {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    NSString *deviceToken = [defaults stringForKey:@"device_token"];
    if (!deviceToken || deviceToken.length == 0) return;
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken || sessionToken.length == 0) return;
    [FileServer performPostRequestWithURL:@"https://rors.ai/add_device"
                                       method:@"POST"
                                  contentType:@"application/json"
                                         body:@{@"device_token": deviceToken, @"session_token": sessionToken}
                            completionHandler:^(NSData *data, NSHTTPURLResponse *response, NSError *error) { if (error) return; }];
}

+ (void)performPostRequestWithURL:(NSString *)urlString
                           method:(NSString *)method
                      contentType:(NSString *)contentType
                             body:(id)body
                completionHandler:(void (^)(NSData *data, NSHTTPURLResponse *response, NSError *error))completion {
    NSURL *url = [NSURL URLWithString:urlString];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    request.HTTPMethod = method;
    [request setValue:contentType forHTTPHeaderField:@"Content-Type"];

    // Handle body based on content type
    if ([contentType isEqualToString:@"application/json"] && [body isKindOfClass:[NSDictionary class]]) {
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:body options:0 error:nil];
        request.HTTPBody = jsonData;
    } else if ([contentType containsString:@"multipart/form-data"] && [body isKindOfClass:[NSData class]]) {
        [request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)[(NSData *)body length]] forHTTPHeaderField:@"Content-Length"];
        request.HTTPBody = body;
    } else if (body && [body isKindOfClass:[NSData class]]) {
        request.HTTPBody = body; // Raw data for other cases
    }

    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithRequest:request
                                                                 completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
        if (completion) {
            completion(data, httpResponse, error);
        }
    }];
    [task resume];
}

- (void)concatenateMP4Files:(NSArray<NSString *> *)filePaths completion:(void (^)(NSString *outputPath, NSError *error))completion {
    AVMutableComposition *composition = [AVMutableComposition composition];
    AVMutableCompositionTrack *videoTrack = [composition addMutableTrackWithMediaType:AVMediaTypeVideo preferredTrackID:kCMPersistentTrackID_Invalid];

    CMTime currentTime = kCMTimeZero;

    for (NSString *filePath in filePaths) {
        AVAsset *asset = [AVAsset assetWithURL:[NSURL fileURLWithPath:filePath]];

        if (asset.tracks.count == 0) continue;
        AVAssetTrack *videoAssetTrack = [[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];

        NSError *error = nil;
        [videoTrack insertTimeRange:CMTimeRangeMake(kCMTimeZero, asset.duration)
                            ofTrack:videoAssetTrack
                             atTime:currentTime
                              error:&error];
        currentTime = CMTimeAdd(currentTime, asset.duration);
    }

    NSString *outputPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject
                               stringByAppendingPathComponent:@"merged_video.mp4"];
    NSURL *outputURL = [NSURL fileURLWithPath:outputPath];
    [[NSFileManager defaultManager] removeItemAtURL:outputURL error:nil];//todo remove after user has downloaded too?

    AVAssetExportSession *exportSession = [[AVAssetExportSession alloc] initWithAsset:composition presetName:AVAssetExportPresetHighestQuality];
    exportSession.outputURL = outputURL;
    exportSession.outputFileType = AVFileTypeMPEG4;
    exportSession.shouldOptimizeForNetworkUse = YES;

    [exportSession exportAsynchronouslyWithCompletionHandler:^{
        if (exportSession.status == AVAssetExportSessionStatusCompleted) {
            completion(outputPath, nil);
        } else {
            completion(nil, exportSession.error);
        }
    }];
}

- (NSArray *)fetchEventDataFromCoreData:(NSManagedObjectContext *)context {
    if (!context) return @[];
    __block NSArray *eventDataArray = nil;
    
    // Perform fetch synchronously on the context's queue
    [context performBlockAndWait:^{
        NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"EventEntity"];
        NSSortDescriptor *sortDescriptor = [[NSSortDescriptor alloc] initWithKey:@"timeStamp"
                                                                     ascending:YES];
        [fetchRequest setSortDescriptors:@[sortDescriptor]];
        
        NSError *error = nil;
        NSArray *fetchedEvents = nil;
        
        @try {
            fetchedEvents = [context executeFetchRequest:fetchRequest error:&error];
        }
        @catch (NSException *exception) {
            eventDataArray = @[];
            return;
        }
        
        if (error || !fetchedEvents) {
            eventDataArray = @[];
            return;
        }
        
        NSMutableArray *tempArray = [NSMutableArray arrayWithCapacity:fetchedEvents.count];
        NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
        [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
        
        for (NSManagedObject *event in fetchedEvents) {
            // Safely access attributes with nil checks
            NSNumber *timeStampNumber = [event valueForKey:@"timeStamp"];
            if (!timeStampNumber) continue;
            
            NSTimeInterval timestamp = [timeStampNumber doubleValue];
            long long roundedTimestamp = (long long)timestamp;
            NSString *readableDate = [dateFormatter stringFromDate:[NSDate dateWithTimeIntervalSince1970:timestamp]];
            
            NSDictionary *eventDict = @{
                @"timeStamp": readableDate ?: @"",
                @"classType": [event valueForKey:@"classType"] ?: @"unknown",
                @"quantity": [event valueForKey:@"quantity"] ?: @(0),
                @"imageURL": [NSString stringWithFormat:@"images/%lld_small.jpg", roundedTimestamp]
            };
            [tempArray addObject:eventDict];
        }
        
        eventDataArray = [tempArray copy];
    }];
    
    return eventDataArray ?: @[];
}

- (void)sendFileData:(FILE *)file toSocket:(int)socket withContentLength:(NSUInteger)contentLength {
    char buffer[64 * 1024];
    size_t bytesToSend = contentLength;
    while (bytesToSend > 0) {
        size_t chunkSize = fread(buffer, 1, sizeof(buffer), file);
        if (chunkSize == 0) break;
        ssize_t bytesSent = send(socket, buffer, chunkSize, 0);
        if (bytesSent < 0) break;
        bytesToSend -= bytesSent;
    }
}

- (void)deleteAllDayEntitiesAndEventsInContext:(NSManagedObjectContext *)context {
    NSError *fetchError = nil;

    // Fetch and delete all EventEntity objects
    NSFetchRequest *eventFetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"EventEntity"];
    NSArray *eventEntities = [context executeFetchRequest:eventFetchRequest error:&fetchError];

    if (fetchError) return;
    for (NSManagedObject *eventEntity in eventEntities) {
        [context deleteObject:eventEntity];
    }

    // Fetch and delete all DayEntity objects
    NSFetchRequest *dayFetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"DayEntity"];
    NSArray *dayEntities = [context executeFetchRequest:dayFetchRequest error:&fetchError];

    if (fetchError) return;

    for (NSManagedObject *dayEntity in dayEntities) {
        [context deleteObject:dayEntity];
    }

    // Save the context after deletion
    NSError *saveError = nil;
    [context save:&saveError];
}


- (NSString *)processVideoDownloadWithLowRes:(BOOL)low_res
                                 startTime:(NSTimeInterval)startTimeStamp
                                   endTime:(NSTimeInterval)endTimeStamp
                                   context:(NSManagedObjectContext *)context {
    NSDate *startDate = [NSDate dateWithTimeIntervalSince1970:startTimeStamp];
    NSDate *endDate = [NSDate dateWithTimeIntervalSince1970:endTimeStamp];
    NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
    [formatter setDateFormat:@"yyyy-MM-dd"];

    NSString *formattedStartDate = [formatter stringFromDate:startDate];
    NSString *formattedEndDate = [formatter stringFromDate:endDate];

    if (![formattedStartDate isEqualToString:formattedEndDate]) {
        return nil;
    }

    NSCalendar *calendar = [NSCalendar currentCalendar];
    NSDate *midnight = [calendar startOfDayForDate:startDate];

    NSTimeInterval relativeStart = [startDate timeIntervalSinceDate:midnight];
    NSTimeInterval relativeEnd = [endDate timeIntervalSinceDate:midnight];
    NSTimeInterval requestedDuration = relativeEnd - relativeStart;

    NSArray *segments = [self fetchAndProcessSegmentsFromCoreDataForDateParam:formattedStartDate start:0 context:context];

    if (segments.count == 0) {
        return nil;
    }

    NSMutableArray<NSString *> *segmentFilePaths = [NSMutableArray array];
    NSString *tempDir = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:@"temp"];
    [[NSFileManager defaultManager] createDirectoryAtPath:tempDir withIntermediateDirectories:YES attributes:nil error:nil];

    dispatch_semaphore_t trimSema = dispatch_semaphore_create(0);
    __block NSInteger trimCount = 0;

    NSInteger orientation = -1;

    for (NSInteger i = 0; i < segments.count; i++) {
        NSTimeInterval segmentStart = [segments[i][@"timeStamp"] doubleValue];
        NSTimeInterval segmentDuration = [segments[i][@"duration"] doubleValue];
        NSTimeInterval segmentEnd = segmentStart + segmentDuration;

        if (segmentEnd <= relativeStart || segmentStart >= relativeEnd) {
            continue;
        }
        if(orientation == -1) orientation = [segments[i][@"orientation"] intValue];


        NSString *originalFilePath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:segments[i][@"url"]];
        if (![[NSFileManager defaultManager] fileExistsAtPath:originalFilePath]) continue;

        NSTimeInterval trimStart = MAX(0, relativeStart - segmentStart);
        NSTimeInterval trimEnd = MIN(segmentDuration, relativeEnd - segmentStart);
        NSTimeInterval trimmedDuration = trimEnd - trimStart;

        NSString *trimmedFilePath = [tempDir stringByAppendingPathComponent:[NSString stringWithFormat:@"trimmed_%ld.mp4", (long)i]];
        AVURLAsset *asset = [AVURLAsset assetWithURL:[NSURL fileURLWithPath:originalFilePath]];
        AVAssetExportSession *exportSession = [[AVAssetExportSession alloc] initWithAsset:asset presetName:AVAssetExportPresetHighestQuality];
        exportSession.outputFileType = AVFileTypeMPEG4;
        exportSession.outputURL = [NSURL fileURLWithPath:trimmedFilePath];

        CMTime startTime = CMTimeMakeWithSeconds(trimStart, 600);
        CMTime durationTime = CMTimeMakeWithSeconds(trimmedDuration, 600);
        CMTimeRange timeRange = CMTimeRangeMake(startTime, durationTime);
        if (CMTimeCompare(CMTimeAdd(startTime, durationTime), CMTimeMakeWithSeconds(segmentDuration, 600)) > 0) {
            durationTime = CMTimeSubtract(CMTimeMakeWithSeconds(segmentDuration, 600), startTime);
            timeRange = CMTimeRangeMake(startTime, durationTime);
        }
        exportSession.timeRange = timeRange;

        AVAssetTrack *videoTrack = [[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
        CGSize naturalSize = videoTrack.naturalSize;
        AVMutableVideoComposition *videoComposition = [AVMutableVideoComposition videoComposition];
        videoComposition.frameDuration = CMTimeMake(1, 24); // 24 fps

        AVMutableVideoCompositionInstruction *instruction = [AVMutableVideoCompositionInstruction videoCompositionInstruction];
        instruction.timeRange = timeRange;
        AVMutableVideoCompositionLayerInstruction *layerInstruction = [AVMutableVideoCompositionLayerInstruction videoCompositionLayerInstructionWithAssetTrack:videoTrack];
        CGAffineTransform transform = CGAffineTransformIdentity;
        CGSize renderSize = naturalSize;

        if (low_res) {
            CGFloat scale = MIN(1280.0 / naturalSize.width, 720.0 / naturalSize.height);
            transform = CGAffineTransformScale(transform, scale, scale);
            renderSize = CGSizeApplyAffineTransform(naturalSize, CGAffineTransformMakeScale(scale, scale));
        }

        if (orientation == 1) {
            renderSize = CGSizeMake(naturalSize.height, naturalSize.width);
            CGAffineTransform translate = CGAffineTransformMakeTranslation(naturalSize.height, 0);
            CGAffineTransform rotate = CGAffineTransformRotate(translate, M_PI_2);
            transform = rotate;
        } else if (orientation == 4) {
            transform = CGAffineTransformTranslate(transform, naturalSize.width / 2.0, naturalSize.height / 2.0);
            transform = CGAffineTransformRotate(transform, M_PI);
            transform = CGAffineTransformTranslate(transform, -naturalSize.width / 2.0, -naturalSize.height / 2.0);
        }

        [layerInstruction setTransform:transform atTime:kCMTimeZero];
        instruction.layerInstructions = @[layerInstruction];
        videoComposition.instructions = @[instruction];
        videoComposition.renderSize = renderSize;

        if (low_res) {
            exportSession.videoComposition = videoComposition;
            exportSession.shouldOptimizeForNetworkUse = YES;
            exportSession.fileLengthLimit = 20 * 1024 * 1024 * (trimmedDuration / 60.0);
        } else if (orientation == 1 || orientation == 4) {
            exportSession.videoComposition = videoComposition;
        }

        trimCount++;
        [exportSession exportAsynchronouslyWithCompletionHandler:^{
            dispatch_semaphore_signal(trimSema);
        }];
    }

    for (NSInteger i = 0; i < trimCount; i++) {
        dispatch_semaphore_wait(trimSema, DISPATCH_TIME_FOREVER);
    }

    for (NSInteger i = 0; i < segments.count; i++) {
        NSString *trimmedFilePath = [tempDir stringByAppendingPathComponent:[NSString stringWithFormat:@"trimmed_%ld.mp4", (long)i]];
        if ([[NSFileManager defaultManager] fileExistsAtPath:trimmedFilePath]) {
            [segmentFilePaths addObject:trimmedFilePath];
        }
    }

    if (segmentFilePaths.count == 0) {
        return nil;
    }

    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    __block NSString *outputPath = nil;
    __block NSError *mergeError = nil;

    [self concatenateMP4Files:segmentFilePaths completion:^(NSString *resultPath, NSError *error) {
        outputPath = resultPath;
        mergeError = error;
        dispatch_semaphore_signal(sema);
    }];

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);

    for (NSString *tempFile in segmentFilePaths) {
        if ([tempFile containsString:@"trimmed"]) {
            [[NSFileManager defaultManager] removeItemAtPath:tempFile error:nil];
        }
    }

    if (mergeError || !outputPath || ![[NSFileManager defaultManager] fileExistsAtPath:outputPath]) {
        return nil;
    }

    // Rename the output file using timestamps
    NSDateFormatter *renameFormatter = [[NSDateFormatter alloc] init];
    [renameFormatter setDateFormat:@"yyyy-MM-dd_HH-mm-ss"];
    NSString *startTimeStr = [renameFormatter stringFromDate:[NSDate dateWithTimeIntervalSince1970:startTimeStamp]];
    NSString *endTimeStr = [renameFormatter stringFromDate:[NSDate dateWithTimeIntervalSince1970:endTimeStamp]];
    NSString *newFileName = [NSString stringWithFormat:@"%@-%@.mp4", startTimeStr, endTimeStr];

    // Get the directory of the original outputPath and construct the new path
    NSString *directory = [outputPath stringByDeletingLastPathComponent];
    NSString *newOutputPath = [directory stringByAppendingPathComponent:newFileName];

    // Rename the file
    NSError *renameError = nil;
    [[NSFileManager defaultManager] moveItemAtPath:outputPath toPath:newOutputPath error:&renameError];
    if (renameError) {
        return outputPath;
    }

    return newOutputPath;
}

@end
