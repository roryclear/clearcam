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
#import <CoreData/CoreData.h>
#import "SecretManager.h"

@interface FileServer ()
@property (nonatomic, strong) NSString *basePath;
@property (nonatomic, strong) NSMutableDictionary *durationCache;
@property (nonatomic, assign) int serverSocket;//todo
@property (nonatomic, assign) BOOL isServerRunning;
@end

@implementation FileServer

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

    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSURL *documentsURL = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    NSString *documentsPath = [documentsURL path];

    NSError *error;
    NSArray *contents = [fileManager contentsOfDirectoryAtPath:documentsPath error:&error];

    if (error) {
        NSLog(@"Failed to get contents of Documents directory: %@", error.localizedDescription);
        return;
    }

    if ([SettingsManager sharedManager].delete_on_launch) {
        [[SecretManager sharedManager] deleteAllKeysWithError:&error];
        for (NSString *file in contents) {
            if ([file hasPrefix:@"batch_req"]) continue;
            NSString *filePath = [documentsPath stringByAppendingPathComponent:file];
            NSError *error = nil;
            BOOL success = [fileManager removeItemAtPath:filePath error:&error];

            if (!success) {
                NSLog(@"Failed to delete %@: %@", file, error.localizedDescription);
            }
        }
        [self deleteAllDayEntitiesAndEventsInContext:self.context];
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:@"LastDeletedDayIndex"];
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:@"LastDeletedSegmentIndex"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
    
    self.segment_length = 1;
    self.scanner = [[PortScanner alloc] init];
    self.last_req_time = [NSDate now];
    self.basePath = [self getDocumentsDirectory];
    self.durationCache = [[NSMutableDictionary alloc] init];
    
    // Check initial state and start server if enabled
    BOOL streamViaWiFiEnabled = [[NSUserDefaults standardUserDefaults] boolForKey:@"stream_via_wifi_enabled"];
    if (streamViaWiFiEnabled) {
        [self startServer];
    }
}

- (void)dealloc {
    [self stopServer];
    [[NSUserDefaults standardUserDefaults] removeObserver:self
                                              forKeyPath:@"stream_via_wifi_enabled"];
}

- (void)startServer {
    if (self.isServerRunning) {
        return;  // Server already running
    }
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            signal(SIGPIPE, SIG_IGN);
            
            self.serverSocket = socket(AF_INET, SOCK_STREAM, 0);
            if (self.serverSocket == -1) {
                NSLog(@"Failed to create socket: %s", strerror(errno));
                return;
            }
            
            struct sockaddr_in serverAddr;
            memset(&serverAddr, 0, sizeof(serverAddr));
            serverAddr.sin_family = AF_INET;
            serverAddr.sin_addr.s_addr = INADDR_ANY;
            serverAddr.sin_port = htons(80);
            
            int opt = 1;
            setsockopt(self.serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            
            if (bind(self.serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1) {
                NSLog(@"Failed to bind socket: %s", strerror(errno));
                close(self.serverSocket);
                self.serverSocket = -1;
                return;
            }
            
            if (listen(self.serverSocket, 5) == -1) {
                NSLog(@"Failed to listen on socket: %s", strerror(errno));
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
            NSLog(@"Exception in server: %@", exception);
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
    if (!context) {
        NSLog(@"Context is nil, skipping fetch.");
        return @[];
    }

    __block NSArray *processedSegments = @[];

    [context performBlockAndWait:^{
        NSError *error = nil;

        // Fetch the DayEntity to get its ID
        NSFetchRequest *dayFetchRequest = [[NSFetchRequest alloc] initWithEntityName:@"DayEntity"];
        dayFetchRequest.predicate = [NSPredicate predicateWithFormat:@"date == %@", dateParam];
        dayFetchRequest.fetchLimit = 1; // We only need one DayEntity

        NSArray *fetchedDays = [context executeFetchRequest:dayFetchRequest error:&error];
        if (error || fetchedDays.count == 0) {
            NSLog(@"Failed to fetch DayEntity: %@ or no day found for %@", error.localizedDescription, dateParam);
            return;
        }

        NSManagedObject *dayEntity = fetchedDays.firstObject;

        // Fetch segments with a filter based on timeStamp
        NSFetchRequest *segmentFetchRequest = [[NSFetchRequest alloc] initWithEntityName:@"SegmentEntity"];
        segmentFetchRequest.predicate = [NSPredicate predicateWithFormat:@"day == %@ AND timeStamp >= %f", dayEntity, startTime];
        segmentFetchRequest.sortDescriptors = @[[NSSortDescriptor sortDescriptorWithKey:@"timeStamp" ascending:YES]]; // Consistent ordering
        segmentFetchRequest.propertiesToFetch = @[@"url", @"timeStamp", @"duration"];
        segmentFetchRequest.resultType = NSDictionaryResultType; // Return dictionaries directly

        NSArray *fetchedSegments = [context executeFetchRequest:segmentFetchRequest error:&error];
        if (error) {
            NSLog(@"Failed to fetch segments: %@", error.localizedDescription);
            return;
        }

        processedSegments = fetchedSegments;
    }];

    return processedSegments;
}
- (NSArray *)fetchFramesWithURLsFromCoreDataForDateParam:(NSString *)dateParam
                                                   start:(NSInteger)start
                                                 context:(NSManagedObjectContext *)context {
    if (!context) {
        NSLog(@"Context is nil, skipping fetch.");
        return @[];
    }

    __block NSArray *copiedSegments = @[];

    [context performBlockAndWait:^{
        NSError *error = nil;

        // Fetch the DayEntity for the given date
        NSFetchRequest *dayFetchRequest = [[NSFetchRequest alloc] initWithEntityName:@"DayEntity"];
        dayFetchRequest.predicate = [NSPredicate predicateWithFormat:@"date == %@", dateParam];

        NSArray *fetchedDays = [context executeFetchRequest:dayFetchRequest error:&error];

        if (error) {
            NSLog(@"Failed to fetch DayEntity: %@", error.localizedDescription);
            return;
        }

        if (fetchedDays.count == 0) {
            NSLog(@"No DayEntity found for date %@", dateParam);
            return;
        }

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
    self.last_req_time = [NSDate now];
    char requestBuffer[1024];
    ssize_t bytesRead = recv(clientSocket, requestBuffer, sizeof(requestBuffer) - 1, 0);
    if (bytesRead < 0) {
        NSLog(@"Failed to read request from client: %s", strerror(errno));
        return;
    }
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
    if ([filePath hasPrefix:@"change-classes"]) {
        NSString *indexesParam = nil;
        NSRange queryRange = [filePath rangeOfString:@"?"];
        
        if (queryRange.location != NSNotFound) {
            NSString *queryString = [filePath substringFromIndex:queryRange.location + 1];
            NSArray *queryItems = [queryString componentsSeparatedByString:@"&"];
            for (NSString *item in queryItems) {
                NSArray *keyValue = [item componentsSeparatedByString:@"="];
                if (keyValue.count == 2 && [keyValue[0] isEqualToString:@"indexes"]) {
                    indexesParam = keyValue[1];
                    break;
                }
            }
        }
        if (indexesParam) {
            if ([[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] objectForKey:indexesParam]) {
                [[SettingsManager sharedManager] updateYoloIndexesKey:indexesParam];
            } else {
                [[SettingsManager sharedManager] updateYoloIndexesKey:@"all"];
            }
            NSString *httpHeader = @"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            return;
        }
    }
    
    if ([filePath isEqualToString:@"get-presets"]) {
        NSDictionary *presets = [[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"];
        
        NSError *error;
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:[presets allKeys] options:0 error:&error];

        if (!jsonData) {
            NSLog(@"JSON Serialization Error: %@", error.localizedDescription);
            NSString *errorResponse = @"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        NSString *httpHeader = [NSString stringWithFormat:@"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %lu\r\n\r\n", (unsigned long)[jsonData length]];
        send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
        send(clientSocket, [jsonData bytes], [jsonData length], 0);
        return;
    }

    if ([filePath hasPrefix:@"get-classes"]) {
        NSString *currentClasses = [[NSUserDefaults standardUserDefaults] stringForKey:@"yolo_preset_idx"] ?: @"all"; // Default to "all" if not set
        [self sendJson200:@[currentClasses] toClient:clientSocket]; // Send as an array
        return;
    }
    
    if ([filePath hasPrefix:@"get-devices"]) {
        // Respond immediately with cached list
        @synchronized (self.scanner.cachedOpenPorts) {
            [self sendJson200:self.scanner.cachedOpenPorts toClient:clientSocket];
        }
        
        [self.scanner updateCachedOpenPortsForPort:80];
        return;
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
        if (self.segment_length == 60) { // todo, only for live req
            self.segment_length = 1;
        }
        
        NSString *startParam = nil;
        NSString *dateParam = nil;
        NSRange queryRange = [filePath rangeOfString:@"?"];
        if (queryRange.location != NSNotFound) {
            NSString *queryString = [filePath substringFromIndex:queryRange.location + 1];
            NSArray *queryItems = [queryString componentsSeparatedByString:@"&"];
            for (NSString *item in queryItems) {
                NSArray *keyValue = [item componentsSeparatedByString:@"="];
                if (keyValue.count == 2) {
                    if ([keyValue[0] isEqualToString:@"start"]) {
                        startParam = keyValue[1];
                    } else if ([keyValue[0] isEqualToString:@"date"]) {
                        dateParam = keyValue[1];
                    }
                }
            }
        }
        
        if (!dateParam) {
            NSString *httpHeader = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n";
            NSString *errorMessage = @"{\"error\": \"Missing or invalid date parameter\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [errorMessage UTF8String], errorMessage.length, 0);
            return;
        }
        
        double startTime = startParam ? [startParam doubleValue] : 0;
        NSArray *segmentsForDate = [self fetchAndProcessSegmentsFromCoreDataForDateParam:dateParam
                                                                                  start:startTime
                                                                                context:self.context];
        
        [self sendJson200:segmentsForDate toClient:clientSocket];
    }

    if ([filePath hasPrefix:@"download"]) {
        NSString *startParam = nil;
        NSString *endParam = nil;
        
        NSRange queryRange = [filePath rangeOfString:@"?"];
        if (queryRange.location != NSNotFound) {
            NSString *queryString = [filePath substringFromIndex:queryRange.location + 1];
            NSArray *queryItems = [queryString componentsSeparatedByString:@"&"];

            for (NSString *item in queryItems) {
                NSArray *keyValue = [item componentsSeparatedByString:@"="];
                if (keyValue.count == 2) {
                    if ([keyValue[0] isEqualToString:@"start"]) {
                        startParam = keyValue[1];
                    } else if ([keyValue[0] isEqualToString:@"end"]) {
                        endParam = keyValue[1];
                    }
                }
            }
        }
        if (!startParam || !endParam) {
            NSString *errorResponse = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"Missing or invalid start/end parameter\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        NSTimeInterval startTimeStamp = [startParam doubleValue];
        NSTimeInterval endTimeStamp = [endParam doubleValue];
        NSDate *startDate = [NSDate dateWithTimeIntervalSince1970:startTimeStamp];
        NSDate *endDate = [NSDate dateWithTimeIntervalSince1970:endTimeStamp];
        NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
        [formatter setDateFormat:@"yyyy-MM-dd"];

        NSString *formattedStartDate = [formatter stringFromDate:startDate];
        NSString *formattedEndDate = [formatter stringFromDate:endDate];

        if (![formattedStartDate isEqualToString:formattedEndDate]) {
            NSString *errorResponse = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"Start and end must be on the same day\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        NSCalendar *calendar = [NSCalendar currentCalendar];
        NSDate *midnight = [calendar startOfDayForDate:startDate];

        NSTimeInterval relativeStart = [startDate timeIntervalSinceDate:midnight];
        NSTimeInterval relativeEnd = [endDate timeIntervalSinceDate:midnight];

        NSArray *segments = [self fetchAndProcessSegmentsFromCoreDataForDateParam:formattedStartDate start:0 context:self.context];

        if (segments.count == 0) {
            NSString *errorResponse = @"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"No segments found for this date\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }
        NSInteger startIndex = -1;
        for (NSInteger i = segments.count - 1; i >= 0; i--) {
            NSTimeInterval segmentTimeStamp = [segments[i][@"timeStamp"] doubleValue];

            if (segmentTimeStamp < relativeStart) {
                startIndex = i;
                break;
            }
        }
        if (startIndex == -1) startIndex = 0;
        NSInteger endIndex = -1;
        for (NSInteger i = startIndex; i < segments.count; i++) {
            NSTimeInterval segmentStart = [segments[i][@"timeStamp"] doubleValue];
            NSTimeInterval segmentDuration = [segments[i][@"duration"] doubleValue];
            NSTimeInterval segmentEnd = segmentStart + segmentDuration;

            if (segmentEnd > relativeEnd) {
                endIndex = i;
                break;
            }
        }

        if (endIndex == -1) endIndex = segments.count - 1; // If none found, include everything

        NSMutableArray<NSString *> *segmentFilePaths = [NSMutableArray array];

        for (NSInteger i = startIndex; i <= endIndex; i++) {
            NSString *segmentURL = segments[i][@"url"];
            NSString *fullFilePath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:segmentURL];

            if (![[NSFileManager defaultManager] fileExistsAtPath:fullFilePath]) {
                NSLog(@"Segment file not found: %@", segmentURL);
                continue;
            }

            [segmentFilePaths addObject:fullFilePath];
        }

        if (segmentFilePaths.count == 0) {
            NSString *errorResponse = @"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"No valid segments to merge\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
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

        if (mergeError || !outputPath || ![[NSFileManager defaultManager] fileExistsAtPath:outputPath]) {
            NSString *errorResponse = @"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"Failed to merge video\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        FILE *mergedFile = fopen([outputPath UTF8String], "rb");
        if (!mergedFile) {
            NSLog(@"Failed to open merged video file for sending: %s", strerror(errno));
            return;
        }

        if (clientSocket < 0) {
            NSLog(@"Invalid socket: %d", clientSocket);
            fclose(mergedFile);
            return;
        }

        NSFileManager *fileManager = [NSFileManager defaultManager];
        NSDictionary *fileAttributes = [fileManager attributesOfItemAtPath:outputPath error:nil];
        NSUInteger fileSize = [fileAttributes[NSFileSize] unsignedIntegerValue];

        dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
        dprintf(clientSocket, "Content-Type: video/mp4\r\n");
        dprintf(clientSocket, "Content-Disposition: attachment; filename=\"%s-%s.mp4\"\r\n",
            [[^{ NSDateFormatter *f = [NSDateFormatter new]; f.dateFormat = @"yyyy-MM-dd_HH-mm-ss"; return f; }() stringFromDate:startDate] UTF8String],
            [[^{ NSDateFormatter *f = [NSDateFormatter new]; f.dateFormat = @"yyyy-MM-dd_HH-mm-ss"; return f; }() stringFromDate:endDate] UTF8String]);
        dprintf(clientSocket, "Content-Length: %lu\r\n", (unsigned long)fileSize);
        dprintf(clientSocket, "Accept-Ranges: bytes\r\n");
        dprintf(clientSocket, "\r\n");
        char buffer[64 * 1024];
        size_t bytesRead;
        while ((bytesRead = fread(buffer, 1, sizeof(buffer), mergedFile)) > 0) {
            ssize_t bytesSent = send(clientSocket, buffer, bytesRead, 0);
            if (bytesSent < 0) {
                NSLog(@"Error sending file data: %s", strerror(errno));
                break;
            }
        }

        fclose(mergedFile);
    }

    
    if ([filePath hasPrefix:@"get-frames"]) {
        NSString *urlParam = nil;
        NSRange queryRange = [filePath rangeOfString:@"?"];
        if (queryRange.location != NSNotFound) {
            NSString *queryString = [filePath substringFromIndex:queryRange.location + 1];
            NSArray *queryItems = [queryString componentsSeparatedByString:@"&"];
            for (NSString *item in queryItems) {
                NSArray *keyValue = [item componentsSeparatedByString:@"="];
                if (keyValue.count == 2 && [keyValue[0] isEqualToString:@"url"]) {
                    urlParam = keyValue[1];
                    break;
                }
            }
        }

        if (!urlParam) {
            NSString *httpHeader = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n";
            NSString *errorMessage = @"{\"error\": \"Missing or invalid url parameter\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [errorMessage UTF8String], errorMessage.length, 0);
            return;
        }
        
        [self sendJson200:[self fetchFramesForURL:urlParam context:self.context] toClient:clientSocket];
    }
    
    if ([filePath hasPrefix:@"get-events"]) [self sendJson200: [self fetchEventDataFromCoreData:self.context] toClient:clientSocket];

    if ([filePath hasPrefix:@"delete-event"]) {
        NSURLComponents *components = [NSURLComponents componentsWithString:[NSString stringWithFormat:@"http://localhost/%@", filePath]];
        NSString *eventTimeStamp = nil;
        
        for (NSURLQueryItem *item in components.queryItems) {
            if ([item.name isEqualToString:@"timeStamp"]) {
                eventTimeStamp = item.value;
                break;
            }
        }
        
        if (!eventTimeStamp) {
            NSString *httpHeader = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n";
            NSString *errorMessage = @"{\"error\": \"Missing timeStamp parameter\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [errorMessage UTF8String], errorMessage.length, 0);
            return;
        }

        // Convert the timeStamp to a number
        NSTimeInterval timeStamp = [eventTimeStamp doubleValue];
        
        BOOL success = [self deleteEventWithTimeStamp:timeStamp];
        
        if (success) {
            // Get the app's Documents directory
            NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
            NSString *documentsDirectory = [paths firstObject];
            NSString *imagesDirectory = [documentsDirectory stringByAppendingPathComponent:@"images"];

            // File path for the image (with .jpg extension)
            NSString *imageFilePath = [imagesDirectory stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.jpg", eventTimeStamp]];
            NSString *imageFilePathSmall = [imagesDirectory stringByAppendingPathComponent:[NSString stringWithFormat:@"%@_small.jpg", eventTimeStamp]];
            NSError *error = nil;
            // Check if the image file exists and delete it
            if ([[NSFileManager defaultManager] fileExistsAtPath:imageFilePath]) {
                if (![[NSFileManager defaultManager] removeItemAtPath:imageFilePath error:&error] || [[NSFileManager defaultManager] removeItemAtPath:imageFilePathSmall error:&error]) {
                    NSLog(@"Failed to delete image: %@", error.localizedDescription);
                }
            }

            // Optional cleanup: remove cached preferences for the last deleted event
            [[NSUserDefaults standardUserDefaults] removeObjectForKey:@"LastDeletedDayIndex"];
            [[NSUserDefaults standardUserDefaults] removeObjectForKey:@"LastDeletedSegmentIndex"];
            [[NSUserDefaults standardUserDefaults] synchronize];

            // Respond with success
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
    NSRange queryRange = [fullPath rangeOfString:@"?"];
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

- (BOOL)deleteEventWithTimeStamp:(NSTimeInterval)timeStamp {
    __block BOOL success = NO;
    const int maxRetries = 3;
    int attempt = 0;
    
    while (attempt < maxRetries) {
        attempt++;
        [self.context performBlockAndWait:^{
            NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"EventEntity"];
            double epsilon = 1;
            fetchRequest.predicate = [NSPredicate predicateWithFormat:@"(timeStamp >= %lf) AND (timeStamp <= %lf)", timeStamp - epsilon, timeStamp + epsilon];
            
            NSError *error = nil;
            NSArray *events = [self.context executeFetchRequest:fetchRequest error:&error];
            if (error) {
                NSLog(@"Fetch error (attempt %d): %@", attempt, error.localizedDescription);
                success = NO;
                return;
            }
            if (events.count == 0) {
                success = YES;
                return;
            }
            for (NSManagedObject *event in events) {
                [self.context deleteObject:event];
            }
            if ([self.context save:&error]) {
                success = YES;
            } else {
                NSLog(@"Save error (attempt %d): %@", attempt, error.localizedDescription);
                success = NO;
            }
        }];
        
        if (success) {
            break;
        }
        
        NSLog(@"Retrying delete (attempt %d/%d)...", attempt, maxRetries);
    }
    
    return success;
}


- (NSArray *)fetchFramesForURL:(NSString *)url context:(NSManagedObjectContext *)context {
    if (!context) {
        NSLog(@"Context is nil, skipping fetch.");
        return @[];
    }

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
        if (error) {
            NSLog(@"Failed to fetch segment for URL %@: %@", url, error.localizedDescription);
            return;
        }

        if (segments.count == 0) {
            NSLog(@"No segment found for URL %@", url);
            return;
        }

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

- (void)concatenateMP4Files:(NSArray<NSString *> *)filePaths completion:(void (^)(NSString *outputPath, NSError *error))completion {
    AVMutableComposition *composition = [AVMutableComposition composition];
    AVMutableCompositionTrack *videoTrack = [composition addMutableTrackWithMediaType:AVMediaTypeVideo preferredTrackID:kCMPersistentTrackID_Invalid];

    CMTime currentTime = kCMTimeZero;

    for (NSString *filePath in filePaths) {
        AVAsset *asset = [AVAsset assetWithURL:[NSURL fileURLWithPath:filePath]];

        if (asset.tracks.count == 0) {
            NSLog(@"Error: No tracks found in asset %@", filePath);
            continue;
        }

        AVAssetTrack *videoAssetTrack = [[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];

        NSError *error = nil;
        [videoTrack insertTimeRange:CMTimeRangeMake(kCMTimeZero, asset.duration)
                            ofTrack:videoAssetTrack
                             atTime:currentTime
                              error:&error];

        if (error) NSLog(@"Error inserting video track: %@", error.localizedDescription);

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
            if (![[NSFileManager defaultManager] fileExistsAtPath:outputPath]) NSLog(@"File not found in Documents: %@", outputPath);
            completion(outputPath, nil);
        } else {
            NSLog(@"Export failed: %@", exportSession.error.localizedDescription);
            completion(nil, exportSession.error);
        }
    }];
}

- (NSArray *)fetchEventDataFromCoreData:(NSManagedObjectContext *)context {
    NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"EventEntity"];
    NSSortDescriptor *sortDescriptor = [[NSSortDescriptor alloc] initWithKey:@"timeStamp"
                                                                 ascending:YES];
    [fetchRequest setSortDescriptors:@[sortDescriptor]];
    
    NSError *error = nil;
    NSArray *fetchedEvents = [context executeFetchRequest:fetchRequest error:&error];

    if (error) {
        NSLog(@"Error fetching events: %@, %@", error, error.userInfo);
        return @[];
    }

    NSMutableArray *eventDataArray = [NSMutableArray arrayWithCapacity:fetchedEvents.count];
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];

    for (NSManagedObject *event in fetchedEvents) {
        NSTimeInterval timestamp = [[event valueForKey:@"timeStamp"] doubleValue];
        long long roundedTimestamp = (long long)timestamp;
        NSString *readableDate = [dateFormatter stringFromDate:[NSDate dateWithTimeIntervalSince1970:timestamp]];
        
        NSDictionary *eventDict = @{
            @"timeStamp": readableDate,
            @"classType": [event valueForKey:@"classType"] ?: @"unknown",
            @"quantity": [event valueForKey:@"quantity"] ?: @(0),
            @"imageURL": [NSString stringWithFormat:@"images/%lld_small.jpg", roundedTimestamp]
        };
        [eventDataArray addObject:eventDict];
    }
    return eventDataArray;
}

- (void)sendFileData:(FILE *)file toSocket:(int)socket withContentLength:(NSUInteger)contentLength {
    char buffer[64 * 1024];
    size_t bytesToSend = contentLength;
    while (bytesToSend > 0) {
        size_t chunkSize = fread(buffer, 1, sizeof(buffer), file);
        if (chunkSize == 0) break;
        ssize_t bytesSent = send(socket, buffer, chunkSize, 0);
        if (bytesSent < 0) {
            NSLog(@"Failed to send data: %s", strerror(errno));
            break;
        }
        bytesToSend -= bytesSent;
    }
}

- (void)deleteAllDayEntitiesAndEventsInContext:(NSManagedObjectContext *)context {
    NSError *fetchError = nil;

    // Fetch and delete all EventEntity objects
    NSFetchRequest *eventFetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"EventEntity"];
    NSArray *eventEntities = [context executeFetchRequest:eventFetchRequest error:&fetchError];

    if (fetchError) {
        NSLog(@"Failed to fetch EventEntity objects: %@", fetchError.localizedDescription);
        return;
    }

    for (NSManagedObject *eventEntity in eventEntities) {
        [context deleteObject:eventEntity];
    }

    // Fetch and delete all DayEntity objects
    NSFetchRequest *dayFetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"DayEntity"];
    NSArray *dayEntities = [context executeFetchRequest:dayFetchRequest error:&fetchError];

    if (fetchError) {
        NSLog(@"Failed to fetch DayEntity objects: %@", fetchError.localizedDescription);
        return;
    }

    for (NSManagedObject *dayEntity in dayEntities) {
        [context deleteObject:dayEntity];
    }

    // Save the context after deletion
    NSError *saveError = nil;
    if (![context save:&saveError]) NSLog(@"Failed to delete DayEntity and EventEntity objects: %@", saveError.localizedDescription);
}
    
@end
