#import "fileserver.h"
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

@interface FileServer ()
@property (nonatomic, strong) NSString *basePath;
@property (nonatomic, strong) NSMutableDictionary *durationCache;
@property (nonatomic, strong) NSString *currentClasses;
@end

@implementation FileServer

- (void)start {
    
    //obv todo
    if([SettingsManager sharedManager].yolo_indexes.count == 80){
        self.currentClasses = @"all";
    } else {
        self.currentClasses = @"vehiclesPeople";
    }
    
    // coredata stuff
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
        for (NSString *file in contents) {
            if ([file hasPrefix:@"batch_req"]) continue;
            NSString *filePath = [documentsPath stringByAppendingPathComponent:file];
            NSError *error = nil;
            BOOL success = [fileManager removeItemAtPath:filePath error:&error];

            if (!success) {
                NSLog(@"Failed to delete %@: %@", file, error.localizedDescription);
            } else {
                NSLog(@"Deleted: %@", file);
            }
        }
        [self deleteAllDayEntitiesAndEventsInContext:self.context];
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:@"LastDeletedDayIndex"];
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:@"LastDeletedSegmentIndex"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        self.segment_length = 60;
        self.scanner = [[PortScanner alloc] init];
        self.last_req_time = [NSDate now];
        self.basePath = [self getDocumentsDirectory];
        self.durationCache = [[NSMutableDictionary alloc] init];
        [self startHTTPServerWithBasePath:self.basePath];
    });
}

- (NSString *)getDocumentsDirectory {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    return [paths firstObject];
}

- (NSArray *)fetchAndProcessSegmentsFromCoreDataForDateParam:(NSString *)dateParam
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
        if (start >= segments.count) {
            NSLog(@"Start index out of range (%ld/%lu)", (long)start, (unsigned long)segments.count);
            return;
        }

        copiedSegments = [[segments array] subarrayWithRange:NSMakeRange(start, segments.count - start)];
    }];

    // Process outside of performBlockAndWait to avoid blocking Core Data
    NSMutableArray *processedSegments = [NSMutableArray array];

    for (NSManagedObject *segment in copiedSegments) {
        NSString *url = [segment valueForKey:@"url"];
        double timeStamp = [[segment valueForKey:@"timeStamp"] doubleValue];
        double duration = [[segment valueForKey:@"duration"] doubleValue];

        NSDictionary *segmentDict = @{
            @"url": url,
            @"timeStamp": @(timeStamp),
            @"duration": @(duration)
        };

        [processedSegments addObject:segmentDict];
    }

    NSLog(@"Fetched and processed %lu segments (start=%ld) for date %@",
          (unsigned long)processedSegments.count, (long)start, dateParam);

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
        if (start >= segments.count) {
            NSLog(@"Start index out of range (%ld/%lu)", (long)start, (unsigned long)segments.count);
            return;
        }

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

    NSLog(@"Fetched and processed %lu frames with URLs (start=%ld) for date %@",
          (unsigned long)framesWithURLs.count, (long)start, dateParam);

    return framesWithURLs;
}

- (void)startHTTPServerWithBasePath:(NSString *)basePath {
    @try {
        signal(SIGPIPE, SIG_IGN);
        
        int serverSocket = -1;
        struct sockaddr_in serverAddr;
        
        while (serverSocket == -1) {
            serverSocket = socket(AF_INET, SOCK_STREAM, 0);
            [NSThread sleepForTimeInterval:0.1];
        }
        
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(8080);
        while (bind(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1) [NSThread sleepForTimeInterval:0.1];
        while (listen(serverSocket, 5) == -1) [NSThread sleepForTimeInterval:0.1];

        while (1) {
            int clientSocket = accept(serverSocket, NULL, NULL);
            if (clientSocket != -1) {
                dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                    [self handleClientRequest:clientSocket withBasePath:basePath];
                    close(clientSocket);
                });
            }
        }

    } @catch (NSException *exception) {
        NSLog(@"Exception: %@", exception);
    }
}

- (void)sendJson200:(NSArray *)array toClient:(int)clientSocket {
    NSError *error;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:array options:0 error:&error];
    NSString *httpHeader = [NSString stringWithFormat:@"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %lu\r\n\r\n", (unsigned long)jsonData.length];
    send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
    send(clientSocket, jsonData.bytes, jsonData.length, 0);
}

- (void)handleClientRequest:(int)clientSocket withBasePath:(NSString *)basePath {
    if(self.segment_length == 60){ //todo, only for live req
        self.segment_length = 1;
        sleep(2); //todo, this is bad
    }
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
            NSArray<NSNumber *> *newIndexes = nil;

            if ([indexesParam isEqualToString:@"all"]) {
                self.currentClasses = @"all";
                newIndexes = @[@0, @1, @2, @3, @4, @5, @6, @7, @8, @9, @10, @11, @12, @13, @14, @15,
                               @16, @17, @18, @19, @20, @21, @22, @23, @24, @25, @26, @27, @28, @29,
                               @30, @31, @32, @33, @34, @35, @36, @37, @38, @39, @40, @41, @42, @43,
                               @44, @45, @46, @47, @48, @49, @50, @51, @52, @53, @54, @55, @56, @57,
                               @58, @59, @60, @61, @62, @63, @64, @65, @66, @67, @68, @69, @70, @71,
                               @72, @73, @74, @75, @76, @77, @78, @79];
                [[SettingsManager sharedManager] updateYoloIndexesKey:@"all"];
            } else if ([indexesParam isEqualToString:@"vehiclesPeople"]) {
                self.currentClasses = @"vehiclesPeople";
                newIndexes = @[@0, @1, @2, @3, @5, @7];
                [[SettingsManager sharedManager] updateYoloIndexesKey:@"vehiclesPeople"];
            }

            [[SettingsManager sharedManager] updateYoloIndexes:newIndexes];

            NSString *httpHeader = @"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            return;
        }
    }

    if ([filePath hasPrefix:@"get-classes"]) {
        NSString *currentClasses = self.currentClasses ?: @"all"; // Default to "all" if not set
        [self sendJson200:@[currentClasses] toClient:clientSocket]; // Send as an array
        return;
    }
    
    if ([filePath hasPrefix:@"get-devices"]) {
        NSLog(@"get-devices??");
        
        // Respond immediately with cached list
        @synchronized (self.scanner.cachedOpenPorts) {
            [self sendJson200:self.scanner.cachedOpenPorts toClient:clientSocket];
        }
        
        [self.scanner updateCachedOpenPortsForPort:8080];
        return;
    }

    
    if ([filePath hasPrefix:@"main"] || [filePath hasPrefix:@"downloads"]) {
        NSString *fileName = [filePath hasPrefix:@"main"] ? @"main.html" : @"downloads.html";
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

        NSInteger start = startParam ? [startParam integerValue] : 0;
        NSArray *segmentsForDate = [self fetchAndProcessSegmentsFromCoreDataForDateParam:dateParam
                                                                                   start:start
                                                                                 context:self.context];
        [self sendJson200:segmentsForDate toClient:clientSocket];
    }

    if ([filePath hasPrefix:@"download"]) {
        NSString *startParam = nil;
        NSString *endParam = nil;
        
        // ✅ Extract query params
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

        // ✅ Validate params
        if (!startParam || !endParam) {
            NSString *errorResponse = @"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"Missing or invalid start/end parameter\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        NSTimeInterval startTimeStamp = [startParam doubleValue];
        NSTimeInterval endTimeStamp = [endParam doubleValue];

        // ✅ Convert timestamps to "seconds since midnight"
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

        // ✅ Fetch segments for the date
        NSArray *segments = [self fetchAndProcessSegmentsFromCoreDataForDateParam:formattedStartDate start:0 context:self.context];

        if (segments.count == 0) {
            NSString *errorResponse = @"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"No segments found for this date\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        // ✅ Find the last segment that starts BEFORE `relativeStart`
        NSInteger startIndex = -1;
        for (NSInteger i = segments.count - 1; i >= 0; i--) {
            NSTimeInterval segmentTimeStamp = [segments[i][@"timeStamp"] doubleValue];

            if (segmentTimeStamp < relativeStart) {
                startIndex = i;
                break;
            }
        }

        if (startIndex == -1) startIndex = 0; // If none found, start from the first segment

        // ✅ Find the first segment that ENDS AFTER `relativeEnd`
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

        // ✅ Collect segment file paths
        NSMutableArray<NSString *> *segmentFilePaths = [NSMutableArray array];

        for (NSInteger i = startIndex; i <= endIndex; i++) {
            NSString *segmentURL = segments[i][@"url"];
            NSString *fullFilePath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:segmentURL];

            if (![[NSFileManager defaultManager] fileExistsAtPath:fullFilePath]) {
                NSLog(@"❌ Segment file not found: %@", segmentURL);
                continue;
            }

            [segmentFilePaths addObject:fullFilePath];
        }

        if (segmentFilePaths.count == 0) {
            NSString *errorResponse = @"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"No valid segments to merge\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        // ✅ Merge files synchronously
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

        // ✅ Open merged file
        FILE *mergedFile = fopen([outputPath UTF8String], "rb");
        if (!mergedFile) {
            NSLog(@"❌ Failed to open merged video file for sending: %s", strerror(errno));
            return;
        }

        if (clientSocket < 0) {
            NSLog(@"❌ Invalid socket: %d", clientSocket);
            fclose(mergedFile);
            return;
        }

        // ✅ Get file size
        NSFileManager *fileManager = [NSFileManager defaultManager];
        NSDictionary *fileAttributes = [fileManager attributesOfItemAtPath:outputPath error:nil];
        NSUInteger fileSize = [fileAttributes[NSFileSize] unsignedIntegerValue];

        // ✅ Send HTTP Headers
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
        dprintf(clientSocket, "Content-Type: video/mp4\r\n");
        dprintf(clientSocket, "Content-Disposition: attachment; filename=\"merged_video.mp4\"\r\n");
        dprintf(clientSocket, "Content-Length: %lu\r\n", (unsigned long)fileSize);
        dprintf(clientSocket, "Accept-Ranges: bytes\r\n");
        dprintf(clientSocket, "\r\n");

        // ✅ Send File Data
        char buffer[64 * 1024];
        size_t bytesRead;
        while ((bytesRead = fread(buffer, 1, sizeof(buffer), mergedFile)) > 0) {
            ssize_t bytesSent = send(clientSocket, buffer, bytesRead, 0);
            if (bytesSent < 0) {
                NSLog(@"❌ Error sending file data: %s", strerror(errno));
                break;
            }
        }

        fclose(mergedFile);
        NSLog(@"✅ Merged video sent successfully.");
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

            NSError *error = nil;

            // Check if the image file exists and delete it
            if ([[NSFileManager defaultManager] fileExistsAtPath:imageFilePath]) {
                if (![[NSFileManager defaultManager] removeItemAtPath:imageFilePath error:&error]) {
                    NSLog(@"Failed to delete image: %@", error.localizedDescription);
                } else {
                    NSLog(@"Image deleted at path: %@", imageFilePath);
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
    NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"EventEntity"];
    double epsilon = 1;
    fetchRequest.predicate = [NSPredicate predicateWithFormat:@"(timeStamp >= %lf) AND (timeStamp <= %lf)", timeStamp - epsilon, timeStamp + epsilon];
    NSError *error = nil;
    NSArray *events = [self.context executeFetchRequest:fetchRequest error:&error];
    if (error) return NO;
    if (events.count == 0) return YES; //already gone?
    for (NSManagedObject *event in events) [self.context deleteObject:event];
    if (![self.context save:&error]) return NO;
    return YES;
}

- (NSArray *)fetchFramesForURL:(NSString *)url context:(NSManagedObjectContext *)context {
    if (!context) {
        NSLog(@"Context is nil, skipping fetch.");
        return @[];
    }

    __block NSArray *segmentObjectIDs = @[];

    [context performBlockAndWait:^{
        NSError *error = nil;
        NSFetchRequest *segmentFetchRequest = [[NSFetchRequest alloc] initWithEntityName:@"SegmentEntity"];
        segmentFetchRequest.predicate = [NSPredicate predicateWithFormat:@"url == %@", url];

        NSArray *segments = [context executeFetchRequest:segmentFetchRequest error:&error];

        if (error) {
            NSLog(@"Failed to fetch segments for URL %@: %@", url, error.localizedDescription);
            return;
        }

        if (segments.count == 0) {
            NSLog(@"No segments found for URL %@", url);
            return;
        }

        // Store object IDs instead of NSManagedObject references
        segmentObjectIDs = [segments valueForKey:@"objectID"];
    }];

    NSMutableArray *framesForURL = [NSMutableArray array];

    // Use a new context tied to the current queue
    NSManagedObjectContext *bgContext = [[NSManagedObjectContext alloc] initWithConcurrencyType:NSPrivateQueueConcurrencyType];
    bgContext.parentContext = context;

    [bgContext performBlockAndWait:^{
        for (NSManagedObjectID *segmentID in segmentObjectIDs) {
            NSManagedObject *segment = [bgContext objectWithID:segmentID];
            NSArray *frames = [[segment valueForKey:@"frames"] array];

            for (NSManagedObject *frame in frames) {
                double frameTimeStamp = [[frame valueForKey:@"frame_timeStamp"] doubleValue];
                double aspectRatio = [[frame valueForKey:@"aspect_ratio"] doubleValue];
                int res = [[frame valueForKey:@"res"] intValue];

                NSMutableArray *squareDicts = [NSMutableArray array];
                NSArray *squares = [[frame valueForKey:@"squares"] array];

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
                    @"url": url,
                    @"frame_timeStamp": @(frameTimeStamp),
                    @"aspect_ratio": @(aspectRatio),
                    @"res": @(res),
                    @"squares": squareDicts
                };

                [framesForURL addObject:frameDict];
            }
        }
    }];

    NSLog(@"Fetched and processed %lu frames for URL %@", (unsigned long)framesForURL.count, url);

    return framesForURL;
}

- (void)concatenateMP4Files:(NSArray<NSString *> *)filePaths completion:(void (^)(NSString *outputPath, NSError *error))completion {
    AVMutableComposition *composition = [AVMutableComposition composition];
    AVMutableCompositionTrack *videoTrack = [composition addMutableTrackWithMediaType:AVMediaTypeVideo preferredTrackID:kCMPersistentTrackID_Invalid];

    CMTime currentTime = kCMTimeZero;

    for (NSString *filePath in filePaths) {
        AVAsset *asset = [AVAsset assetWithURL:[NSURL fileURLWithPath:filePath]];

        if (asset.tracks.count == 0) {
            NSLog(@"❌ Error: No tracks found in asset %@", filePath);
            continue;
        }

        AVAssetTrack *videoAssetTrack = [[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];

        NSError *error = nil;
        [videoTrack insertTimeRange:CMTimeRangeMake(kCMTimeZero, asset.duration)
                            ofTrack:videoAssetTrack
                             atTime:currentTime
                              error:&error];

        if (error) {
            NSLog(@"❌ Error inserting video track: %@", error.localizedDescription);
        }

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
            NSLog(@"✅ Export successful: %@", outputPath);
            if (![[NSFileManager defaultManager] fileExistsAtPath:outputPath]) {
                NSLog(@"❌ File not found in Documents: %@", outputPath);
            } else {
                NSLog(@"✅ File successfully saved at: %@", outputPath);
            }

            completion(outputPath, nil);
        } else {
            NSLog(@"❌ Export failed: %@", exportSession.error.localizedDescription);
            completion(nil, exportSession.error);
        }
    }];
}

- (NSArray *)fetchEventDataFromCoreData:(NSManagedObjectContext *)context {
    NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"EventEntity"];
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
            @"imageURL": [NSString stringWithFormat:@"images/%lld.jpg", roundedTimestamp]
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
    if (![context save:&saveError]) {
        NSLog(@"Failed to delete DayEntity and EventEntity objects: %@", saveError.localizedDescription);
    } else {
        NSLog(@"All DayEntity and EventEntity objects deleted successfully");
    }
}
    
@end
