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
    
    if ([filePath hasPrefix:@"change-classes"]) {
        if (queryParams[@"indexes"]) {
            if ([[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] objectForKey:queryParams[@"indexes"]]) {
                [[SettingsManager sharedManager] updateYoloIndexesKey:queryParams[@"indexes"]];
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
        NSTimeInterval requestedDuration = relativeEnd - relativeStart;

        NSArray *segments = [self fetchAndProcessSegmentsFromCoreDataForDateParam:formattedStartDate start:0 context:self.context];

        if (segments.count == 0) {
            NSString *errorResponse = @"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n[{\"error\": \"No segments found for this date\"}]";
            send(clientSocket, [errorResponse UTF8String], errorResponse.length, 0);
            return;
        }

        NSMutableArray<NSString *> *segmentFilePaths = [NSMutableArray array];
        NSString *tempDir = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:@"temp"];
        [[NSFileManager defaultManager] createDirectoryAtPath:tempDir withIntermediateDirectories:YES attributes:nil error:nil];

        dispatch_semaphore_t trimSema = dispatch_semaphore_create(0);
        __block NSInteger trimCount = 0;

        BOOL low_res = YES;

        NSLog(@"Requested range: %.2f to %.2f (duration: %.2f seconds)", relativeStart, relativeEnd, requestedDuration);

        for (NSInteger i = 0; i < segments.count; i++) {
            NSTimeInterval segmentStart = [segments[i][@"timeStamp"] doubleValue];
            NSTimeInterval segmentDuration = [segments[i][@"duration"] doubleValue];
            NSTimeInterval segmentEnd = segmentStart + segmentDuration;

            if (segmentEnd <= relativeStart || segmentStart >= relativeEnd) {
                continue;
            }

            NSString *originalFilePath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:segments[i][@"url"]];
            if (![[NSFileManager defaultManager] fileExistsAtPath:originalFilePath]) {
                NSLog(@"Segment file not found: %@", originalFilePath);
                continue;
            }

            // Calculate trim points
            NSTimeInterval trimStart = MAX(0, relativeStart - segmentStart);
            NSTimeInterval trimEnd = MIN(segmentDuration, relativeEnd - segmentStart);
            NSTimeInterval trimmedDuration = trimEnd - trimStart;

            NSLog(@"Segment %ld: %.2f to %.2f (duration: %.2f), trimming to %.2f-%.2f (%.2f seconds)", (long)i, segmentStart, segmentEnd, segmentDuration, trimStart, trimEnd, trimmedDuration);

            NSString *trimmedFilePath = [tempDir stringByAppendingPathComponent:[NSString stringWithFormat:@"trimmed_%ld.mp4", (long)i]];
            AVURLAsset *asset = [AVURLAsset assetWithURL:[NSURL fileURLWithPath:originalFilePath]];
            AVAssetExportSession *exportSession = [[AVAssetExportSession alloc] initWithAsset:asset presetName:AVAssetExportPresetHighestQuality];
            exportSession.outputFileType = AVFileTypeMPEG4;
            exportSession.outputURL = [NSURL fileURLWithPath:trimmedFilePath];

            // Ensure time range is valid
            CMTime startTime = CMTimeMakeWithSeconds(trimStart, 600);
            CMTime durationTime = CMTimeMakeWithSeconds(trimmedDuration, 600);
            CMTimeRange timeRange = CMTimeRangeMake(startTime, durationTime);
            if (CMTimeCompare(CMTimeAdd(startTime, durationTime), CMTimeMakeWithSeconds(segmentDuration, 600)) > 0) {
                durationTime = CMTimeSubtract(CMTimeMakeWithSeconds(segmentDuration, 600), startTime);
                timeRange = CMTimeRangeMake(startTime, durationTime);
                NSLog(@"Adjusted duration for segment %ld to fit asset: %.2f seconds", (long)i, CMTimeGetSeconds(durationTime));
            }
            exportSession.timeRange = timeRange;

            // Quality adjustment block
            if (low_res) {
                AVMutableVideoComposition *videoComposition = [AVMutableVideoComposition videoComposition];
                AVAssetTrack *videoTrack = [[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
                if (videoTrack) {
                    videoComposition.renderSize = CGSizeMake(960, 540); // 540p
                    videoComposition.frameDuration = CMTimeMake(1, 24); // 24 fps

                    AVMutableVideoCompositionInstruction *instruction = [AVMutableVideoCompositionInstruction videoCompositionInstruction];
                    instruction.timeRange = timeRange;

                    AVMutableVideoCompositionLayerInstruction *layerInstruction = [AVMutableVideoCompositionLayerInstruction videoCompositionLayerInstructionWithAssetTrack:videoTrack];
                    CGSize naturalSize = videoTrack.naturalSize;
                    CGFloat scale = MIN(960.0 / naturalSize.width, 540.0 / naturalSize.height);
                    CGAffineTransform transform = CGAffineTransformMakeScale(scale, scale);
                    [layerInstruction setTransform:transform atTime:kCMTimeZero];

                    instruction.layerInstructions = @[layerInstruction];
                    videoComposition.instructions = @[instruction];
                }
                exportSession.videoComposition = videoComposition;
                exportSession.shouldOptimizeForNetworkUse = YES;
                
                exportSession.fileLengthLimit = 5 * 1024 * 1024 * (trimmedDuration / 60.0); // ~1 MB per minute
            }

            trimCount++;
            [exportSession exportAsynchronouslyWithCompletionHandler:^{
                switch (exportSession.status) {
                    case AVAssetExportSessionStatusCompleted:
                        NSLog(@"Processed segment %ld successfully%@, duration: %.2f seconds", (long)i, low_res ? @" at 1280x720" : @"", CMTimeGetSeconds(timeRange.duration));
                        break;
                    case AVAssetExportSessionStatusFailed:
                        NSLog(@"Failed to process segment %ld: %@", (long)i, exportSession.error);
                        break;
                    default:
                        break;
                }
                dispatch_semaphore_signal(trimSema);
            }];
        }

        // Wait for all processing to complete
        for (NSInteger i = 0; i < trimCount; i++) {
            dispatch_semaphore_wait(trimSema, DISPATCH_TIME_FOREVER);
        }

        // Add processed files to the list
        for (NSInteger i = 0; i < segments.count; i++) {
            NSString *trimmedFilePath = [tempDir stringByAppendingPathComponent:[NSString stringWithFormat:@"trimmed_%ld.mp4", (long)i]];
            if ([[NSFileManager defaultManager] fileExistsAtPath:trimmedFilePath]) {
                [segmentFilePaths addObject:trimmedFilePath];
            }
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

        // Clean up temporary files
        for (NSString *tempFile in segmentFilePaths) {
            if ([tempFile containsString:@"trimmed"]) {
                [[NSFileManager defaultManager] removeItemAtPath:tempFile error:nil];
            }
        }

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
    
    // Validate context
    if (!self.context) {
        NSLog(@"Error: Managed object context is nil");
        return NO;
    }
    
    // Verify entity exists in model
    NSEntityDescription *entity = [NSEntityDescription entityForName:@"EventEntity"
                                             inManagedObjectContext:self.context];
    if (!entity) {
        NSLog(@"Error: Entity 'EventEntity' not found in model");
        return NO;
    }
    
    // Validate timestamp range
    if (startTimeStamp > endTimeStamp) {
        NSLog(@"Error: startTimeStamp (%lf) must be less than or equal to endTimeStamp (%lf)", startTimeStamp, endTimeStamp);
        return NO;
    }
    
    // Get the app's Documents directory and images folder
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths firstObject];
    NSString *imagesDirectory = [documentsDirectory stringByAppendingPathComponent:@"images"];
    
    while (attempt < maxRetries && !success) {
        attempt++;
        NSLog(@"Attempt %d to delete events between %lf and %lf", attempt, startTimeStamp, endTimeStamp);
        
        [self.context performBlockAndWait:^{
            NSFetchRequest *fetchRequest = [[NSFetchRequest alloc] init];
            if (!fetchRequest) {
                NSLog(@"Failed to create fetch request (attempt %d)", attempt);
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
                NSLog(@"Fetched %lu events for deletion (attempt %d)", (unsigned long)events.count, attempt);
            }
            @catch (NSException *exception) {
                NSLog(@"Fetch exception (attempt %d): %@", attempt, exception.reason);
                success = NO;
                return;
            }
            
            if (fetchError) {
                NSLog(@"Fetch error (attempt %d): %@", attempt, fetchError.localizedDescription);
                success = NO;
                return;
            }
            
            if (!events) {
                NSLog(@"Fetch returned nil array (attempt %d)", attempt);
                success = YES; // Treat as success if no objects to delete
                return;
            }
            
            if (events.count == 0) {
                NSLog(@"No events found in range (attempt %d)", attempt);
                success = YES; // No events in range, still a success
                return;
            }
            
            // Delete events and their associated images
            NSFileManager *fileManager = [NSFileManager defaultManager];
            for (NSManagedObject *event in events) {
                NSNumber *timeStampNumber = [event valueForKey:@"timeStamp"];
                if (!timeStampNumber) {
                    NSLog(@"Warning: Event missing timestamp, skipping image deletion but deleting event");
                    [self.context deleteObject:event];
                    continue;
                }
                
                NSTimeInterval timeStamp = [timeStampNumber doubleValue];
                long long roundedTimestamp = (long long)floor(timeStamp); // Floor to integer
                NSString *imageFileName = [NSString stringWithFormat:@"%lld", roundedTimestamp];
                NSString *imageFilePath = [imagesDirectory stringByAppendingPathComponent:[imageFileName stringByAppendingString:@".jpg"]];
                NSString *smallImageFilePath = [imagesDirectory stringByAppendingPathComponent:[imageFileName stringByAppendingString:@"_small.jpg"]];
                
                // Delete regular image if it exists
                NSError *fileError = nil;
                if ([fileManager fileExistsAtPath:imageFilePath]) {
                    if ([fileManager removeItemAtPath:imageFilePath error:&fileError]) {
                        NSLog(@"Deleted image at %@", imageFilePath);
                    } else {
                        NSLog(@"Failed to delete image at %@: %@", imageFilePath, fileError.localizedDescription);
                    }
                }
                
                // Delete small image if it exists
                fileError = nil;
                if ([fileManager fileExistsAtPath:smallImageFilePath]) {
                    if ([fileManager removeItemAtPath:smallImageFilePath error:&fileError]) {
                        NSLog(@"Deleted small image at %@", smallImageFilePath);
                    } else {
                        NSLog(@"Failed to delete small image at %@: %@", smallImageFilePath, fileError.localizedDescription);
                    }
                }
                
                // Delete the event from Core Data
                [self.context deleteObject:event];
                NSLog(@"Marked event with timestamp %lf for deletion", timeStamp);
            }
            
            // Save changes
            if ([self.context hasChanges]) {
                NSLog(@"Context has %lu deleted objects to save (attempt %d)", (unsigned long)[self.context.deletedObjects count], attempt);
                NSError *saveError = nil;
                @try {
                    success = [self.context save:&saveError];
                    if (success) {
                        NSLog(@"Successfully saved deletions (attempt %d)", attempt);
                    } else {
                        NSLog(@"Save failed (attempt %d): %@", attempt, saveError.localizedDescription);
                    }
                }
                @catch (NSException *exception) {
                    NSLog(@"Save exception (attempt %d): %@", attempt, exception.reason);
                    success = NO;
                }
            } else {
                NSLog(@"No changes detected in context (attempt %d)", attempt);
                success = YES; // No changes to save, but this shouldn't happen if events were deleted
            }
        }];
        
        if (!success) {
            NSLog(@"Retrying batch delete (attempt %d/%d)...", attempt, maxRetries);
            [NSThread sleepForTimeInterval:0.1];
        }
    }
    
    if (!success) {
        NSLog(@"Failed to delete events between %lf and %lf after %d attempts", startTimeStamp, endTimeStamp, maxRetries);
    } else {
        NSLog(@"Successfully deleted events between %lf and %lf", startTimeStamp, endTimeStamp);
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
    // Validate context
    if (!context) {
        NSLog(@"Error: Managed object context is nil");
        return @[];
    }
    
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
            NSLog(@"Fetch exception: %@", exception.reason);
            eventDataArray = @[];
            return;
        }
        
        if (error) {
            NSLog(@"Error fetching events: %@, %@", error, error.userInfo);
            eventDataArray = @[];
            return;
        }
        
        if (!fetchedEvents) {
            NSLog(@"Fetch returned nil array");
            eventDataArray = @[];
            return;
        }
        
        NSMutableArray *tempArray = [NSMutableArray arrayWithCapacity:fetchedEvents.count];
        NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
        [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
        
        for (NSManagedObject *event in fetchedEvents) {
            // Safely access attributes with nil checks
            NSNumber *timeStampNumber = [event valueForKey:@"timeStamp"];
            if (!timeStampNumber) {
                NSLog(@"Warning: Event missing timestamp, skipping");
                continue;
            }
            
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
