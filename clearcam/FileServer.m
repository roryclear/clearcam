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
@end

@implementation FileServer

- (void)start {
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
                if (keyValue.count == 2) {
                    if ([keyValue[0] isEqualToString:@"indexes"]) {
                        indexesParam = keyValue[1];
                        break;
                    }
                }
            }
        }
        
        if (indexesParam) {
            NSMutableArray<NSNumber *> *newIndexes = [NSMutableArray array];
            NSArray *indexesArray = [indexesParam componentsSeparatedByString:@","];
            for (NSString *indexString in indexesArray) {
                NSNumber *index = @([indexString intValue]);
                [newIndexes addObject:index];
            }
            [[SettingsManager sharedManager] updateYoloIndexes:[newIndexes copy]];
            NSString *httpHeader = @"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            return;
        }
    }
    if ([filePath hasPrefix:@"get-classes"]) {
        NSArray<NSNumber *> *currentClasses = [SettingsManager sharedManager].yolo_indexes;
        NSError *error;
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:currentClasses options:0 error:&error];
        
        if (!jsonData) {
            NSLog(@"Error serializing JSON: %@", error);
            NSString *httpHeader = @"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            return;
        }
        
        NSString *httpHeader = [NSString stringWithFormat:@"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %lu\r\n\r\n", (unsigned long)jsonData.length];
        send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
        send(clientSocket, jsonData.bytes, jsonData.length, 0);
        return;
    }
    
    if ([filePath hasPrefix:@"get-devices"]) {
        NSLog(@"get-devices??");
        
        // Respond immediately with cached list
        @synchronized (self.scanner.cachedOpenPorts) {
            NSError *error;
            NSData *jsonData = [NSJSONSerialization dataWithJSONObject:self.scanner.cachedOpenPorts options:0 error:&error];
            
            if (!jsonData) {
                NSLog(@"Error serializing JSON: %@", error);
                NSString *httpHeader = @"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
                send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
                return;
            }
            
            NSString *httpHeader = [NSString stringWithFormat:@"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %lu\r\n\r\n", (unsigned long)jsonData.length];
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, jsonData.bytes, jsonData.length, 0);
        }
        
        // Update cached list asynchronously
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
        if (segmentsForDate.count == 0) {
            NSString *httpHeader = @"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n";
            NSString *errorMessage = @"{\"error\": \"No segments found or start index out of range\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [errorMessage UTF8String], errorMessage.length, 0);
            return;
        }

        NSData *slicedJsonData = [NSJSONSerialization dataWithJSONObject:segmentsForDate options:0 error:nil];
        NSString *httpHeader = [NSString stringWithFormat:@"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %lu\r\n\r\n", (unsigned long)slicedJsonData.length];
        send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
        send(clientSocket, slicedJsonData.bytes, slicedJsonData.length, 0);
    }
    
    if ([filePath hasPrefix:@"get-events"]) {
        NSArray *eventDataArray = [self fetchEventDataFromCoreData:self.context];
        
        if (eventDataArray.count == 0) {
            NSString *httpHeader = @"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n";
            NSString *errorMessage = @"{\"error\": \"No events found\"}";
            send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
            send(clientSocket, [errorMessage UTF8String], errorMessage.length, 0);
            return;
        }

        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:eventDataArray options:0 error:nil];
        NSString *httpHeader = [NSString stringWithFormat:@"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %lu\r\n\r\n", (unsigned long)jsonData.length];
        send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
        send(clientSocket, jsonData.bytes, jsonData.length, 0);
    }

    if ([filePath hasPrefix:@"delete-event"]) {
        // Extract the timeStamp from the URL query
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
            
            //TODO//
            [[NSUserDefaults standardUserDefaults] removeObjectForKey:@"LastDeletedDayIndex"];
            [[NSUserDefaults standardUserDefaults] removeObjectForKey:@"LastDeletedSegmentIndex"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            //THIS WILL BE SLOW LONG TERM, use a queue or something
            
            NSString *httpHeader = @"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n";
            NSString *successMessage = @"{\"success\": \"Event deleted\"}";
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
    }
    fclose(file);
}

- (BOOL)deleteEventWithTimeStamp:(NSTimeInterval)timeStamp {
    if (!self.context) {
        NSLog(@"Error: Core Data context is nil!");
        return NO;
    }

    NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"EventEntity"];
    
    // TODO, this delete everything within a second of the rounded timestamp, fine for now
    double epsilon = 1;
    fetchRequest.predicate = [NSPredicate predicateWithFormat:@"(timeStamp >= %lf) AND (timeStamp <= %lf)", timeStamp - epsilon, timeStamp + epsilon];

    NSError *error = nil;
    NSArray *events = [self.context executeFetchRequest:fetchRequest error:&error];

    if (error) {
        NSLog(@"Core Data Fetch Error: %@", error.localizedDescription);
        return NO;
    }

    if (events.count == 0) {
        NSLog(@"No EventEntity found for timeStamp: %lf", timeStamp);
        return NO;
    }

    for (NSManagedObject *event in events) {
        [self.context deleteObject:event];
    }

    if (![self.context save:&error]) {
        NSLog(@"Core Data Save Error: %@", error.localizedDescription);
        return NO;
    }

    NSLog(@"Successfully deleted EventEntity with timeStamp: %lf", timeStamp);
    return YES;
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
        NSString *readableDate = [dateFormatter stringFromDate:[NSDate dateWithTimeIntervalSince1970:timestamp]];
        
        NSDictionary *eventDict = @{
            @"timeStamp": readableDate,  // Readable date format
            @"classType": [event valueForKey:@"classType"] ?: @"unknown",
            @"quantity": [event valueForKey:@"quantity"] ?: @(0)
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
