#import "fileserver.h"
#import <sys/types.h>
#import <sys/socket.h>
#import <netinet/in.h>
#import <arpa/inet.h>
#import <unistd.h>
#import <signal.h>
#import <errno.h>
#import <AVFoundation/AVFoundation.h>

@interface FileServer ()
@property (nonatomic, strong) NSString *basePath;
@property (nonatomic, strong) NSMutableDictionary *durationCache;
@end

@implementation FileServer

- (void)start {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        self.basePath = [self getDocumentsDirectory];
        self.durationCache = [[NSMutableDictionary alloc] init];
        [self startHTTPServerWithBasePath:self.basePath];
    });
}

- (NSString *)getDocumentsDirectory {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    return [paths firstObject];
}

- (void)startHTTPServerWithBasePath:(NSString *)basePath {
    @try {
        signal(SIGPIPE, SIG_IGN);

        int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == -1) {
            NSLog(@"Failed to create socket: %s", strerror(errno));
            return;
        }

        struct sockaddr_in serverAddr;
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(8081);

        if (bind(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1) {
            NSLog(@"Failed to bind socket: %s", strerror(errno));
            close(serverSocket);
            return;
        }

        if (listen(serverSocket, 5) == -1) {
            NSLog(@"Failed to listen on socket: %s", strerror(errno));
            close(serverSocket);
            return;
        }

        NSLog(@"Serving files at http://localhost:8081/");

        while (1) {
            int clientSocket = accept(serverSocket, NULL, NULL);
            if (clientSocket == -1) continue;

            // Create a new thread or dispatch queue for each client connection
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                [self handleClientRequest:clientSocket withBasePath:basePath];
                close(clientSocket);
            });
        }

    } @catch (NSException *exception) {
        NSLog(@"Exception: %@", exception);
    }
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
    NSLog(@"Client request: %s", requestBuffer);

    NSRange range = [request rangeOfString:@"GET /"];
    if (range.location == NSNotFound) {
        dprintf(clientSocket, "HTTP/1.1 400 Bad Request\r\n\r\n");
        return;
    }
    NSString *filePath = [[request substringFromIndex:NSMaxRange(range)] componentsSeparatedByString:@" "][0];
    filePath = [filePath stringByRemovingPercentEncoding];
    if ([filePath isEqualToString:@"/"]) filePath = @"";

    // Handle /get-segments API endpoint
    if ([filePath isEqualToString:@"get-segments"]) {
        NSString *documentsPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
        NSArray *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:documentsPath error:nil];
        NSPredicate *mp4Filter = [NSPredicate predicateWithFormat:@"SELF ENDSWITH '.mp4'"];
        NSArray *mp4Files = [files filteredArrayUsingPredicate:mp4Filter];

        // Sort .mp4 files by numeric values
        NSArray *sortedFiles = [mp4Files sortedArrayUsingComparator:^NSComparisonResult(NSString *file1, NSString *file2) {
            NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:@"_(\\d+)_"
                                                                                   options:0
                                                                                     error:nil];
            NSTextCheckingResult *match1 = [regex firstMatchInString:file1 options:0 range:NSMakeRange(0, file1.length)];
            NSTextCheckingResult *match2 = [regex firstMatchInString:file2 options:0 range:NSMakeRange(0, file2.length)];

            NSInteger num1 = match1 ? [[file1 substringWithRange:[match1 rangeAtIndex:1]] integerValue] : 0;
            NSInteger num2 = match2 ? [[file2 substringWithRange:[match2 rangeAtIndex:1]] integerValue] : 0;

            return [@(num1) compare:@(num2)];
        }];

        // Exclude the last segment from the list
        if (sortedFiles.count > 0) {
            sortedFiles = [sortedFiles subarrayWithRange:NSMakeRange(0, sortedFiles.count - 1)];
        }

        // Prepare JSON response
        NSMutableArray *responseArray = [NSMutableArray array];
        for (NSString *file in sortedFiles) {
            NSString *filePath = [documentsPath stringByAppendingPathComponent:file];

            NSNumber *cachedDuration = self.durationCache[file];
            Float64 duration;
            if (cachedDuration) {
                duration = [cachedDuration doubleValue];
            } else {
                // Get duration using AVAsset
                AVAsset *asset = [AVAsset assetWithURL:[NSURL fileURLWithPath:filePath]];
                CMTime time = asset.duration;
                duration = CMTimeGetSeconds(time);

                if (isfinite(duration)) {
                    duration = round(duration * 1000) / 1000.0;  // Round to 3 decimal places
                    self.durationCache[file] = @(duration);
                }
            }

            if (isfinite(duration)) {
                [responseArray addObject:@{
                    @"url": file,
                    @"duration": @(duration)
                }];
            }
        }

        // Send JSON response
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:responseArray options:0 error:nil];
        NSString *httpHeader = @"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n";
        send(clientSocket, [httpHeader UTF8String], httpHeader.length, 0);
        send(clientSocket, jsonData.bytes, jsonData.length, 0);
        return;
    }


    NSString *fullPath = [basePath stringByAppendingPathComponent:filePath];
    BOOL isDirectory = NO;
    if (![[NSFileManager defaultManager] fileExistsAtPath:fullPath isDirectory:&isDirectory]) {
        dprintf(clientSocket, "HTTP/1.1 404 Not Found\r\n\r\n");
        return;
    }

    if (isDirectory) {
        // Directory listing
        NSArray *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:fullPath error:nil];
        NSArray *sortedFiles = [files sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n");
        dprintf(clientSocket, "<html><body><h1>Directory Listing</h1><ul>");
        for (NSString *file in sortedFiles) {
            NSString *fileLink = [file stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLPathAllowedCharacterSet]];
            dprintf(clientSocket, "<li>%s <a href=\"/%s\">Stream</a> <a href=\"/%s\" download>Download</a></li>", file.UTF8String, fileLink.UTF8String, fileLink.UTF8String);
        }
        dprintf(clientSocket, "</ul></body></html>");
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
        } else {
            dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
            dprintf(clientSocket, "Content-Type: video/mp4\r\n");
            dprintf(clientSocket, "Content-Length: %lu\r\n", fileSize);
            dprintf(clientSocket, "Accept-Ranges: bytes\r\n");
            dprintf(clientSocket, "\r\n");
            [self sendFileData:file toSocket:clientSocket withContentLength:fileSize];
        }
    } else if ([fileExtension isEqualToString:@"txt"]) {
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
        dprintf(clientSocket, "Content-Type: text/plain\r\n");
        dprintf(clientSocket, "Content-Length: %lu\r\n", fileSize);
        dprintf(clientSocket, "\r\n");
        [self sendFileData:file toSocket:clientSocket withContentLength:fileSize];
    } else if ([fileExtension isEqualToString:@"html"]) {
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
        dprintf(clientSocket, "Content-Type: text/html\r\n");
        dprintf(clientSocket, "Content-Length: %lu\r\n", fileSize);
        dprintf(clientSocket, "\r\n");
        [self sendFileData:file toSocket:clientSocket withContentLength:fileSize];
    } else {
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\n");
        dprintf(clientSocket, "Content-Type: application/octet-stream\r\n");
        dprintf(clientSocket, "Content-Disposition: attachment; filename=\"%s\"\r\n", [[fullPath lastPathComponent] UTF8String]);
        dprintf(clientSocket, "Content-Length: %lu\r\n", fileSize);
        dprintf(clientSocket, "\r\n");
        [self sendFileData:file toSocket:clientSocket withContentLength:fileSize];
    }
    fclose(file);
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
    
@end
