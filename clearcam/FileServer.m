#import "fileserver.h"
#import <sys/types.h>
#import <sys/socket.h>
#import <netinet/in.h>
#import <arpa/inet.h>
#import <unistd.h>
#import <signal.h>
#import <errno.h>

@interface FileServer ()
@property (nonatomic, strong) NSString *basePath;
@end

@implementation FileServer

- (void)start {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        self.basePath = [self getDocumentsDirectory];
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
        serverAddr.sin_port = htons(8080);

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

        NSLog(@"Serving files at http://localhost:8080/");

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
    if (bytesRead < 0) return;

    requestBuffer[bytesRead] = '\0';
    NSString *request = [NSString stringWithUTF8String:requestBuffer];
    if (![request containsString:@"GET /"]) {
        dprintf(clientSocket, "HTTP/1.1 400 Bad Request\r\n\r\n");
        return;
    }
    
    NSString *filePath = [[request componentsSeparatedByString:@" "] objectAtIndex:1];
    filePath = [filePath stringByRemovingPercentEncoding];
    NSString *fullPath = [basePath stringByAppendingPathComponent:[filePath isEqualToString:@"/"] ? @"" : filePath];

    BOOL isDirectory = NO;
    if (![[NSFileManager defaultManager] fileExistsAtPath:fullPath isDirectory:&isDirectory]) {
        dprintf(clientSocket, "HTTP/1.1 404 Not Found\r\n\r\n");
        return;
    }

    if (isDirectory) {
        NSArray *files = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:fullPath error:nil] sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body><h1>Directory Listing</h1><ul>");
        for (NSString *file in files) {
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
    NSString *contentType, *contentDisposition;

    if ([fileExtension isEqualToString:@"mp4"]) {
        if ([request containsString:@"Range: bytes="]) {
            NSRange range = [request rangeOfString:@"bytes="];
            NSString *byteRange = [[request substringFromIndex:NSMaxRange(range)] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            NSUInteger start = [[byteRange componentsSeparatedByString:@"-"].firstObject integerValue];
            NSUInteger end = [[byteRange componentsSeparatedByString:@"-"].lastObject integerValue] ?: fileSize - 1;

            if (start >= fileSize || end >= fileSize || start > end) {
                dprintf(clientSocket, "HTTP/1.1 416 Requested Range Not Satisfiable\r\nContent-Range: bytes */%lu\r\n\r\n", fileSize);
            } else {
                dprintf(clientSocket, "HTTP/1.1 206 Partial Content\r\nContent-Type: video/mp4\r\nContent-Range: bytes %lu-%lu/%lu\r\nContent-Length: %lu\r\nAccept-Ranges: bytes\r\n\r\n", start, end, fileSize, end - start + 1);
                fseek(file, start, SEEK_SET);
                [self sendFileData:file toSocket:clientSocket withContentLength:end - start + 1];
            }
        } else {
            contentType = @"video/mp4";
            contentDisposition = @"";
        }
    } else if ([fileExtension isEqualToString:@"txt"]) {
        contentType = @"text/plain";
        contentDisposition = @"";
    } else {
        contentType = @"application/octet-stream";
        contentDisposition = [NSString stringWithFormat:@"attachment; filename=\"%s\"", [[fullPath lastPathComponent] UTF8String]];
    }

    if (contentType) {
        dprintf(clientSocket, "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Disposition: %s\r\nContent-Length: %lu\r\n\r\n", [contentType UTF8String], [contentDisposition UTF8String], fileSize);
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

