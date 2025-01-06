#import "FileServer.h"
#include <netinet/in.h>
#include <sys/socket.h>

@interface FileServer ()
@property (nonatomic) int serverSocket;
@end

@implementation FileServer

- (void)start {
    self.serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (self.serverSocket < 0) {
        NSLog(@"Error: Unable to create socket.");
        return;
    }

    // Disable SIGPIPE
    int optval = 1;
    setsockopt(self.serverSocket, SOL_SOCKET, SO_NOSIGPIPE, &optval, sizeof(optval));

    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(8080);
    serverAddress.sin_addr.s_addr = INADDR_ANY;

    if (bind(self.serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) < 0) {
        NSLog(@"Error: Unable to bind socket.");
        close(self.serverSocket);
        return;
    }

    if (listen(self.serverSocket, 5) < 0) {
        NSLog(@"Error: Unable to listen on socket.");
        close(self.serverSocket);
        return;
    }

    NSLog(@"FileServer started on port 8080. Access it at http://<your-ip>:8080/");
    [self handleConnections];
}


- (void)handleConnections {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        while (YES) {
            struct sockaddr_in clientAddress;
            socklen_t clientAddressLen = sizeof(clientAddress);
            int clientSocket = accept(self.serverSocket, (struct sockaddr *)&clientAddress, &clientAddressLen);
            if (clientSocket >= 0) {
                [self handleClient:clientSocket];
            }
        }
    });
}

- (void)handleClient:(int)clientSocket {
    char buffer[1024];
    ssize_t receivedBytes = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
    if (receivedBytes <= 0) {
        close(clientSocket);
        return;
    }

    buffer[receivedBytes] = '\0';
    NSString *request = [NSString stringWithUTF8String:buffer];
    NSLog(@"Request: %@", request);

    if ([request containsString:@"GET /?file="]) {
        NSString *fileName = [self extractFileNameFromRequest:request];
        BOOL isDownload = [self shouldDownloadFromRequest:request];
        [self serveFile:fileName toSocket:clientSocket asDownload:isDownload];
    } else {
        [self serveFileListToSocket:clientSocket];
    }

    close(clientSocket);
}

- (NSString *)extractFileNameFromRequest:(NSString *)request {
    NSRange fileRange = [request rangeOfString:@"file="];
    if (fileRange.location != NSNotFound) {
        NSString *partialRequest = [request substringFromIndex:(fileRange.location + fileRange.length)];
        NSRange endRange = [partialRequest rangeOfString:@"&"];
        if (endRange.location != NSNotFound) {
            NSString *fileName = [partialRequest substringToIndex:endRange.location];
            return [fileName stringByRemovingPercentEncoding];
        } else {
            return [partialRequest stringByRemovingPercentEncoding];
        }
    }
    return nil;
}

- (BOOL)shouldDownloadFromRequest:(NSString *)request {
    NSRange actionRange = [request rangeOfString:@"action="];
    if (actionRange.location != NSNotFound) {
        NSString *partialRequest = [request substringFromIndex:(actionRange.location + actionRange.length)];
        NSRange endRange = [partialRequest rangeOfString:@" "];
        if (endRange.location != NSNotFound) {
            NSString *action = [partialRequest substringToIndex:endRange.location];
            return [action isEqualToString:@"download"];
        }
    }
    return NO; // Default to view
}

- (void)serveFile:(NSString *)fileName toSocket:(int)clientSocket asDownload:(BOOL)isDownload {
    NSString *decodedFileName = [fileName stringByRemovingPercentEncoding];
    NSString *documentsPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
    NSString *filePath = [documentsPath stringByAppendingPathComponent:decodedFileName];

    NSLog(@"Attempting to serve file at path: %@", filePath);

    NSData *fileData = [NSData dataWithContentsOfFile:filePath];
    if (!fileData) {
        NSLog(@"Error: File not found at path: %@", filePath);
        NSString *response = @"HTTP/1.1 404 Not Found\r\n\r\nFile not found.";
        send(clientSocket, [response UTF8String], [response length], 0);
        return;
    }

    NSString *contentDisposition = isDownload ? @"attachment" : @"inline";
    NSString *contentType = [self contentTypeForFileAtPath:filePath];

    NSString *header = [NSString stringWithFormat:
                        @"HTTP/1.1 200 OK\r\n"
                        "Content-Disposition: %@; filename=\"%@\"\r\n"
                        "Content-Length: %lu\r\n"
                        "Content-Type: %@\r\n"
                        "Content-Transfer-Encoding: binary\r\n"
                        "Accept-Ranges: bytes\r\n\r\n",
                        contentDisposition, decodedFileName, (unsigned long)[fileData length], contentType];

    send(clientSocket, [header UTF8String], [header length], 0);
    send(clientSocket, [fileData bytes], [fileData length], 0);
}

- (NSString *)contentTypeForFileAtPath:(NSString *)filePath {
    NSString *extension = [filePath pathExtension].lowercaseString;

    if ([extension isEqualToString:@"mov"]) {
        return @"video/quicktime";
    } else if ([extension isEqualToString:@"mp4"]) {
        return @"video/mp4";
    } else if ([extension isEqualToString:@"txt"]) {
        return @"text/plain";
    }

    return @"application/octet-stream";
}

- (void)serveFileListToSocket:(int)clientSocket {
    NSString *html = @"<html><head><title>Documents Folder</title></head><body>";
    html = [html stringByAppendingString:@"<h1>Files in Documents Folder:</h1><ul>"];

    NSArray *documentFiles = [self listFilesInDocumentsFolder];
    for (NSString *fileName in documentFiles) {
        NSString *encodedFileName = [fileName stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];
        html = [html stringByAppendingFormat:
                @"<li>%@ - <a href=\"/?file=%@&action=view\">View</a> | <a href=\"/?file=%@&action=download\">Download</a></li>",
                fileName, encodedFileName, encodedFileName];
    }

    html = [html stringByAppendingString:@"</ul></body></html>"];
    NSString *response = [NSString stringWithFormat:
                          @"HTTP/1.1 200 OK\r\n"
                          "Content-Type: text/html\r\n"
                          "Content-Length: %lu\r\n\r\n"
                          "%@", (unsigned long)[html length], html];

    send(clientSocket, [response UTF8String], [response length], 0);
}

- (NSArray *)listFilesInDocumentsFolder {
    NSString *documentsPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
    NSError *error;
    NSArray *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:documentsPath error:&error];
    if (error) {
        NSLog(@"Error listing files: %@", error.localizedDescription);
        return @[];
    }
    return files;
}

- (void)dealloc {
    close(self.serverSocket);
}

@end
