#import "notification.h"
#import "SecretManager.h"
#import "StoreManager.h"
#import "FileServer.h"
#import <UIKit/UIKit.h>

@implementation notification

+ (instancetype)sharedInstance {
    static notification *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (void)sendNotification {
    NSString *server = @"https://www.rors.ai";
    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"use_own_server_enabled"]) {
        server = [[NSUserDefaults standardUserDefaults] valueForKey:@"own_notification_server_address"];
        if(!server) server = @"http://192.168.1.1:8080";
        if (![server hasPrefix:@"http"]) {
            server = [@"http://" stringByAppendingString:server];
        }
    }

    NSString *endpoint = @"/send";
    NSString *boundary = [NSString stringWithFormat:@"Boundary-%@", [[NSUUID UUID] UUIDString]];
    NSMutableData *bodyData = [NSMutableData data];

    // Retrieve session token
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken) return;

    // Add "session_token" field
    [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"Content-Disposition: form-data; name=\"session_token\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[sessionToken dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

    [FileServer performPostRequestWithURL:[server stringByAppendingString:endpoint]
                                   method:@"POST"
                              contentType:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary]
                                     body:bodyData
                        completionHandler:^(NSData *data, NSHTTPURLResponse *response, NSError *error) {}];
}


- (void)uploadImageAtPath:(NSString *)imagePath {
    NSString *server = @"https://www.rors.ai";

    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"use_own_server_enabled"]) {
        server = [[NSUserDefaults standardUserDefaults] valueForKey:@"own_notification_server_address"];
        if(!server) server = @"http://192.168.1.1:8080";
        if (![server hasPrefix:@"http"]) {
            server = [@"http://" stringByAppendingString:server];
        }
    }

    if (imagePath.length == 0) return;

    NSData *imageData = [NSData dataWithContentsOfFile:imagePath];
    if (!imageData) return;

    NSString *fileName = [imagePath lastPathComponent];
    BOOL encryptImage = ![[NSUserDefaults standardUserDefaults] boolForKey:@"use_own_server_enabled"];
    if (encryptImage) {
        NSString *encryptionKey = [[SecretManager sharedManager] getEncryptionKey];
        if (!encryptionKey) return;
        NSData *encryptedData = [[SecretManager sharedManager] encryptData:imageData withKey:encryptionKey];
        if (!encryptedData) return;
        fileName = [fileName stringByAppendingString:@".aes"];
        imageData = encryptedData;
    }

    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken) return;
    NSUInteger fileSize = [imageData length];
    NSURLComponents *components = [NSURLComponents componentsWithString:[server stringByAppendingString:@"/upload"]];
    NSURLQueryItem *fileItem = [NSURLQueryItem queryItemWithName:@"filename" value:fileName];
    NSURLQueryItem *sessionItem = [NSURLQueryItem queryItemWithName:@"session_token" value:sessionToken];
    NSURLQueryItem *sizeItem = [NSURLQueryItem queryItemWithName:@"size" value:[NSString stringWithFormat:@"%lu", (unsigned long)fileSize]];
    components.queryItems = @[fileItem, sessionItem, sizeItem];

    NSURL *uploadRequestURL = components.URL;
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:uploadRequestURL];
    [request setHTTPMethod:@"GET"];

    NSURLSessionDataTask *uploadTask = [[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) return;
        NSError *jsonError;
        NSDictionary *responseDict = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError || !responseDict[@"url"]) return;
        NSString *presignedR2URL = responseDict[@"url"];
        NSMutableURLRequest *r2Request = [NSMutableURLRequest requestWithURL:[NSURL URLWithString:presignedR2URL]];
        [r2Request setHTTPMethod:@"PUT"];
        [r2Request setValue:@"application/octet-stream" forHTTPHeaderField:@"Content-Type"];
        [r2Request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)fileSize] forHTTPHeaderField:@"Content-Length"]; // Add Content-Length header

        NSURLSessionUploadTask *r2UploadTask = [[NSURLSession sharedSession] uploadTaskWithRequest:r2Request fromData:imageData completionHandler:^(NSData *r2Data, NSURLResponse *r2Response, NSError *r2Error) {
            if (!r2Error) {
                NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)r2Response;
            }
        }];

        [r2UploadTask resume];
    }];

    [uploadTask resume];
}

@end

