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

    // Retrieve session token
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken) {
        NSLog(@"❌ No session token available");
        return;
    }

    // 🔹 STEP 1: Request a Presigned R2 URL from the Server
    NSURLComponents *components = [NSURLComponents componentsWithString:[server stringByAppendingString:@"/upload"]];
    NSURLQueryItem *fileItem = [NSURLQueryItem queryItemWithName:@"filename" value:fileName];
    NSURLQueryItem *sessionItem = [NSURLQueryItem queryItemWithName:@"session_token" value:sessionToken];
    components.queryItems = @[fileItem, sessionItem];  // Include session_token here

    NSURL *uploadRequestURL = components.URL;
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:uploadRequestURL];
    [request setHTTPMethod:@"GET"];

    NSURLSessionDataTask *uploadTask = [[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"❌ Server request failed: %@", error.localizedDescription);
            return;
        }

        NSError *jsonError;
        NSDictionary *responseDict = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError || !responseDict[@"url"]) {
            NSLog(@"❌ Failed to parse server response or missing URL");
            return;
        }

        NSString *presignedR2URL = responseDict[@"url"];  // 🔥 URL for R2 upload

        // 🔹 STEP 2: Upload the File to Cloudflare R2
        NSMutableURLRequest *r2Request = [NSMutableURLRequest requestWithURL:[NSURL URLWithString:presignedR2URL]];
        [r2Request setHTTPMethod:@"PUT"];
        [r2Request setValue:@"application/octet-stream" forHTTPHeaderField:@"Content-Type"];

        NSURLSessionUploadTask *r2UploadTask = [[NSURLSession sharedSession] uploadTaskWithRequest:r2Request fromData:imageData completionHandler:^(NSData *r2Data, NSURLResponse *r2Response, NSError *r2Error) {
            if (r2Error) {
                NSLog(@"❌ R2 upload failed: %@", r2Error.localizedDescription);
            } else {
                NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)r2Response;
                if (httpResponse.statusCode == 200) {
                    NSLog(@"✅ Successfully uploaded to Cloudflare R2!");
                } else {
                    NSLog(@"❌ R2 upload failed with HTTP %ld", (long)httpResponse.statusCode);
                }
            }
        }];

        [r2UploadTask resume];  // Start the R2 upload
    }];

    [uploadTask resume];  // Start the request to get the presigned URL
}




@end
