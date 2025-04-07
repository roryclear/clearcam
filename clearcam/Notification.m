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

    NSString *endpoint = @"/upload";
    NSData *imageData = nil;
    NSString *filePathToSend = imagePath;

    // Generate a blank image if no path is provided
    if (imagePath.length == 0) {
        CGSize imageSize = CGSizeMake(1280, 720);
        UIGraphicsBeginImageContext(imageSize);
        CGContextRef context = UIGraphicsGetCurrentContext();
        CGContextSetFillColorWithColor(context, [UIColor whiteColor].CGColor);
        CGContextFillRect(context, CGRectMake(0, 0, imageSize.width, imageSize.height));
        UIImage *blankImage = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
        
        imageData = UIImageJPEGRepresentation(blankImage, 1.0);
        NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"test_blank_img.jpg"];
        [imageData writeToFile:tempPath atomically:YES];
        filePathToSend = tempPath;
    } else {
        imageData = [NSData dataWithContentsOfFile:imagePath];
    }

    if (!imageData) return;

    NSData *fileData = imageData;
    BOOL encryptImage = ![[NSUserDefaults standardUserDefaults] boolForKey:@"use_own_server_enabled"];
    if (encryptImage) {
        NSString *encryptionKey = [[SecretManager sharedManager] getEncryptionKey];
        if (!encryptionKey) return;
        fileData = [[SecretManager sharedManager] encryptData:imageData withKey:encryptionKey];
        if (!fileData) return;
        filePathToSend = [filePathToSend stringByAppendingString:@".aes"];
    }

    NSString *boundary = [NSString stringWithFormat:@"Boundary-%@", [[NSUUID UUID] UUIDString]];
    NSMutableData *bodyData = [NSMutableData data];

    // Add "file" field (image)
    [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[[NSString stringWithFormat:@"Content-Disposition: form-data; name=\"file\"; filename=\"%@\"\r\n", [filePathToSend lastPathComponent]] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"Content-Type: application/octet-stream\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:fileData];
    [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];

    // Include session_token if not using own notification server
    if (![[NSUserDefaults standardUserDefaults] boolForKey:@"use_own_server_enabled"]) {
        NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
        if (sessionToken) {
            [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
            [bodyData appendData:[@"Content-Disposition: form-data; name=\"session_token\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
            [bodyData appendData:[sessionToken dataUsingEncoding:NSUTF8StringEncoding]];
            [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        }
    }
    [bodyData appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

    // 🔹 STEP 1: Ask your server for a presigned R2 URL
    NSURL *uploadRequestURL = [NSURL URLWithString:[server stringByAppendingString:endpoint]];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:uploadRequestURL];
    [request setHTTPMethod:@"POST"];
    [request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
    [request setHTTPBody:bodyData];

    NSURLSessionDataTask *uploadTask = [[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"❌ Server upload failed: %@", error.localizedDescription);
            return;
        }

        NSError *jsonError;
        NSDictionary *responseDict = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError || !responseDict[@"url"]) {
            NSLog(@"❌ Failed to parse server response or missing URL");
            return;
        }

        NSString *presignedR2URL = responseDict[@"url"];  // 🔥 URL for R2 upload

        // 🔹 STEP 2: Upload the same file to Cloudflare R2
        NSMutableURLRequest *r2Request = [NSMutableURLRequest requestWithURL:[NSURL URLWithString:presignedR2URL]];
        [r2Request setHTTPMethod:@"PUT"];
        [r2Request setValue:@"application/octet-stream" forHTTPHeaderField:@"Content-Type"];

        NSURLSessionUploadTask *r2UploadTask = [[NSURLSession sharedSession] uploadTaskWithRequest:r2Request fromData:fileData completionHandler:^(NSData *r2Data, NSURLResponse *r2Response, NSError *r2Error) {
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

    [uploadTask resume];  // Start the initial server upload request
}


@end
