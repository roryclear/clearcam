#import "Email.h"
#import "SecretManager.h"
#import "StoreManager.h"
#import <UIKit/UIKit.h>

@implementation Email

+ (instancetype)sharedInstance {
    static Email *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (void)sendEmailWithImageAtPath:(NSString *)imagePath {
    NSString *server = @"https://www.rors.ai";
    
    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"use_own_server_enabled"]) {
        server = [[NSUserDefaults standardUserDefaults] valueForKey:@"own_email_server_address"];
        if (![server hasPrefix:@"http"]) {
            server = [@"http://" stringByAppendingString:server];
        }
    }

    NSString *endpoint = @"/send";
    NSData *imageData = nil;
    NSString *filePathToSend = imagePath;

    if (imagePath.length == 0) {
        NSLog(@"Image path is empty. Generating blank image.");
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

    if (!imageData) {
        NSLog(@"Failed to read image data.");
        return;
    }

    NSData *fileData = imageData;
    BOOL encryptImage = [[NSUserDefaults standardUserDefaults] boolForKey:@"send_notif_enabled"];
    if (encryptImage) {
        NSString *encryptionKey = [[SecretManager sharedManager] getEncryptionKey];
        if (!encryptionKey) {
            NSLog(@"Encryption key not found. Encryption aborted.");
            return;
        }
        fileData = [[SecretManager sharedManager] encryptData:imageData withKey:encryptionKey];
        if (!fileData) {
            NSLog(@"Encryption failed.");
            return;
        }
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

    // Include session_token only if not using own email server
    if (![[NSUserDefaults standardUserDefaults] boolForKey:@"use_own_server_enabled"]) {
        NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
        if (sessionToken) {
            [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
            [bodyData appendData:[@"Content-Disposition: form-data; name=\"session_token\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
            [bodyData appendData:[sessionToken dataUsingEncoding:NSUTF8StringEncoding]];
            [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        } else {
            NSLog(@"Warning: No session token found in Keychain.");
        }
    }

    [bodyData appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

    NSURL *url = [NSURL URLWithString:[server stringByAppendingString:endpoint]];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    [request setHTTPMethod:@"POST"];
    [request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
    [request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)bodyData.length] forHTTPHeaderField:@"Content-Length"];
    [request setHTTPBody:bodyData];

    NSURLSession *session = [NSURLSession sharedSession];
    NSURLSessionDataTask *task = [session dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"Error: %@", error.localizedDescription);
        } else {
            NSLog(@"Send request completed successfully.");
        }
    }];
    [task resume];
}

@end
