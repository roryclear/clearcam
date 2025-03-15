#import "Email.h"
#import "SecretManager.h"
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
    NSString *server = @"http://192.168.1.105:8080";
    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"use_own_email_server_enabled"]) {
        server = [[NSUserDefaults standardUserDefaults] valueForKey:@"own_email_server_address"];
    }
    NSString *endpoint = @"/";
    NSString *toEmail = [[NSUserDefaults standardUserDefaults] stringForKey:@"user_email"];
    if (!toEmail) return;

    NSData *imageData = nil;
    NSString *filePathToSend = imagePath;
    
    if (imagePath.length == 0) { //todo, to test user's endpoint
        NSLog(@"Image path is empty. Generating blank image.");
        CGSize imageSize = CGSizeMake(1280, 720);
        UIGraphicsBeginImageContext(imageSize);
        CGContextRef context = UIGraphicsGetCurrentContext();
        CGContextSetFillColorWithColor(context, [UIColor whiteColor].CGColor);
        CGContextFillRect(context, CGRectMake(0, 0, imageSize.width, imageSize.height));
        UIImage *blankImage = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
        
        // Convert blank image to JPEG
        imageData = UIImageJPEGRepresentation(blankImage, 1.0);
        
        // Save temporarily
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

    // Check if encryption is enabled
    BOOL encryptImage = [[NSUserDefaults standardUserDefaults] boolForKey:@"encrypt_email_data_enabled"];
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

    [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"Content-Disposition: form-data; name=\"to\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[toEmail dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];

    [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[[NSString stringWithFormat:@"Content-Disposition: form-data; name=\"file\"; filename=\"%@\"\r\n", [filePathToSend lastPathComponent]] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"Content-Type: application/octet-stream\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:fileData];
    [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
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
            NSString *responseString = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
            NSLog(@"Server Response: %@", responseString);
        }
    }];
    [task resume];
}


@end
