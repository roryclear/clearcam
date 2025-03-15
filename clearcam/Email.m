#import "Email.h"
#import "SecretManager.h"

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
    NSString *endpoint = @"/send";
    NSString *toEmail = [[NSUserDefaults standardUserDefaults] stringForKey:@"user_email"];
    if (!toEmail) return;

    NSData *imageData = [NSData dataWithContentsOfFile:imagePath];
    if (!imageData) {
        NSLog(@"Failed to read image data from path: %@", imagePath);
        return;
    }

    NSData *fileData = imageData;
    NSString *filePathToSend = imagePath;
    
    // Check if encryption is enabled in SettingsManager
    BOOL encryptImage = [[NSUserDefaults standardUserDefaults] boolForKey:@"encrypt_email_data_enabled"];
    
    if (encryptImage) {
        // Retrieve the user's password from the Keychain
        NSString *encryptionKey = [[SecretManager sharedManager] getEncryptionKey];

        if (!encryptionKey) {
            NSLog(@"Encryption key not found in Keychain. Encryption aborted.");
            return;
        }
        
        // Encrypt the image data using the retrieved key
        fileData = [[SecretManager sharedManager] encryptData:imageData withKey:encryptionKey];
        if (!fileData) {
            NSLog(@"Encryption failed.");
            return;
        }
    }


    NSString *boundary = [NSString stringWithFormat:@"Boundary-%@", [[NSUUID UUID] UUIDString]];
    NSMutableData *bodyData = [NSMutableData data];

    [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"Content-Disposition: form-data; name=\"to\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[toEmail dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];

    [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[[NSString stringWithFormat:@"Content-Disposition: form-data; name=\"file\"; filename=\"%@\"\r\n", encryptImage ? [[filePathToSend lastPathComponent] stringByAppendingString:@".aes"] : [filePathToSend lastPathComponent]
] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[[NSString stringWithFormat:@"Content-Type: application/octet-stream\r\n\r\n"] dataUsingEncoding:NSUTF8StringEncoding]];
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
