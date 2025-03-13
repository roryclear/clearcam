#import "SceneState.h"
#import "SettingsManager.h"
#import "SecretManager.h"
#import "AppDelegate.h"
#import <MobileCoreServices/MobileCoreServices.h> // Add this import
#import <CommonCrypto/CommonCryptor.h>

#define MAGIC_NUMBER 0x4D41474943ULL // "MAGIC" in ASCII as a 64-bit value
#define HEADER_SIZE (sizeof(uint64_t)) // Size of the magic number (8 bytes)

@implementation SceneState

- (instancetype)init {
    self = [super init];
    if (self) {
        self.lastN = [NSMutableArray array];
        self.lastN_total = [[NSMutableDictionary alloc] init];
        self.alerts = [SettingsManager sharedManager].alerts;
        self.last_email_time = [NSDate dateWithTimeIntervalSince1970:0];

        // Get Core Data context from AppDelegate
        AppDelegate *appDelegate = (AppDelegate *)[[UIApplication sharedApplication] delegate];
        self.backgroundContext = appDelegate.persistentContainer.newBackgroundContext;
    }
    return self;
}

- (void)processOutput:(NSArray *)array withImage:(CIImage *)image {
    NSArray *events = [[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"][[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_preset_idx"]];
    if(!events) return;
    NSMutableDictionary *frame = [[NSMutableDictionary alloc] init];
    
    // Count occurrences in the current frame
    for (int i = 0; i < array.count; i++) {
        frame[array[i][4]] = frame[array[i][4]] ? @([frame[array[i][4]] intValue] + 1) : @1;
    }
    
    [self.lastN addObject:frame]; // Store this frame's data

    // Process events
    for (int i = 0; i < events.count; i++) {
        NSNumber *totalValue = self.lastN_total[events[i]] ?: @0;
        NSNumber *last_totalValue = [totalValue copy];
        NSNumber *frameValue = frame[events[i]] ?: @0;
        totalValue = @(totalValue.intValue + frameValue.intValue);
        
        if (self.lastN.count > 10) {
            frameValue = self.lastN[0][events[i]] ?: @0;
            totalValue = @(totalValue.intValue - frameValue.intValue);
        }
        
        int current_state = (int)roundf((totalValue ? [totalValue floatValue] : 0.0) / 10.0);
        int last_state = (int)roundf((last_totalValue ? [last_totalValue floatValue] : 0.0) / 10.0);
        self.lastN_total[events[i]] = totalValue;
        
        if (current_state != last_state) {
            NSDate *date = [NSDate date];
            NSTimeInterval unixTimestamp = [date timeIntervalSince1970];
            long long roundedUnixTimestamp = (long long)unixTimestamp;

            // Convert CIImage to CGImage
            CIContext *ciContext = [CIContext context];
            CGImageRef cgImage = [ciContext createCGImage:image fromRect:image.extent];

            if (cgImage) {
                // Convert CGImage to UIImage
                UIImage *uiImage = [UIImage imageWithCGImage:cgImage];

                // Release CGImage since it was manually created
                CGImageRelease(cgImage);

                // Get the app's Documents directory
                NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
                NSString *documentsDirectory = [paths firstObject];
                NSString *imagesDirectory = [documentsDirectory stringByAppendingPathComponent:@"images"];

                // Create the images directory if it doesn't exist
                if (![[NSFileManager defaultManager] fileExistsAtPath:imagesDirectory]) {
                    [[NSFileManager defaultManager] createDirectoryAtPath:imagesDirectory
                                              withIntermediateDirectories:YES
                                                               attributes:nil
                                                                    error:nil];
                }

                // File path for the image
                NSString *filePath = [imagesDirectory stringByAppendingPathComponent:[NSString stringWithFormat:@"%lld.jpg", roundedUnixTimestamp]];  // Changed file extension to .jpg

                // Convert UIImage to JPEG data
                NSData *imageData = UIImageJPEGRepresentation(uiImage, 1.0); // 1.0 for maximum quality

                // Write the image data to file
                if (![imageData writeToFile:filePath atomically:YES]) {
                    NSLog(@"Failed to save image at path: %@", filePath);
                } else {
                    NSLog(@"Image saved at path: %@", filePath);
                    if (self.last_email_time && [[NSDate date] timeIntervalSinceDate:self.last_email_time] > 300) { // only once per hour? enforce server side!
                        NSLog(@"sending email");
                        if ([[NSUserDefaults standardUserDefaults] boolForKey:@"send_email_alerts_enabled"]) {
                            [self sendEmailWithImageAtPath:filePath];
                            self.last_email_time = [NSDate date]; // Set to now
                        }
                    } else {
                        NSLog(@"NOT sending an email");
                    }
                }
            } else {
                NSLog(@"Failed to create CGImage from CIImage");
            }
            
            [self.backgroundContext performBlockAndWait:^{
                NSManagedObject *newEvent = [NSEntityDescription insertNewObjectForEntityForName:@"EventEntity"
                                                                          inManagedObjectContext:self.backgroundContext];

                [newEvent setValue:@(unixTimestamp) forKey:@"timeStamp"];
                [newEvent setValue:events[i] forKey:@"classType"];
                [newEvent setValue:@(current_state) forKey:@"quantity"];

                NSError *saveError = nil;
                if (![self.backgroundContext save:&saveError]) {
                    NSLog(@"Failed to save EventEntity: %@, %@", saveError, saveError.userInfo);
                }
            }];
        }

    }
    
    // Keep lastN size within 10
    if (self.lastN.count > 10) {
        [self.lastN removeObjectAtIndex:0];
    }
}

- (NSData *)encryptData:(NSData *)data withKey:(NSString *)key {
    // Step 1: Prepare the header (magic number)
    uint64_t magicNumber = MAGIC_NUMBER;
    NSMutableData *headerData = [NSMutableData dataWithBytes:&magicNumber length:HEADER_SIZE];
    
    // Step 2: Combine header and original data
    NSMutableData *dataToEncrypt = [NSMutableData data];
    [dataToEncrypt appendData:headerData];
    [dataToEncrypt appendData:data];
    
    // Step 3: Encrypt the combined data
    char keyPtr[kCCKeySizeAES256 + 1]; // Buffer for key
    bzero(keyPtr, sizeof(keyPtr)); // Zero out buffer
    [key getCString:keyPtr maxLength:sizeof(keyPtr) encoding:NSUTF8StringEncoding];
    
    size_t bufferSize = dataToEncrypt.length + kCCBlockSizeAES128;
    void *buffer = malloc(bufferSize);
    
    size_t numBytesEncrypted = 0;
    CCCryptorStatus status = CCCrypt(kCCEncrypt,
                                     kCCAlgorithmAES,
                                     kCCOptionPKCS7Padding,
                                     keyPtr,
                                     kCCKeySizeAES256,
                                     NULL, // IV (NULL for no IV)
                                     dataToEncrypt.bytes,
                                     dataToEncrypt.length,
                                     buffer,
                                     bufferSize,
                                     &numBytesEncrypted);
    
    if (status == kCCSuccess) {
        return [NSData dataWithBytesNoCopy:buffer length:numBytesEncrypted];
    }
    
    free(buffer);
    return nil;
}

- (void)sendEmailWithImageAtPath:(NSString *)imagePath {
    NSString *server = @"http://192.168.1.100:8080";
    NSString *endpoint = @"/";
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
        fileData = [self encryptData:imageData withKey:encryptionKey];
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

// Helper function to get MIME type for a file
- (NSString *)mimeTypeForFileAtPath:(NSString *)path {
    CFStringRef fileExtension = (__bridge CFStringRef)[path pathExtension];
    CFStringRef UTI = UTTypeCreatePreferredIdentifierForTag(kUTTagClassFilenameExtension, fileExtension, NULL);
    CFStringRef mimeType = UTTypeCopyPreferredTagWithClass(UTI, kUTTagClassMIMEType);
    CFRelease(UTI);
    return (__bridge_transfer NSString *)mimeType ?: @"application/octet-stream";
}

@end
