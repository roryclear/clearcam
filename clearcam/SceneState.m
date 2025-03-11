#import "SceneState.h"
#import "SettingsManager.h"
#import "AppDelegate.h"
#import <MobileCoreServices/MobileCoreServices.h> // Add this import
#import "pgp.h"

@implementation SceneState

- (instancetype)init {
    self.pgp = [[PGP alloc] init];
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
                        self.last_email_time = [NSDate date]; // Set to now
                        NSLog(@"sending email");
                        [self sendEmailWithImageAtPath:filePath encryptImage:NO];
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

- (void)sendEmailWithImageAtPath:(NSString *)imagePath encryptImage:(BOOL)encryptImage {
    // Server details
    NSString *server = @"http://192.168.1.100:8080";
    NSString *endpoint = @"/";

    // Email recipient address (hardcoded)
    NSString *toEmail = @"roryclear.rc@gmail.com";

    // Read image file
    NSData *imageData = [NSData dataWithContentsOfFile:imagePath];
    if (!imageData) {
        NSLog(@"Failed to read image data from path: %@", imagePath);
        return;
    }

    // Encrypt the image if enabled
    NSString *filePathToSend = imagePath;
    NSString *fileExtension = @"jpg";
    if (encryptImage) {
        // Encrypt the image and get the path to the .pgp file
        [self.pgp encryptImageWithPublicKey:imagePath];
        filePathToSend = [[imagePath stringByDeletingPathExtension] stringByAppendingPathExtension:@"pgp"];
        fileExtension = @"pgp";
        
        // Verify the encrypted file exists
        if (![[NSFileManager defaultManager] fileExistsAtPath:filePathToSend]) {
            NSLog(@"Failed to encrypt image or create .pgp file.");
            return;
        }
    }

    // Read the file to send (either the original image or the encrypted .pgp file)
    NSData *fileData = [NSData dataWithContentsOfFile:filePathToSend];
    if (!fileData) {
        NSLog(@"Failed to read file data from path: %@", filePathToSend);
        return;
    }

    // Generate a unique boundary
    NSString *boundary = [NSString stringWithFormat:@"Boundary-%@", [[NSUUID UUID] UUIDString]];

    // Construct multipart request body
    NSMutableData *bodyData = [NSMutableData data];

    // Add email field
    [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"Content-Disposition: form-data; name=\"to\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[toEmail dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];

    // Add file (either image or encrypted .pgp)
    [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[[NSString stringWithFormat:@"Content-Disposition: form-data; name=\"file\"; filename=\"%@\"\r\n", [filePathToSend lastPathComponent]] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:[[NSString stringWithFormat:@"Content-Type: %@\r\n\r\n", [self mimeTypeForFileAtPath:filePathToSend]] dataUsingEncoding:NSUTF8StringEncoding]];
    [bodyData appendData:fileData];
    [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];

    // End boundary
    [bodyData appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

    // Create URL request
    NSURL *url = [NSURL URLWithString:[server stringByAppendingString:endpoint]];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    [request setHTTPMethod:@"POST"];
    [request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
    [request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)bodyData.length] forHTTPHeaderField:@"Content-Length"];
    [request setHTTPBody:bodyData];

    // Send request using NSURLSession
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

