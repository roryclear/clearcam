#import "SceneState.h"
#import "SettingsManager.h"
#import "AppDelegate.h"

@implementation SceneState

- (instancetype)init {
    self = [super init];
    if (self) {
        self.lastN = [NSMutableArray array];
        self.lastN_total = [[NSMutableDictionary alloc] init];
        self.events = [SettingsManager sharedManager].events;
        self.alerts = [SettingsManager sharedManager].alerts;
        self.last_email_time = [NSDate dateWithTimeIntervalSince1970:0];

        // Get Core Data context from AppDelegate
        AppDelegate *appDelegate = (AppDelegate *)[[UIApplication sharedApplication] delegate];
        self.backgroundContext = appDelegate.persistentContainer.newBackgroundContext;
    }
    return self;
}

- (void)processOutput:(NSArray *)array withImage:(CIImage *)image {
    NSMutableDictionary *frame = [[NSMutableDictionary alloc] init];
    
    // Count occurrences in the current frame
    for (int i = 0; i < array.count; i++) {
        frame[array[i][4]] = frame[array[i][4]] ? @([frame[array[i][4]] intValue] + 1) : @1;
    }
    
    [self.lastN addObject:frame]; // Store this frame's data

    // Process events
    for (int i = 0; i < self.events.count; i++) {
        NSNumber *totalValue = self.lastN_total[self.events[i]] ?: @0;
        NSNumber *last_totalValue = [totalValue copy];
        NSNumber *frameValue = frame[self.events[i]] ?: @0;
        totalValue = @(totalValue.intValue + frameValue.intValue);
        
        if (self.lastN.count > 10) {
            frameValue = self.lastN[0][self.events[i]] ?: @0;
            totalValue = @(totalValue.intValue - frameValue.intValue);
        }
        
        int current_state = (int)roundf((totalValue ? [totalValue floatValue] : 0.0) / 10.0);
        int last_state = (int)roundf((last_totalValue ? [last_totalValue floatValue] : 0.0) / 10.0);
        self.lastN_total[self.events[i]] = totalValue;
        
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
                }
            } else {
                NSLog(@"Failed to create CGImage from CIImage");
            }
            
            if (self.last_email_time && [[NSDate date] timeIntervalSinceDate:self.last_email_time] > 3600) { // only once per hour, enforce server side!
                self.last_email_time = [NSDate date]; // Set to now
                NSLog(@"sending email");
            } else {
                NSLog(@"NOT sending an email");
            }

            [self.backgroundContext performBlockAndWait:^{
                NSManagedObject *newEvent = [NSEntityDescription insertNewObjectForEntityForName:@"EventEntity"
                                                                          inManagedObjectContext:self.backgroundContext];

                [newEvent setValue:@(unixTimestamp) forKey:@"timeStamp"];
                [newEvent setValue:self.events[i] forKey:@"classType"];
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

@end

