#import "SceneState.h"
#import "SettingsManager.h"
#import "SecretManager.h"
#import "AppDelegate.h"
#import "StoreManager.h"
#import "FileServer.h"
#import "notification.h"
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
        self.last_notif_time = [NSDate dateWithTimeIntervalSince1970:0];

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
        
        if (current_state != last_state && self.last_notif_time && [[NSDate date] timeIntervalSinceDate:self.last_notif_time] > 60) { //one event per min for now
            self.last_notif_time = [NSDate now];
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
                NSString *filePath = [imagesDirectory stringByAppendingPathComponent:[NSString stringWithFormat:@"%lld.jpg", roundedUnixTimestamp]];
                NSData *imageData = UIImageJPEGRepresentation(uiImage, 1.0); // 1.0 for maximum quality
                NSString *filePathSmall = [imagesDirectory stringByAppendingPathComponent:[NSString stringWithFormat:@"%lld_small.jpg", roundedUnixTimestamp]];
                // Write the image data to file
                if ([imageData writeToFile:filePath atomically:YES]) {
                    //thumbnail
                    UIGraphicsBeginImageContextWithOptions(CGSizeMake(uiImage.size.width / 2, uiImage.size.height / 2), YES, 1.0);
                    [uiImage drawInRect:CGRectMake(0, 0, uiImage.size.width / 2, uiImage.size.height / 2)];
                    UIImage *resizedImage = UIGraphicsGetImageFromCurrentImageContext();
                    UIGraphicsEndImageContext();
                    NSData *lowResImageData = UIImageJPEGRepresentation(resizedImage, 0.7);
                    [lowResImageData writeToFile:filePathSmall atomically:YES];
                    
                    if(![[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"] || ![[NSDate date] compare:[[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"]] || [[NSDate date] compare:[[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"]] == NSOrderedDescending){
                        NSLog(@"verifying subscription");
                        [[StoreManager sharedInstance] verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
                            [self sendnotification:filePath];
                        }];
                    } else {
                        [self sendnotification:filePath];
                    }
                }
            }
            [self.backgroundContext performBlockAndWait:^{
                NSManagedObject *newEvent = [NSEntityDescription insertNewObjectForEntityForName:@"EventEntity"
                                                                          inManagedObjectContext:self.backgroundContext];

                [newEvent setValue:@(unixTimestamp) forKey:@"timeStamp"];
                [newEvent setValue:events[i] forKey:@"classType"];
                [newEvent setValue:@(current_state) forKey:@"quantity"];

                NSError *saveError = nil;
                [self.backgroundContext save:&saveError];
            }];
        }

    }
    
    // Keep lastN size within 10
    if (self.lastN.count > 10) {
        [self.lastN removeObjectAtIndex:0];
    }
}

- (void)sendnotification:(NSString *)filePath {
    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"send_notif_enabled"] || !([[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"] || [[NSUserDefaults standardUserDefaults] boolForKey:@"use_own_server_enabled"])){ //todo add back other stuff
        NSDate *now = [NSDate date];
        NSDateComponents *components = [[NSCalendar currentCalendar] components:(NSCalendarUnitHour | NSCalendarUnitMinute | NSCalendarUnitWeekday) fromDate:now];
        NSInteger currentTime = components.hour * 60 + components.minute;
        NSString *currentDay = @[@"Sun", @"Mon", @"Tue", @"Wed", @"Thu", @"Fri", @"Sat"][components.weekday - 1];

        for (NSDictionary *schedule in [[NSUserDefaults standardUserDefaults] arrayForKey:@"notification_schedules"] ?: @[]) {
            if (![schedule[@"enabled"] boolValue]) continue;
            if (![schedule[@"days"] containsObject:currentDay]) continue;

            NSInteger startTime = [schedule[@"startHour"] integerValue] * 60 + [schedule[@"startMinute"] integerValue];
            NSInteger endTime = [schedule[@"endHour"] integerValue] * 60 + [schedule[@"endMinute"] integerValue];

            if (currentTime >= startTime && currentTime <= endTime) {
                [[notification sharedInstance] sendNotification];
                if([FileServer sharedInstance].segment_length > 3) [FileServer sharedInstance].segment_length = 3;
                NSTimeInterval start = [[NSDate dateWithTimeIntervalSinceNow:-7.5] timeIntervalSince1970];
                NSTimeInterval end = [[NSDate dateWithTimeIntervalSinceNow:7.5] timeIntervalSince1970];
                id context = [FileServer sharedInstance].context;
                dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC)), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                    NSString *filePath = [[FileServer sharedInstance] processVideoDownloadWithLowRes:YES startTime:start endTime:end context:context];
                    [[notification sharedInstance] uploadImageAtPath:filePath];
                });
                break;
            }
        }
    }
}

@end
