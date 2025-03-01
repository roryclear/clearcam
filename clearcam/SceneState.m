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

        // Get Core Data context from AppDelegate
        AppDelegate *appDelegate = (AppDelegate *)[[UIApplication sharedApplication] delegate];
        self.backgroundContext = appDelegate.persistentContainer.newBackgroundContext;
    }
    return self;
}

- (void)processOutput:(NSArray *)array {
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
            // Get current timestamp
            NSDate *date = [NSDate date]; // NOW inside the loop (fix)
            NSTimeInterval unixTimestamp = [date timeIntervalSince1970]; // Float UNIX timestamp
            
            // Format human-readable timestamp for logging
            NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
            [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
            NSString *formattedTimestamp = [dateFormatter stringFromDate:date];

            // Logging
            NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
            NSString *documentsDirectory = [paths firstObject];

            NSFileManager *fileManager = [NSFileManager defaultManager];
            NSString *contentToWrite = [NSString stringWithFormat:@"%@ class: %@ x%@\n",
                                        formattedTimestamp, self.events[i], @(current_state)];

            // Handle alerts.txt if needed
            if (current_state == [self.alerts[self.events[i]] intValue]) {
                NSString *filePath = [documentsDirectory stringByAppendingPathComponent:@"alerts.txt"];
                if (![fileManager fileExistsAtPath:filePath]) {
                    [fileManager createFileAtPath:filePath contents:nil attributes:nil];
                }
                NSFileHandle *fileHandle = [NSFileHandle fileHandleForWritingAtPath:filePath];
                [fileHandle seekToEndOfFile];
                [fileHandle writeData:[contentToWrite dataUsingEncoding:NSUTF8StringEncoding]];
                [fileHandle closeFile];
            }

            // **Add EventEntity to Core Data (SegmentsModel)**
            [self.backgroundContext performBlockAndWait:^{
                NSManagedObject *newEvent = [NSEntityDescription insertNewObjectForEntityForName:@"EventEntity"
                                                                          inManagedObjectContext:self.backgroundContext];

                [newEvent setValue:@(unixTimestamp) forKey:@"timeStamp"]; // Correct per-event timestamp
                [newEvent setValue:self.events[i] forKey:@"classType"]; // Class type
                [newEvent setValue:@(current_state) forKey:@"quantity"]; // Quantity (current state)

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

