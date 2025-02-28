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
        self.event_times = [[NSMutableArray alloc] init]; //todo init from txt on launch

        // Get Core Data context from AppDelegate
        AppDelegate *appDelegate = (AppDelegate *)[[UIApplication sharedApplication] delegate];
        self.backgroundContext = appDelegate.persistentContainer.newBackgroundContext;
    }
    return self;
}

- (void)processOutput:(NSArray *)array {
    NSMutableDictionary *frame = [[NSMutableDictionary alloc] init];
    for(int i = 0; i < array.count; i++){
        frame[array[i][4]] = frame[array[i][4]] ? @([frame[array[i][4]] intValue] + 1) : @1;
    }
    [self.lastN addObject:frame];
    
    for(int i = 0; i < self.events.count; i++){
        NSNumber *totalValue = self.lastN_total[self.events[i]] ?: @0;
        NSNumber *last_totalValue = [totalValue copy];
        NSNumber *frameValue = frame[self.events[i]] ?: @0;
        totalValue = @(totalValue.intValue + frameValue.intValue);
        
        if(self.lastN.count > 10){
            frameValue = self.lastN[0][self.events[i]] ?: @0;
            totalValue = @(totalValue.intValue - frameValue.intValue);
        }
        int current_state = (int)roundf((totalValue ? [totalValue floatValue] : 0.0) / 10.0);
        int last_state = (int)roundf((last_totalValue ? [last_totalValue floatValue] : 0.0) / 10.0);
        self.lastN_total[self.events[i]] = totalValue;
        if (current_state != last_state) {
            NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
            NSString *documentsDirectory = [paths firstObject];
            NSString *filePath = [documentsDirectory stringByAppendingPathComponent:@"events.txt"];

            NSFileManager *fileManager = [NSFileManager defaultManager];
            if (![fileManager fileExistsAtPath:filePath]) {
                [fileManager createFileAtPath:filePath contents:nil attributes:nil];
            }
            NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
            NSDate *date = [NSDate date];
            [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
            NSString *timestamp = [dateFormatter stringFromDate:date];
            [self.event_times addObject:date];

            NSString *contentToWrite = [NSString stringWithFormat:@"%@ class: %@ x%@\n", timestamp, self.events[i], @(current_state)];
            NSFileHandle *fileHandle = [NSFileHandle fileHandleForWritingAtPath:filePath];
            [fileHandle seekToEndOfFile];
            [fileHandle writeData:[contentToWrite dataUsingEncoding:NSUTF8StringEncoding]];
            [fileHandle closeFile];

            if (current_state == [self.alerts[self.events[i]] intValue]) {
                filePath = [documentsDirectory stringByAppendingPathComponent:@"alerts.txt"];
                if (![fileManager fileExistsAtPath:filePath]) {
                    [fileManager createFileAtPath:filePath contents:nil attributes:nil];
                }
                fileHandle = [NSFileHandle fileHandleForWritingAtPath:filePath];
                [fileHandle seekToEndOfFile];
                [fileHandle writeData:[contentToWrite dataUsingEncoding:NSUTF8StringEncoding]];
                [fileHandle closeFile];
            }

            // **Add EventEntity to Core Data (SegmentsModel)**
            [self.backgroundContext performBlockAndWait:^{
                NSManagedObject *newEvent = [NSEntityDescription insertNewObjectForEntityForName:@"EventEntity"
                                                                          inManagedObjectContext:self.backgroundContext];
                [newEvent setValue:@([date timeIntervalSince1970]) forKey:@"timeStamp"]; // Float value of timestamp

                NSError *saveError = nil;
                if (![self.backgroundContext save:&saveError]) {
                    NSLog(@"Failed to save EventEntity: %@, %@", saveError, saveError.userInfo);
                }
            }];
        }
    }
    if(self.lastN.count > 10){
        [self.lastN removeObjectAtIndex:0];
    }
}
@end

