#import "SceneState.h"
#import "SettingsManager.h"

@implementation SceneState

- (instancetype)init {
    self = [super init];
    if (self) {
        self.lastN = [NSMutableArray array];
        self.lastN_total = [[NSMutableDictionary alloc] init];
        self.events = [SettingsManager sharedManager].events;
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
        if(current_state != last_state){
            NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
            NSString *documentsDirectory = [paths firstObject];
            NSString *filePath = [documentsDirectory stringByAppendingPathComponent:@"events.txt"];

            NSFileManager *fileManager = [NSFileManager defaultManager];
            if (![fileManager fileExistsAtPath:filePath]) {
                [fileManager createFileAtPath:filePath contents:nil attributes:nil];
            }
            NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
            [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
            NSString *timestamp = [dateFormatter stringFromDate:[NSDate date]];

            NSString *contentToWrite = [NSString stringWithFormat:@"%@ class: %@ x%@\n", timestamp, self.events[i], @(current_state)];
            NSFileHandle *fileHandle = [NSFileHandle fileHandleForWritingAtPath:filePath];
            [fileHandle seekToEndOfFile];
            [fileHandle writeData:[contentToWrite dataUsingEncoding:NSUTF8StringEncoding]];
            [fileHandle closeFile];
        }
        
    }
    if(self.lastN.count > 10){
        [self.lastN removeObjectAtIndex:0];
    }
}
@end

