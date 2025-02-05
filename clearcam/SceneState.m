#import "SceneState.h"
#import "SettingsManager.h"

@implementation SceneState

- (instancetype)init {
    self = [super init];
    if (self) {
        self.lastN = [NSMutableArray array];
        self.lastN_total = [[NSMutableDictionary alloc] init];
        self.lastN_state = [[NSMutableDictionary alloc] init];
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
        NSNumber *frameValue = frame[self.events[i]] ?: @0;
        self.lastN_total[self.events[i]] = @(totalValue.intValue + frameValue.intValue);
        
        if(self.lastN.count > 10){
            totalValue = self.lastN_total[self.events[i]] ?: @0;
            frameValue = self.lastN[0][self.events[i]] ?: @0;
            self.lastN_total[self.events[i]] = @(totalValue.intValue - frameValue.intValue);
        }
        int current_state = (int)roundf((self.lastN_total[self.events[i]] ? [self.lastN_total[self.events[i]] floatValue] : 0.0) / 10.0);
        if(!self.lastN_state[self.events[i]]){
            self.lastN_state[self.events[i]] = 0;
        }
        
        if(current_state != [self.lastN_state[self.events[i]] intValue]){
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
            self.lastN_state[self.events[i]] = @(current_state);
        }
        
    }
    if(self.lastN.count > 10){
        [self.lastN removeObjectAtIndex:0];
    }
}
@end
