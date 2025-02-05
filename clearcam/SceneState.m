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
        NSNumber *totalValue = self.lastN_total[self.events[i][0]] ?: @0;
        NSNumber *frameValue = frame[self.events[i][0]] ?: @0;
        self.lastN_total[self.events[i][0]] = @(totalValue.intValue + frameValue.intValue);
        
        if(self.lastN.count > 10){
            totalValue = self.lastN_total[self.events[i][0]] ?: @0;
            frameValue = self.lastN[0][self.events[i][0]] ?: @0;
            self.lastN_total[self.events[i][0]] = @(totalValue.intValue - frameValue.intValue);
        }
        
        //NSLog(@"rory event %d = %@",i,self.events[i][0]);
        //NSLog(@"rory event total = %@",self.lastN_total[self.events[i][0]]);
        
        int current_state = (int)roundf((self.lastN_total[self.events[i][0]] ? [self.lastN_total[self.events[i][0]] floatValue] : 0.0) / 10.0);
        //NSLog(@"rory current state = %d", current_state);
        if(!self.lastN_state[self.events[i][0]]){
            self.lastN_state[self.events[i][0]] = 0;
        }
        
        if(current_state != [self.lastN_state[self.events[i][0]] intValue]){
            if(current_state > [self.lastN_state[self.events[i][0]] intValue]){
                NSLog(@"INCREASE");
                if(current_state >= [self.events[i][1] intValue]){
                    //todo, move elsewhere
                    NSLog(@"event threshold");
                    // Get the path to the documents directory
                    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
                    NSString *documentsDirectory = [paths firstObject];
                    NSString *filePath = [documentsDirectory stringByAppendingPathComponent:@"events.txt"];
                    
                    NSFileManager *fileManager = [NSFileManager defaultManager];
                    if (![fileManager fileExistsAtPath:filePath]) {
                        [fileManager createFileAtPath:filePath contents:nil attributes:nil];
                    }
                    NSString *timestamp = [[NSDate date] description];
                    NSString *contentToWrite = [NSString stringWithFormat:@"%@ %@ x%@\n", timestamp, self.events[i][0],@(current_state)];
                    NSFileHandle *fileHandle = [NSFileHandle fileHandleForWritingAtPath:filePath];
                    [fileHandle seekToEndOfFile];
                    [fileHandle writeData:[contentToWrite dataUsingEncoding:NSUTF8StringEncoding]];
                    [fileHandle closeFile];
                }
            }
            self.lastN_state[self.events[i][0]] = @(current_state);
        }
        
    }
    if(self.lastN.count > 10){
        [self.lastN removeObjectAtIndex:0];
    }
    
    return;
}

- (void)writeToFileWithString:(NSString *)customString { //todo
    NSString *filePath = [NSHomeDirectory() stringByAppendingPathComponent:@"events.txt"];
    NSFileManager *fileManager = [NSFileManager defaultManager];

    // Check if file exists
    if (![fileManager fileExistsAtPath:filePath]) {
        // Create file if it doesn't exist
        [fileManager createFileAtPath:filePath contents:nil attributes:nil];
    }

    // Get the current timestamp
    NSString *timestamp = [[NSDate date] description];
    
    // Create the string to be written
    NSString *contentToWrite = [NSString stringWithFormat:@"%@ %@\n", timestamp, customString];
    
    // Write the content to the file
    NSFileHandle *fileHandle = [NSFileHandle fileHandleForWritingAtPath:filePath];
    [fileHandle seekToEndOfFile];
    [fileHandle writeData:[contentToWrite dataUsingEncoding:NSUTF8StringEncoding]];
    [fileHandle closeFile];
}

@end

