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
        
        NSLog(@"rory event %d = %@",i,self.events[i][0]);
        NSLog(@"rory event total = %@",self.lastN_total[self.events[i][0]]);
        
        float result = (self.lastN_total[self.events[i][0]] ? [self.lastN_total[self.events[i][0]] floatValue] : 0.0) / 10.0;
        NSLog(@"rory current state = %d", (int)roundf(result));
    }
    if(self.lastN.count > 10){
        NSLog(@"> 10 %lu",(unsigned long)self.lastN.count);
        [self.lastN removeObjectAtIndex:0];
    }
    
    return;
}

@end

