#import "SettingsManager.h"

@implementation SettingsManager

+ (instancetype)sharedManager {
    static SettingsManager *sharedManager = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedManager = [[self alloc] init];
    });
    return sharedManager;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        [self loadYoloIndexes];
        [self loadEvents];
        [self loadDeleteOnLaunch];
    }
    return self;
}

- (void)saveYoloIndexes {
    if (self.yolo_indexes) {
        [[NSUserDefaults standardUserDefaults] setObject:self.yolo_indexes forKey:@"yolo_indexes"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

- (void)loadYoloIndexes {
    NSArray *savedIndexes = [[NSUserDefaults standardUserDefaults] arrayForKey:@"yolo_indexes"];
    if (savedIndexes) {
        self.yolo_indexes = savedIndexes;
    } else {
        self.yolo_indexes = [self generateDefaultYoloIndexes];
    }
}

- (void)updateYoloIndexes:(NSArray<NSNumber *> *)newIndexes {
    self.yolo_indexes = newIndexes;
    [self saveYoloIndexes];
}

- (NSArray<NSNumber *> *)generateDefaultYoloIndexes {
    NSMutableArray<NSNumber *> *defaultIndexes = [NSMutableArray array];
    for (int i = 0; i < 80; i++) { //todo unhardcode 80
        [defaultIndexes addObject:@(i)];
    }
    return [defaultIndexes copy];
}

- (void)loadEvents {
    NSArray *savedEvents = [[NSUserDefaults standardUserDefaults] arrayForKey:@"events"];
    if (savedEvents) {
        self.events = savedEvents;
    } else {
        self.events = [self generateDefaultEvents];
    }
}

- (NSArray<NSNumber *> *)generateDefaultEvents {
    NSMutableArray<NSNumber *> *defaultEvents = [NSMutableArray array];
    //[defaultEvents addObject:@0]; // 1 person for now
    //[defaultEvents addObject:@2];
    [defaultEvents addObject:@63]; // laptop for now
    return [defaultEvents copy];
}

- (NSDictionary*)generateDefaultAlerts {
    NSMutableDictionary *defaultAlerts = [[NSMutableDictionary alloc] init];
    defaultAlerts[@2] = @0; // just min 0 for now
    return [defaultAlerts copy];
}

// New property: delete_on_launch
- (void)saveDeleteOnLaunch {
    [[NSUserDefaults standardUserDefaults] setBool:self.delete_on_launch forKey:@"delete_on_launch"];
    [[NSUserDefaults standardUserDefaults] synchronize];
}

- (void)loadDeleteOnLaunch {
    if ([[NSUserDefaults standardUserDefaults] objectForKey:@"delete_on_launch"] == nil) {
        NSLog(@"no delete_on_launch value found");
        self.delete_on_launch = NO;
    } else {
        self.delete_on_launch = [[NSUserDefaults standardUserDefaults] boolForKey:@"delete_on_launch"];
    }
}

- (void)updateDeleteOnLaunch:(BOOL)value {
    self.delete_on_launch = value;
    [self saveDeleteOnLaunch];
}


@end
