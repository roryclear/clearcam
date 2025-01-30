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

- (NSArray<NSNumber *> *)generateDefaultYoloIndexes {
    /*
    NSMutableArray<NSNumber *> *defaultIndexes = [NSMutableArray array];
    for (int i = 0; i < 80; i++) { //todo unhardcode 80
        [defaultIndexes addObject:@(i)];
    }
    */
    NSArray<NSNumber *> *defaultIndexes = @[@0, @1, @2, @3, @5, @7]; //vehicles+people preset for now (person, bicycle, car, motorcycle, bus, truck)
    return [defaultIndexes copy];
}

@end
