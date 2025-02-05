#import "SceneState.h"

@implementation SceneState

- (instancetype)init {
    self = [super init];
    if (self) {
        self.lastN = [NSMutableArray array];
    }
    return self;
}

@end
