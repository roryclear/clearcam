#import "AppDelegate.h"

@interface AppDelegate ()
@end

@implementation AppDelegate

@synthesize persistentContainer = _persistentContainer;

- (NSPersistentContainer *)persistentContainer {
    if (_persistentContainer == nil) {
        _persistentContainer = [[NSPersistentContainer alloc] initWithName:@"SegmentsModel"];
        [_persistentContainer loadPersistentStoresWithCompletionHandler:^(NSPersistentStoreDescription *storeDescription, NSError *error) {
            if (error) {
                NSLog(@"Unresolved error %@, %@", error, error.userInfo);
                abort();
            }
        }];
    }
    return _persistentContainer;
}
@end
