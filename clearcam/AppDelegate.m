#import "AppDelegate.h"
#import <UserNotifications/UserNotifications.h>
#import "GalleryViewController.h"

@interface AppDelegate () <UNUserNotificationCenterDelegate>
@property (nonatomic, strong) NSDictionary *pendingNotification;
@end

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    NSLog(@"App launched with options: %@", launchOptions);
    UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
    center.delegate = self;
    
    [center requestAuthorizationWithOptions:(UNAuthorizationOptionAlert | UNAuthorizationOptionSound | UNAuthorizationOptionBadge)
                          completionHandler:^(BOOL granted, NSError * _Nullable error) {
        if (granted) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [application registerForRemoteNotifications];
            });
        } else {
            NSLog(@"Notification permission denied: %@", error);
        }
    }];
    
    // Handle launch from notification
    if (launchOptions[UIApplicationLaunchOptionsRemoteNotificationKey]) {
        NSDictionary *userInfo = launchOptions[UIApplicationLaunchOptionsRemoteNotificationKey];
        self.pendingNotification = userInfo;
        NSLog(@"Launched from notification: %@", userInfo);
        // Only run background code if content-available is present (receipt, not tap)
        if (userInfo[@"aps"][@"content-available"]) {
            [self handleNotificationReceived:userInfo];
        }
    }
    
    return YES;
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
    NSLog(@"Application became active");
    if (self.pendingNotification) {
        [self handleNotificationNavigation];
        self.pendingNotification = nil;
    }
}

#pragma mark - Notification Handling

- (void)handleNotificationNavigation {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSLog(@"Handling notification navigation (user tapped)");
        UIWindow *window = [UIApplication sharedApplication].keyWindow;
        UIViewController *rootVC = window.rootViewController;
        
        UIViewController *topVC = rootVC;
        while (topVC.presentedViewController) {
            topVC = topVC.presentedViewController;
        }
        
        if ([topVC isKindOfClass:[UINavigationController class]]) {
            UINavigationController *nav = (UINavigationController *)topVC;
            if ([nav.topViewController isKindOfClass:[GalleryViewController class]]) {
                NSLog(@"GalleryViewController already visible");
                return;
            }
        }
        
        GalleryViewController *galleryVC = [[GalleryViewController alloc] init];
        
        if ([topVC isKindOfClass:[UINavigationController class]]) {
            [(UINavigationController *)topVC pushViewController:galleryVC animated:YES];
        } else if (topVC.navigationController) {
            [topVC.navigationController pushViewController:galleryVC animated:YES];
        } else {
            UINavigationController *navController = [[UINavigationController alloc] initWithRootViewController:galleryVC];
            [topVC presentViewController:navController animated:YES completion:nil];
        }
    });
}

- (void)handleNotificationReceived:(NSDictionary *)userInfo {
    NSLog(@"Notification received (user did NOT tap): %@", userInfo);
    GalleryViewController *gallery = [[GalleryViewController alloc] init];
    sleep(15);
    [gallery getEvents];
}

#pragma mark - Remote Notification Methods

- (void)application:(UIApplication *)application didRegisterForRemoteNotificationsWithDeviceToken:(NSData *)deviceToken {
    const unsigned char *dataBuffer = (const unsigned char *)[deviceToken bytes];
    if (!dataBuffer) {
        return;
    }
    NSMutableString *token = [NSMutableString stringWithCapacity:(deviceToken.length * 2)];
    for (int i = 0; i < deviceToken.length; i++) {
        [token appendFormat:@"%02x", dataBuffer[i]];
    }
    NSLog(@"Device Token: %@", token);
}

- (void)application:(UIApplication *)application didFailToRegisterForRemoteNotificationsWithError:(NSError *)error {
    NSLog(@"Failed to register for remote notifications: %@", error);
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
       willPresentNotification:(UNNotification *)notification
         withCompletionHandler:(void (^)(UNNotificationPresentationOptions))completionHandler {
    NSDictionary *userInfo = notification.request.content.userInfo;
    NSLog(@"Will present notification: %@", userInfo);
    if (userInfo[@"aps"][@"content-available"]) {
        [self handleNotificationReceived:userInfo];
    }
    completionHandler(UNNotificationPresentationOptionAlert | UNNotificationPresentationOptionSound);
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
didReceiveNotificationResponse:(UNNotificationResponse *)response
         withCompletionHandler:(void (^)(void))completionHandler {
    NSDictionary *userInfo = response.notification.request.content.userInfo;
    NSLog(@"User tapped notification: %@", userInfo);
    // Only handle navigation, no custom code here
    [self handleNotificationNavigation];
    completionHandler();
}

- (void)application:(UIApplication *)application
didReceiveRemoteNotification:(NSDictionary *)userInfo
fetchCompletionHandler:(void (^)(UIBackgroundFetchResult))completionHandler {
    NSLog(@"Received remote notification: %@", userInfo);
    
    if (userInfo[@"aps"][@"content-available"]) {
        NSLog(@"Background notification detected");
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            [self handleNotificationReceived:userInfo];
            completionHandler(UIBackgroundFetchResultNewData);
        });
    } else {
        if (application.applicationState == UIApplicationStateInactive) {
            NSLog(@"App inactive - handling navigation");
            self.pendingNotification = userInfo;
        } else if (application.applicationState == UIApplicationStateBackground) {
            NSLog(@"App in background - storing notification");
            self.pendingNotification = userInfo;
        } else {
            NSLog(@"App in foreground - regular notification received");
        }
        completionHandler(UIBackgroundFetchResultNoData);
    }
}

#pragma mark - Core Data Stack

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
