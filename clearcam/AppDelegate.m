#import "AppDelegate.h"
#import <UserNotifications/UserNotifications.h>
#import "GalleryViewController.h"

@interface AppDelegate () <UNUserNotificationCenterDelegate>
@property (nonatomic, strong) NSDictionary *pendingNotification;
@end

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    // Notification setup
    UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
    center.delegate = self;
    
    [center requestAuthorizationWithOptions:(UNAuthorizationOptionAlert | UNAuthorizationOptionSound | UNAuthorizationOptionBadge)
                          completionHandler:^(BOOL granted, NSError * _Nullable error) {
        if (granted) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [application registerForRemoteNotifications];
            });
        }
    }];
    
    // Check for notification launch
    if (launchOptions[UIApplicationLaunchOptionsRemoteNotificationKey]) {
        self.pendingNotification = launchOptions[UIApplicationLaunchOptionsRemoteNotificationKey];
    }
    
    return YES;
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
    if (self.pendingNotification) {
        [self handleNotificationNavigation];
        self.pendingNotification = nil;
    }
}

#pragma mark - Notification Handling

- (void)handleNotificationNavigation {
    dispatch_async(dispatch_get_main_queue(), ^{
        // Get the key window's root view controller
        UIWindow *window = [UIApplication sharedApplication].keyWindow;
        UIViewController *rootVC = window.rootViewController;
        
        // Find the topmost presented view controller
        UIViewController *topVC = rootVC;
        while (topVC.presentedViewController) {
            topVC = topVC.presentedViewController;
        }
        
        // Check if we already have a GalleryViewController visible
        if ([topVC isKindOfClass:[UINavigationController class]]) {
            UINavigationController *nav = (UINavigationController *)topVC;
            if ([nav.topViewController isKindOfClass:[GalleryViewController class]]) {
                return; // Already showing gallery
            }
        }
        
        // Create and show the GalleryViewController
        GalleryViewController *galleryVC = [[GalleryViewController alloc] init];
        
        if ([topVC isKindOfClass:[UINavigationController class]]) {
            // Push onto existing nav stack
            [(UINavigationController *)topVC pushViewController:galleryVC animated:YES];
        } else if (topVC.navigationController) {
            // Push onto existing nav controller
            [topVC.navigationController pushViewController:galleryVC animated:YES];
        } else {
            // Present modally with new nav controller
            UINavigationController *navController = [[UINavigationController alloc] initWithRootViewController:galleryVC];
            [topVC presentViewController:navController animated:YES completion:nil];
        }
    });
}

#pragma mark - Remote Notification Methods

- (void)application:(UIApplication *)application didRegisterForRemoteNotificationsWithDeviceToken:(NSData *)deviceToken {
    NSString *token = [[deviceToken description] stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"<>"]];
    token = [token stringByReplacingOccurrencesOfString:@" " withString:@""];
    NSLog(@"Device Token: %@", token);
}

- (void)application:(UIApplication *)application didFailToRegisterForRemoteNotificationsWithError:(NSError *)error {
    NSLog(@"Failed to register for remote notifications: %@", error);
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
       willPresentNotification:(UNNotification *)notification
         withCompletionHandler:(void (^)(UNNotificationPresentationOptions))completionHandler {
    // Show notification when app is in foreground
    completionHandler(UNNotificationPresentationOptionAlert | UNNotificationPresentationOptionSound);
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
didReceiveNotificationResponse:(UNNotificationResponse *)response
         withCompletionHandler:(void (^)(void))completionHandler {
    // Handle notification tap (foreground or background)
    [self handleNotificationNavigation];
    completionHandler();
}

- (void)application:(UIApplication *)application
didReceiveRemoteNotification:(NSDictionary *)userInfo
fetchCompletionHandler:(void (^)(UIBackgroundFetchResult))completionHandler {
    
    if (application.applicationState == UIApplicationStateInactive) {
        // App was in background and user tapped notification
        [self handleNotificationNavigation];
    } else if (application.applicationState == UIApplicationStateBackground) {
        // App was in background - store notification for when app becomes active
        self.pendingNotification = userInfo;
    }
    
    completionHandler(UIBackgroundFetchResultNewData);
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
