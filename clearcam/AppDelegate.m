#import "AppDelegate.h"
#import <UserNotifications/UserNotifications.h>
#import "GalleryViewController.h"
#import "StoreManager.h"
#import "FileServer.h"

@interface AppDelegate () <UNUserNotificationCenterDelegate>
@property (nonatomic, strong) NSDictionary *pendingNotification;
@end

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
    center.delegate = self;

    // Handle launch from notification
    if (launchOptions[UIApplicationLaunchOptionsRemoteNotificationKey]) {
        NSDictionary *userInfo = launchOptions[UIApplicationLaunchOptionsRemoteNotificationKey];
        self.pendingNotification = userInfo;
        // Only process if notifications are enabled
        if ([[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"] && userInfo[@"aps"][@"content-available"]) {
            [self handleNotificationReceived:userInfo];
        }
    }
    
    return YES;
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
    if (self.pendingNotification && [[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"]) {
        [self handleNotificationNavigation];
        self.pendingNotification = nil;
    }
}

#pragma mark - Notification Handling

- (void)handleNotificationNavigation {
    dispatch_async(dispatch_get_main_queue(), ^{
        UIWindow *window = [UIApplication sharedApplication].keyWindow;
        UIViewController *rootVC = window.rootViewController;
        
        UIViewController *topVC = rootVC;
        while (topVC.presentedViewController) {
            topVC = topVC.presentedViewController;
        }
        
        if ([topVC isKindOfClass:[UINavigationController class]]) {
            UINavigationController *nav = (UINavigationController *)topVC;
            if ([nav.topViewController isKindOfClass:[GalleryViewController class]]) return;
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
    GalleryViewController *gallery = [[GalleryViewController alloc] init];
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), ^{
        sleep(15);
        [gallery getEvents];
    });
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
    [[NSUserDefaults standardUserDefaults] setObject:token forKey:@"device_token"];
    [[NSUserDefaults standardUserDefaults] synchronize];
    [FileServer sendDeviceTokenToServer];
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
       willPresentNotification:(UNNotification *)notification
         withCompletionHandler:(void (^)(UNNotificationPresentationOptions))completionHandler {
    NSDictionary *userInfo = notification.request.content.userInfo;
    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"]) {
        if (userInfo[@"aps"][@"content-available"]) {
            [self handleNotificationReceived:userInfo];
        }
        completionHandler(UNNotificationPresentationOptionAlert | UNNotificationPresentationOptionSound);
    } else {
        completionHandler(0); // No presentation options when disabled
    }
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
didReceiveNotificationResponse:(UNNotificationResponse *)response
         withCompletionHandler:(void (^)(void))completionHandler {
    NSDictionary *userInfo = response.notification.request.content.userInfo;
    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"]) {
        [self handleNotificationNavigation];
    }
    completionHandler();
}

- (void)application:(UIApplication *)application
didReceiveRemoteNotification:(NSDictionary *)userInfo
fetchCompletionHandler:(void (^)(UIBackgroundFetchResult))completionHandler {
    if (![[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"]) {
        completionHandler(UIBackgroundFetchResultNoData);
        return;
    }
    
    if (userInfo[@"aps"][@"content-available"]) {
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            [self handleNotificationReceived:userInfo];
            completionHandler(UIBackgroundFetchResultNewData);
        });
    } else {
        if (application.applicationState == UIApplicationStateInactive || application.applicationState == UIApplicationStateBackground) self.pendingNotification = userInfo;
        completionHandler(UIBackgroundFetchResultNoData);
    }
}

@synthesize persistentContainer = _persistentContainer;

- (NSPersistentContainer *)persistentContainer {
    if (_persistentContainer == nil) {
        _persistentContainer = [[NSPersistentContainer alloc] initWithName:@"SegmentsModel"];
        [_persistentContainer loadPersistentStoresWithCompletionHandler:^(NSPersistentStoreDescription *storeDescription, NSError *error) {
            if (error) abort();
        }];
    }
    return _persistentContainer;
}

@end
