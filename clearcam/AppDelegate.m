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
    NSLog(@"App launched with options: %@", launchOptions);
    UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
    center.delegate = self;

    // Handle launch from notification
    if (launchOptions[UIApplicationLaunchOptionsRemoteNotificationKey]) {
        NSDictionary *userInfo = launchOptions[UIApplicationLaunchOptionsRemoteNotificationKey];
        self.pendingNotification = userInfo;
        NSLog(@"Launched from notification: %@", userInfo);
        // Only process if notifications are enabled
        if ([[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"] && userInfo[@"aps"][@"content-available"]) {
            [self handleNotificationReceived:userInfo];
        }
    }
    
    return YES;
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
    NSLog(@"Application became active");
    if (self.pendingNotification && [[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"]) {
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
    sleep(15); // Consider replacing this with a proper async operation
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
    
    // Save token to NSUserDefaults
    [[NSUserDefaults standardUserDefaults] setObject:token forKey:@"device_token"];
    [[NSUserDefaults standardUserDefaults] synchronize];
    [self sendDeviceTokenToServer];
}


- (void)application:(UIApplication *)application didFailToRegisterForRemoteNotificationsWithError:(NSError *)error {
    NSLog(@"Failed to register for remote notifications: %@", error);
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
       willPresentNotification:(UNNotification *)notification
         withCompletionHandler:(void (^)(UNNotificationPresentationOptions))completionHandler {
    NSDictionary *userInfo = notification.request.content.userInfo;
    NSLog(@"Will present notification: %@", userInfo);
    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"]) {
        if (userInfo[@"aps"][@"content-available"]) {
            [self handleNotificationReceived:userInfo];
        }
        completionHandler(UNNotificationPresentationOptionAlert | UNNotificationPresentationOptionSound);
    } else {
        completionHandler(0); // No presentation options when disabled
    }
}

- (void)sendDeviceTokenToServer { //todo duplicate!!
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    NSString *deviceToken = [defaults stringForKey:@"device_token"];
    
    if (!deviceToken || deviceToken.length == 0) {
        NSLog(@"No device token found, skipping API call.");
        return;
    }
    
    // Retrieve session token from Keychain
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken || sessionToken.length == 0) {
        NSLog(@"No session token found in Keychain. Skipping API call.");
        return;
    }
    
    [FileServer performPostRequestWithURL:@"https://rors.ai/add_device"
                                       method:@"POST"
                                  contentType:@"application/json"
                                         body:@{@"device_token": deviceToken, @"session_token": sessionToken}
                            completionHandler:^(NSData *data, NSHTTPURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"Error sending device token: %@", error.localizedDescription);
            return;
        }
        if (response.statusCode == 200) {
            NSLog(@"Device token successfully sent to server");
        } else {
            NSLog(@"Failed to send device token, server responded with status code: %ld", (long)response.statusCode);
        }
    }];
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
didReceiveNotificationResponse:(UNNotificationResponse *)response
         withCompletionHandler:(void (^)(void))completionHandler {
    NSDictionary *userInfo = response.notification.request.content.userInfo;
    NSLog(@"User tapped notification: %@", userInfo);
    if ([[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"]) {
        [self handleNotificationNavigation];
    }
    completionHandler();
}

- (void)application:(UIApplication *)application
didReceiveRemoteNotification:(NSDictionary *)userInfo
fetchCompletionHandler:(void (^)(UIBackgroundFetchResult))completionHandler {
    NSLog(@"Received remote notification: %@", userInfo);
    
    if (![[NSUserDefaults standardUserDefaults] boolForKey:@"receive_notif_enabled"]) {
        completionHandler(UIBackgroundFetchResultNoData);
        return;
    }
    
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
