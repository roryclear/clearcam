#import "SceneDelegate.h"
#import "MainViewController.h"
#import "LoginViewController.h"
#import "StoreManager.h"

@interface SceneDelegate ()
@end

@implementation SceneDelegate

- (void)scene:(UIScene *)scene
willConnectToSession:(UISceneSession *)session
      options:(UISceneConnectionOptions *)connectionOptions {
    [[StoreManager sharedInstance] clearSessionTokenFromKeychain]; // todo, for testing
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene *)scene];
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    UIViewController *rootVC;
    
    if (sessionToken && sessionToken.length > 0) {
        rootVC = [[MainViewController alloc] init];
    } else {
        rootVC = [[LoginViewController alloc] init];
    }
    
    UINavigationController *navigationController = [[UINavigationController alloc] initWithRootViewController:rootVC];
    self.window.rootViewController = navigationController;
    [self.window makeKeyAndVisible];
}

@end

