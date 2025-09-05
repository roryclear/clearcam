#import "SceneDelegate.h"
#import "GalleryViewController.h"
#import "LoginViewController.h"
#import "StoreManager.h"

@interface SceneDelegate ()
@property (nonatomic, strong) UIActivityIndicatorView *activityIndicator;
@end

@implementation SceneDelegate

- (void)scene:(UIScene *)scene
willConnectToSession:(UISceneSession *)session
      options:(UISceneConnectionOptions *)connectionOptions {
    //[[StoreManager sharedInstance] clearSessionTokenFromKeychain]; // todo for testing
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene *)scene];
    
    // Create temporary loading view
    UIViewController *loadingVC = [UIViewController new];
    loadingVC.view.backgroundColor = [UIColor systemBackgroundColor];
    self.activityIndicator = [[UIActivityIndicatorView alloc] initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleLarge];
    self.activityIndicator.center = loadingVC.view.center;
    [loadingVC.view addSubview:self.activityIndicator];
    [self.activityIndicator startAnimating];
    
    self.window.rootViewController = loadingVC;
    [self.window makeKeyAndVisible];
    [self verifyAuthentication];
}

- (void)verifyAuthentication {
    [[StoreManager sharedInstance] checkInternetWithCompletion:^(BOOL hasInternet) {
        if (!hasInternet) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self switchToRootViewController:[[GalleryViewController alloc] init]];
            });
            return;
        }
        [self attemptVerifySubscriptionWithRetry:2];
    }];
}

- (void)attemptVerifySubscriptionWithRetry:(NSInteger)remainingAttempts {
    [[StoreManager sharedInstance] verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (isActive) {
                [self switchToRootViewController:[[GalleryViewController alloc] init]];
            } else if (remainingAttempts > 0) {
                [self attemptVerifySubscriptionWithRetry:remainingAttempts - 1];
            } else {
                [self switchToRootViewController:[[LoginViewController alloc] init]];
            }
        });
    }];
}


- (void)switchToRootViewController:(UIViewController *)rootVC {
    UINavigationController *navigationController = [[UINavigationController alloc] initWithRootViewController:rootVC];
    [UIView transitionWithView:self.window
                      duration:0.3
                       options:UIViewAnimationOptionTransitionCrossDissolve
                    animations:^{
        self.window.rootViewController = navigationController;
    } completion:nil];
    
    [self.activityIndicator stopAnimating];
}

@end

