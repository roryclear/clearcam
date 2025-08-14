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
    
    // Check authentication status properly
    [self verifyAuthentication];
}

- (void)verifyAuthentication {
    [[StoreManager sharedInstance] verifySubscriptionWithCompletionIfSubbed:^(BOOL isActive, NSDate *expiryDate) {
        dispatch_async(dispatch_get_main_queue(), ^{
            UIViewController *rootVC;
            
            if (isActive) {
                rootVC = [[GalleryViewController alloc] init];
            } else {
                rootVC = [[LoginViewController alloc] init];
                // Clear any invalid token
                [[StoreManager sharedInstance] clearSessionTokenFromKeychain];
            }
            
            UINavigationController *navigationController = [[UINavigationController alloc] initWithRootViewController:rootVC];
            
            // Animate the transition
            [UIView transitionWithView:self.window
                              duration:0.3
                               options:UIViewAnimationOptionTransitionCrossDissolve
                            animations:^{
                self.window.rootViewController = navigationController;
            } completion:nil];
            
            [self.activityIndicator stopAnimating];
        });
    }];
}

@end
