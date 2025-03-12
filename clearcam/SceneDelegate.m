#import "SceneDelegate.h"
#import "ViewController.h" // Import your main view controller
#import <UIKit/UIKit.h>

@interface SceneDelegate () <UIDocumentInteractionControllerDelegate>

@property (strong, nonatomic) UIDocumentInteractionController *docController;
@property (strong, nonatomic) NSURL *pendingURL; // Store the URL if the app is launched from scratch

@end

@implementation SceneDelegate

- (void)scene:(UIScene *)scene willConnectToSession:(UISceneSession *)session options:(UISceneConnectionOptions *)connectionOptions {
    // Set up the window and root view controller
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene *)scene];
    
    // Set the root view controller to your main ViewController
    ViewController *rootViewController = [[ViewController alloc] init];
    UINavigationController *navigationController = [[UINavigationController alloc] initWithRootViewController:rootViewController];
    self.window.rootViewController = navigationController;
    [self.window makeKeyAndVisible];
    
    if (connectionOptions.URLContexts.count > 0) {
        NSURL *url = connectionOptions.URLContexts.anyObject.URL;
    }
}

- (void)scene:(UIScene *)scene openURLContexts:(NSSet<UIOpenURLContext *> *)URLContexts {
    // Handle the URL when the app is already running
    NSURL *url = URLContexts.anyObject.URL;
}

#pragma mark - UIDocumentInteractionControllerDelegate

- (UIViewController *)documentInteractionControllerViewControllerForPreview:(UIDocumentInteractionController *)controller {
    // Return the root view controller to present the preview
    return self.window.rootViewController;
}

- (void)documentInteractionControllerDidEndPreview:(UIDocumentInteractionController *)controller {
    // This method is called when the user is done viewing the image
    // You can add any cleanup or navigation logic here if needed
    NSLog(@"User is done viewing the image.");
}

@end
