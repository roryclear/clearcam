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
    
    // Check if the app was launched with a URL (e.g., opening a .pgp file)
    if (connectionOptions.URLContexts.count > 0) {
        NSURL *url = connectionOptions.URLContexts.anyObject.URL;
        if ([url.pathExtension isEqualToString:@"pgp"]) {
            // Store the URL for later use
            self.pendingURL = url;
            
            // Handle the pending URL after a slight delay to ensure the window is set up
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                [self handlePGPFileAtURL:self.pendingURL];
                self.pendingURL = nil; // Clear the pending URL
            });
        }
    }
}

- (void)scene:(UIScene *)scene openURLContexts:(NSSet<UIOpenURLContext *> *)URLContexts {
    // Handle the URL when the app is already running
    NSURL *url = URLContexts.anyObject.URL;
    if ([url.pathExtension isEqualToString:@"pgp"]) {
        [self handlePGPFileAtURL:url];
    }
}

- (void)handlePGPFileAtURL:(NSURL *)url {
    // Get the path to the app's documents directory
    NSURL *documentsDirectoryURL = [[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask].firstObject;
    
    // Create a URL for the "image.jpg" file in the documents directory
    NSURL *imageURL = [documentsDirectoryURL URLByAppendingPathComponent:@"image.jpg"];
    
    // Check if the file exists at the path
    if ([[NSFileManager defaultManager] fileExistsAtPath:imageURL.path]) {
        // Create a UIDocumentInteractionController for the image file
        self.docController = [UIDocumentInteractionController interactionControllerWithURL:imageURL];
        
        // Set the delegate to self
        self.docController.delegate = self;
        
        // Present the document interaction controller from the root view controller
        [self.docController presentPreviewAnimated:YES];
    } else {
        // If the file doesn't exist, log an error or show an alert
        NSLog(@"File 'image.jpg' not found in documents folder.");
    }
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
