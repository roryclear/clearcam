#import "SceneDelegate.h"
#import <UIKit/UIKit.h>

@interface SceneDelegate () <UIDocumentInteractionControllerDelegate>

@property (strong, nonatomic) UIDocumentInteractionController *docController;

@end

@implementation SceneDelegate

// This method will be called when your app is opened with a URL (e.g., a .pgp file)
- (void)scene:(UIScene *)scene openURLContexts:(NSSet<UIOpenURLContext *> *)URLContexts {
    // Get the URL of the opened file
    NSURL *url = URLContexts.anyObject.URL;
    
    // Check if the file has a .pgp extension
    if ([url.pathExtension isEqualToString:@"pgp"]) {
        // Handle the .pgp file (e.g., decrypt or process it)
        [self handlePGPFileAtURL:url];
    }
}

// Custom method to handle the .pgp file and open "image.jpg" from the documents folder
- (void)handlePGPFileAtURL:(NSURL *)url {
    // Get the path to the app's documents directory
    NSURL *documentsDirectoryURL = [[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask].firstObject;
    
    // Create a URL for the "image.jpg" file in the documents directory
    NSURL *imageURL = [documentsDirectoryURL URLByAppendingPathComponent:@"image.jpg"];
    
    // Check if the file exists at the path
    if ([[NSFileManager defaultManager] fileExistsAtPath:imageURL.path]) {
        // Get the root view controller of the window to present the document controller
        UIWindow *window = [UIApplication sharedApplication].keyWindow;
        UIViewController *rootViewController = window.rootViewController;
        
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

// Implementing the required delegate method
- (UIViewController *)documentInteractionControllerViewControllerForPreview:(UIDocumentInteractionController *)controller {
    // Return the view controller that will present the preview
    UIWindow *window = [UIApplication sharedApplication].keyWindow;
    return window.rootViewController;
}

@end
