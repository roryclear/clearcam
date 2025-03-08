#import "SceneDelegate.h"
#import "ViewController.h" // Import your main view controller
#import "PGP.h" // Import your PGP class
#import <UIKit/UIKit.h>

@interface SceneDelegate () <UIDocumentInteractionControllerDelegate>

@property (strong, nonatomic) UIDocumentInteractionController *docController;
@property (strong, nonatomic) NSURL *pendingURL; // Store the URL if the app is launched from scratch
@property (strong, nonatomic) PGP *pgp; // Instance of your PGP class

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
    
    // Initialize the PGP instance
    self.pgp = [[PGP alloc] init];
    if (!self.pgp) {
        NSLog(@"Failed to initialize PGP.");
    }
    
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
    if (!url) {
        NSLog(@"Error: URL is nil.");
        return;
    }

    // Start accessing security-scoped resource (for iCloud/Downloads files)
    BOOL isSecured = [url startAccessingSecurityScopedResource];
    if (!isSecured) {
        NSLog(@"Error: Unable to access security-scoped resource.");
        return;
    }

    // Get the path to the app's Documents directory
    NSURL *documentsDirectoryURL = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];

    // Destination path inside the app's Documents directory
    NSURL *localPGPFileURL = [documentsDirectoryURL URLByAppendingPathComponent:url.lastPathComponent];

    // Check if file is already in Documents, otherwise copy it
    if (![url.path isEqualToString:localPGPFileURL.path]) {
        NSError *copyError = nil;
        [[NSFileManager defaultManager] copyItemAtURL:url toURL:localPGPFileURL error:&copyError];
        
        if (copyError) {
            NSLog(@"Error copying file: %@", copyError.localizedDescription);
            [url stopAccessingSecurityScopedResource]; // Stop access before returning
            return;
        }
    }

    // Stop accessing the security-scoped resource after copying
    [url stopAccessingSecurityScopedResource];

    // Decrypt the copied .pgp file using your PGP class
    NSString *encryptedFilePath = localPGPFileURL.path;
    [self.pgp decryptImageWithPrivateKey:encryptedFilePath];

    // Get the path to the decrypted .jpg file
    NSString *decryptedFilePath = [[encryptedFilePath stringByDeletingPathExtension] stringByAppendingPathExtension:@"jpg"];
    NSURL *decryptedFileURL = [NSURL fileURLWithPath:decryptedFilePath];

    // Check if the decrypted file exists
    if ([[NSFileManager defaultManager] fileExistsAtPath:decryptedFilePath]) {
        // Create a UIDocumentInteractionController for the decrypted file
        self.docController = [UIDocumentInteractionController interactionControllerWithURL:decryptedFileURL];

        // Set the delegate to self
        self.docController.delegate = self;

        // Present the document interaction controller from the root view controller
        [self.docController presentPreviewAnimated:YES];
    } else {
        NSLog(@"Decrypted file not found at %@", decryptedFilePath);
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
