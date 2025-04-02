#import "SceneDelegate.h"
#import "ViewController.h" // Import your main view controller
#import <UIKit/UIKit.h>
#import <CommonCrypto/CommonCryptor.h>
#import "SecretManager.h"

@interface SceneDelegate () <UIDocumentInteractionControllerDelegate>

@property (strong, nonatomic) UIDocumentInteractionController *docController;
@property (strong, nonatomic) NSURL *pendingURL; // Store the URL if the app is launched from scratch
@property (strong, nonatomic) NSMetadataQuery *iCloudQuery; // For monitoring iCloud downloads

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
    
    // Handle URL contexts if the app is launched with a file
    if (connectionOptions.URLContexts.count > 0) {
        UIOpenURLContext *context = connectionOptions.URLContexts.anyObject;
        NSURL *url = context.URL;
        [self handleURL:url];
    }
}

- (void)scene:(UIScene *)scene openURLContexts:(NSSet<UIOpenURLContext *> *)URLContexts {
    // Handle the URL when the app is already running
    UIOpenURLContext *context = URLContexts.anyObject;
    NSURL *url = context.URL;
    [self handleURL:url];
}

- (void)handleURL:(NSURL *)url {
    // Handle security-scoped URL
    BOOL isSecurityScoped = [url startAccessingSecurityScopedResource];
    // Handle the .aes file
    [self handleAESFile:url];

    // Stop accessing security-scoped resource
    if (isSecurityScoped) [url stopAccessingSecurityScopedResource];
}

- (void)handleAESFile:(NSURL *)aesFileURL {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    BOOL isUbiquitous = [fileManager isUbiquitousItemAtURL:aesFileURL];
    if (isUbiquitous) {
        // Check the download status
        NSString *downloadStatus = nil;
        NSError *downloadError = nil;
        [aesFileURL getResourceValue:&downloadStatus forKey:NSURLUbiquitousItemDownloadingStatusKey error:&downloadError];
        if (downloadError) {
            NSLog(@"Error checking iCloud download status: %@", downloadError.localizedDescription);
            [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to check iCloud status: %@", downloadError.localizedDescription]];
            return;
        }

        NSLog(@"iCloud download status: %@", downloadStatus);

        if (![downloadStatus isEqualToString:NSURLUbiquitousItemDownloadingStatusCurrent]) {
            NSError *downloadStartError = nil;
            [fileManager startDownloadingUbiquitousItemAtURL:aesFileURL error:&downloadStartError];
            if (downloadStartError) {
                NSLog(@"Failed to start downloading iCloud file: %@", downloadStartError.localizedDescription);
                [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to start download: %@", downloadStartError.localizedDescription]];
                return;
            }
            [self monitorICloudDownloadForURL:aesFileURL];
            [self showDownloadAlert];
            return;
        }
    } else {
        // Check if the file exists locally (for non-iCloud files)
        if (![fileManager fileExistsAtPath:aesFileURL.path]) {
            NSLog(@"File does not exist at path: %@", aesFileURL.path);
            [self showErrorAlertWithMessage:@"The file does not exist or is not accessible."];
            return;
        }
    }

    // Use NSFileCoordinator to safely read the file
    NSFileCoordinator *fileCoordinator = [[NSFileCoordinator alloc] initWithFilePresenter:nil];
    NSError *coordinationError = nil;
    [fileCoordinator coordinateReadingItemAtURL:aesFileURL
                                       options:0
                                         error:&coordinationError
                                    byAccessor:^(NSURL *newURL) {
        // Read the .aes file data securely
        NSError *readError = nil;
        NSData *encryptedData = [NSData dataWithContentsOfURL:newURL options:0 error:&readError];
        if (!encryptedData) {
            NSLog(@"Failed to read .aes file: %@", readError.localizedDescription);
            [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to read the file: %@", readError.localizedDescription]];
            return;
        }
        NSArray<NSString *> *storedKeys = [[SecretManager sharedManager] getAllDecryptionKeys];
        __block NSData *decryptedData = nil; // Add __block specifier
        __block NSString *successfulKey = nil; // Add __block specifier

        for (NSString *key in storedKeys) {
            decryptedData = [[SecretManager sharedManager] decryptData:encryptedData withKey:key];
            if (decryptedData) {
                successfulKey = key;
                break;
            }
        }

        // If no stored key worked, prompt the user for a key
        if (!decryptedData) {
            [self promptUserForKeyWithAESFileURL:aesFileURL encryptedData:encryptedData];
            return; // Exit the accessor block to wait for user input
        }

        // If decryption succeeded with a stored key, proceed immediately
        if (decryptedData) {
            NSURL *url = [self handleDecryptedData:decryptedData fromURL:aesFileURL];
            if(url) [self openImageInPhotoViewer:url];
        }
    }];

    if (coordinationError) {
        NSLog(@"File coordination failed: %@", coordinationError.localizedDescription);
        [self showErrorAlertWithMessage:[NSString stringWithFormat:@"File access failed: %@", coordinationError.localizedDescription]];
    }
}

// Helper method to handle decrypted data
- (NSURL *)handleDecryptedData:(NSData *)decryptedData fromURL:(NSURL *)aesFileURL {
    // Remove only the .aes extension
    NSString *fileName = [aesFileURL lastPathComponent];
    if ([fileName hasSuffix:@".aes"]) {
        fileName = [fileName stringByReplacingOccurrencesOfString:@".aes" withString:@"" options:NSBackwardsSearch range:NSMakeRange(0, fileName.length)];
    }
    NSURL *decFileURL = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask][0] URLByAppendingPathComponent:fileName];
    NSError *writeError = nil;
    [decryptedData writeToURL:decFileURL options:NSDataWritingAtomic error:&writeError];
    if (writeError) {
        [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to save the decrypted file: %@", writeError.localizedDescription]];
        return nil;
    }
    return decFileURL;
}

- (void)promptUserForKeyWithAESFileURL:(NSURL *)aesFileURL encryptedData:(NSData *)encryptedData {
    [self promptUserForKeyWithCompletion:^(NSString *userProvidedKey) {
        if (userProvidedKey) { // User provided a key
            NSData *decryptedData = [[SecretManager sharedManager] decryptData:encryptedData withKey:userProvidedKey];
            if (decryptedData) { // Key worked
                NSError *saveError = nil;
                NSString *fileName = [[aesFileURL lastPathComponent] stringByDeletingPathExtension];
                NSString *keyPrefix = [userProvidedKey substringToIndex:MIN(6, userProvidedKey.length)];
                NSString *keyIdentifier = [NSString stringWithFormat:@"decryption_key_%@_%@", fileName, keyPrefix];
                if (![[SecretManager sharedManager] saveDecryptionKey:userProvidedKey withIdentifier:keyIdentifier error:&saveError]) {
                    NSLog(@"Failed to save key to SecretManager: %@", saveError.localizedDescription);
                }
                NSURL *url = [self handleDecryptedData:decryptedData fromURL:aesFileURL];
                if(url) [self openImageInPhotoViewer:url];
            } else { // Key didn't work, prompt again
                NSLog(@"Failed to decrypt data with user-provided key.");
                [self showErrorAlertWithMessage:@"The provided key is incorrect. Please try again or cancel." completion:^{
                    // Call the method again instead of the block
                    [self promptUserForKeyWithAESFileURL:aesFileURL encryptedData:encryptedData];
                }];
            }
        } else { // User canceled
            NSLog(@"User canceled key entry.");
            [self showErrorAlertWithMessage:@"Decryption canceled. A valid key is required to decrypt the file."];
        }
    }];
}

- (void)promptUserForKeyWithCompletion:(void (^)(NSString *))completion {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Enter Decryption Key"
                                                                  message:@"Please provide the key to decrypt the file."
                                                           preferredStyle:UIAlertControllerStyleAlert];

    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Decryption Key";
        textField.secureTextEntry = YES; // Hide input for security
    }];

    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK"
                                                       style:UIAlertActionStyleDefault
                                                     handler:^(UIAlertAction * _Nonnull action) {
        NSString *key = alert.textFields.firstObject.text;
        completion(key);
    }];

    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:^(UIAlertAction * _Nonnull action) {
        completion(nil); // User canceled
    }];

    [alert addAction:okAction];
    [alert addAction:cancelAction];

    // Present the alert using the window of the current scene
    UIWindow *window = self.window; // Use the window property of SceneDelegate
    UIViewController *topController = window.rootViewController;
    while (topController.presentedViewController) {
        topController = topController.presentedViewController;
    }
    [topController presentViewController:alert animated:YES completion:nil];
}

// Modified showErrorAlertWithMessage to support a completion handler
- (void)showErrorAlertWithMessage:(NSString *)message completion:(void (^)(void))completion {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Error"
                                                                  message:message
                                                           preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        if (completion) {
            completion();
        }
    }];
    [alert addAction:okAction];
    [self.window.rootViewController presentViewController:alert animated:YES completion:nil];
}

- (void)openImageInPhotoViewer:(NSURL *)imageURL {
    self.docController = [UIDocumentInteractionController interactionControllerWithURL:imageURL];
    self.docController.delegate = self;
    [self.docController presentPreviewAnimated:YES];
}

- (void)monitorICloudDownloadForURL:(NSURL *)aesFileURL {
    // Stop any existing query
    if (self.iCloudQuery) {
        [self.iCloudQuery stopQuery];
        [[NSNotificationCenter defaultCenter] removeObserver:self name:NSMetadataQueryDidUpdateNotification object:self.iCloudQuery];
        [[NSNotificationCenter defaultCenter] removeObserver:self name:NSMetadataQueryDidFinishGatheringNotification object:self.iCloudQuery];
    }

    // Create a new query
    self.iCloudQuery = [[NSMetadataQuery alloc] init];
    self.iCloudQuery.searchScopes = @[NSMetadataQueryUbiquitousDocumentsScope];
    self.iCloudQuery.predicate = [NSPredicate predicateWithFormat:@"%K == %@", NSMetadataItemURLKey, aesFileURL];

    // Observe query updates
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(queryDidUpdate:)
                                                 name:NSMetadataQueryDidUpdateNotification
                                               object:self.iCloudQuery];

    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(queryDidFinish:)
                                                 name:NSMetadataQueryDidFinishGatheringNotification
                                               object:self.iCloudQuery];

    [self.iCloudQuery startQuery];
}

- (void)queryDidUpdate:(NSNotification *)notification {
    NSMetadataQuery *query = notification.object;
    [query enumerateResultsUsingBlock:^(NSMetadataItem *result, NSUInteger idx, BOOL *stop) {
        NSString *downloadStatus = [result valueForAttribute:NSMetadataUbiquitousItemDownloadingStatusKey];
        NSLog(@"Download status updated: %@", downloadStatus);

        if ([downloadStatus isEqualToString:NSURLUbiquitousItemDownloadingStatusCurrent]) {
            // File is downloaded, stop the query and handle the file
            [query stopQuery];
            [[NSNotificationCenter defaultCenter] removeObserver:self name:NSMetadataQueryDidUpdateNotification object:query];
            [[NSNotificationCenter defaultCenter] removeObserver:self name:NSMetadataQueryDidFinishGatheringNotification object:query];
            self.iCloudQuery = nil;
            [self handleAESFile:[result valueForAttribute:NSMetadataItemURLKey]];
        }
    }];
}

- (void)queryDidFinish:(NSNotification *)notification {
    NSMetadataQuery *query = notification.object;
    [query enumerateResultsUsingBlock:^(NSMetadataItem *result, NSUInteger idx, BOOL *stop) {
        NSString *downloadStatus = [result valueForAttribute:NSMetadataUbiquitousItemDownloadingStatusKey];
        if ([downloadStatus isEqualToString:NSURLUbiquitousItemDownloadingStatusCurrent]) {
            // File is downloaded, stop the query and handle the file
            [query stopQuery];
            [[NSNotificationCenter defaultCenter] removeObserver:self name:NSMetadataQueryDidUpdateNotification object:query];
            [[NSNotificationCenter defaultCenter] removeObserver:self name:NSMetadataQueryDidFinishGatheringNotification object:query];
            self.iCloudQuery = nil;
            [self handleAESFile:[result valueForAttribute:NSMetadataItemURLKey]];
        }
    }];
}

- (void)showDownloadAlert {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Downloading File"
                                                                  message:@"The file is being downloaded from iCloud. Please wait a moment and try again."
                                                           preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil];
    [alert addAction:okAction];
    [self.window.rootViewController presentViewController:alert animated:YES completion:nil];
}

- (void)showErrorAlertWithMessage:(NSString *)message {
    [self showErrorAlertWithMessage:message completion:nil];
}

#pragma mark - UIDocumentInteractionControllerDelegate

- (UIViewController *)documentInteractionControllerViewControllerForPreview:(UIDocumentInteractionController *)controller {
    // Return the root view controller to present the preview
    return self.window.rootViewController;
}

- (void)documentInteractionControllerDidEndPreview:(UIDocumentInteractionController *)controller {
    self.docController = nil; // Clear the controller
}

@end
