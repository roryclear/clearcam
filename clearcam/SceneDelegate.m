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
        NSLog(@"App launched with URL: %@", url);
        [self handleURL:url];
    }
}

- (void)scene:(UIScene *)scene openURLContexts:(NSSet<UIOpenURLContext *> *)URLContexts {
    // Handle the URL when the app is already running
    UIOpenURLContext *context = URLContexts.anyObject;
    NSURL *url = context.URL;
    NSLog(@"App opened URL: %@", url);
    [self handleURL:url];
}

- (void)handleURL:(NSURL *)url {
    // Handle security-scoped URL
    BOOL isSecurityScoped = [url startAccessingSecurityScopedResource];
    if (isSecurityScoped) {
        NSLog(@"Started accessing security-scoped resource.");
    }

    // Handle the .aes file
    [self handleAESFile:url];

    // Stop accessing security-scoped resource
    if (isSecurityScoped) {
        [url stopAccessingSecurityScopedResource];
        NSLog(@"Stopped accessing security-scoped resource.");
    }
}

#define MAGIC_NUMBER 0x4D41474943ULL // "MAGIC" in ASCII as a 64-bit value
#define HEADER_SIZE (sizeof(uint64_t)) // Size of the magic number (8 bytes)

- (NSData *)decryptData:(NSData *)encryptedData withKey:(NSString *)key {
    // Step 1: Decrypt the entire data first (we need at least the header)
    char keyPtr[kCCKeySizeAES256 + 1]; // Buffer for key
    bzero(keyPtr, sizeof(keyPtr)); // Zero out buffer
    [key getCString:keyPtr maxLength:sizeof(keyPtr) encoding:NSUTF8StringEncoding];
    
    size_t bufferSize = encryptedData.length + kCCBlockSizeAES128;
    void *buffer = malloc(bufferSize);
    
    size_t numBytesDecrypted = 0;
    CCCryptorStatus status = CCCrypt(kCCDecrypt,
                                     kCCAlgorithmAES,
                                     kCCOptionPKCS7Padding,
                                     keyPtr,
                                     kCCKeySizeAES256,
                                     NULL, // IV (NULL for no IV)
                                     encryptedData.bytes,
                                     encryptedData.length,
                                     buffer,
                                     bufferSize,
                                     &numBytesDecrypted);
    
    if (status != kCCSuccess) {
        free(buffer);
        return nil;
    }
    
    // Step 2: Check if the decrypted data is long enough to contain the header
    if (numBytesDecrypted < HEADER_SIZE) {
        free(buffer);
        return nil; // Data is too short to contain a valid header
    }
    
    // Step 3: Extract and verify the magic number from the header
    uint64_t decryptedMagicNumber;
    memcpy(&decryptedMagicNumber, buffer, HEADER_SIZE);
    if (decryptedMagicNumber != MAGIC_NUMBER) {
        free(buffer);
        return nil; // Wrong key or corrupted data
    }
    
    // Step 4: Extract the original data (skip the header)
    size_t originalDataLength = numBytesDecrypted - HEADER_SIZE;
    void *originalDataBuffer = malloc(originalDataLength);
    memcpy(originalDataBuffer, buffer + HEADER_SIZE, originalDataLength);
    
    // Step 5: Return the original data
    NSData *result = [NSData dataWithBytesNoCopy:originalDataBuffer length:originalDataLength];
    free(buffer); // Free the full decrypted buffer
    return result;
}

- (void)handleAESFile:(NSURL *)aesFileURL {
    NSError *error = nil;
    NSFileManager *fileManager = [NSFileManager defaultManager];

    // Log the file path for debugging
    NSLog(@"Attempting to access file at path: %@", aesFileURL.path);

    // Check if the file is an iCloud file
    BOOL isUbiquitous = [fileManager isUbiquitousItemAtURL:aesFileURL];
    if (isUbiquitous) {
        NSLog(@"File is an iCloud file.");

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
            // File is not downloaded locally, initiate download
            NSLog(@"File is not downloaded locally. Starting download...");
            NSError *downloadStartError = nil;
            [fileManager startDownloadingUbiquitousItemAtURL:aesFileURL error:&downloadStartError];
            if (downloadStartError) {
                NSLog(@"Failed to start downloading iCloud file: %@", downloadStartError.localizedDescription);
                [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to start download: %@", downloadStartError.localizedDescription]];
                return;
            }
            NSLog(@"Started downloading iCloud file. Monitoring download progress...");
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
        NSLog(@"Read .aes file successfully. Size: %lu bytes", (unsigned long)encryptedData.length);

        // Attempt to decrypt the data using stored keys
        NSArray<NSString *> *storedKeys = [[SecretManager sharedManager] getAllStoredKeys];
        __block NSData *decryptedData = nil; // Add __block specifier
        __block NSString *successfulKey = nil; // Add __block specifier

        // Try each stored key
        for (NSString *key in storedKeys) {
            decryptedData = [self decryptData:encryptedData withKey:key];
            if (decryptedData) {
                successfulKey = key;
                break;
            }
        }

        // If no stored key worked, prompt the user for a key
        if (!decryptedData) {
            [self promptUserForKeyWithCompletion:^(NSString *userProvidedKey) {
                if (userProvidedKey) {
                    decryptedData = [self decryptData:encryptedData withKey:userProvidedKey];
                    if (decryptedData) {
                        // Save the correct key to the secret manager
                        NSError *saveError = nil;
                        if (![[SecretManager sharedManager] saveKey:userProvidedKey error:&saveError]) {
                            NSLog(@"Failed to save key to SecretManager: %@", saveError.localizedDescription);
                        }
                        successfulKey = userProvidedKey;
                    } else {
                        NSLog(@"Failed to decrypt data with user-provided key.");
                        [self showErrorAlertWithMessage:@"Failed to decrypt the file. The key or file may be incorrect."];
                        return;
                    }
                } else {
                    NSLog(@"User canceled key entry.");
                    [self showErrorAlertWithMessage:@"Decryption canceled. A valid key is required to decrypt the file."];
                    return;
                }

                // Proceed with file handling if decryption succeeded
                [self handleDecryptedData:decryptedData fromURL:aesFileURL];
            }];
            return; // Exit the accessor block to wait for user input
        }

        // If decryption succeeded with a stored key, proceed immediately
        if (decryptedData) {
            [self handleDecryptedData:decryptedData fromURL:aesFileURL];
        }
    }];

    if (coordinationError) {
        NSLog(@"File coordination failed: %@", coordinationError.localizedDescription);
        [self showErrorAlertWithMessage:[NSString stringWithFormat:@"File access failed: %@", coordinationError.localizedDescription]];
    }
}

// Helper method to handle decrypted data
- (void)handleDecryptedData:(NSData *)decryptedData fromURL:(NSURL *)aesFileURL {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSLog(@"Decrypted data successfully. Size: %lu bytes", (unsigned long)decryptedData.length);

    // Remove only the .aes extension
    NSString *fileName = [aesFileURL lastPathComponent];
    if ([fileName hasSuffix:@".aes"]) {
        fileName = [fileName stringByReplacingOccurrencesOfString:@".aes" withString:@"" options:NSBackwardsSearch range:NSMakeRange(0, fileName.length)];
    }
    // todo: can be any file type
    NSURL *jpgFileURL = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask][0] URLByAppendingPathComponent:fileName];

    NSError *writeError = nil;
    [decryptedData writeToURL:jpgFileURL options:NSDataWritingAtomic error:&writeError];
    if (writeError) {
        NSLog(@"Failed to save .jpg file: %@", writeError.localizedDescription);
        [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to save the decrypted file: %@", writeError.localizedDescription]];
        return;
    }

    NSLog(@"Decrypted and saved .jpg file: %@", jpgFileURL);

    // Open the .jpg file in the system's photo viewer
    [self openImageInPhotoViewer:jpgFileURL];

    // Delete the .aes file after successful decryption and saving
    NSError *deleteError = nil;
    [fileManager removeItemAtURL:aesFileURL error:&deleteError];
    if (deleteError) {
        NSLog(@"Failed to delete .aes file: %@", deleteError.localizedDescription);
        [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to delete the original file: %@", deleteError.localizedDescription]];
    } else {
        NSLog(@"Successfully deleted .aes file: %@", aesFileURL.path);
    }
}

// Method to prompt the user for a key
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

    // Present the alert (assumes this is called from a view controller)
    UIViewController *topController = [UIApplication sharedApplication].keyWindow.rootViewController;
    while (topController.presentedViewController) {
        topController = topController.presentedViewController;
    }
    [topController presentViewController:alert animated:YES completion:nil];
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
        NSLog(@"Download status finished: %@", downloadStatus);

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
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Error"
                                                                  message:message
                                                           preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil];
    [alert addAction:okAction];
    [self.window.rootViewController presentViewController:alert animated:YES completion:nil];
}


#pragma mark - UIDocumentInteractionControllerDelegate

- (UIViewController *)documentInteractionControllerViewControllerForPreview:(UIDocumentInteractionController *)controller {
    // Return the root view controller to present the preview
    return self.window.rootViewController;
}

- (void)documentInteractionControllerDidEndPreview:(UIDocumentInteractionController *)controller {
    // This method is called when the user is done viewing the image
    NSLog(@"User is done viewing the image.");
    self.docController = nil; // Clear the controller
}

@end
