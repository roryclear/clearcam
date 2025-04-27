#import "DeviceStreamViewController.h"
#import "StoreManager.h"
#import "SecretManager.h"
#import <AVFoundation/AVFoundation.h>
#import <Security/Security.h>
#import <CommonCrypto/CommonCryptor.h>

@interface DeviceStreamViewController ()
@property (nonatomic, strong) AVQueuePlayer *player;
@property (nonatomic, strong) AVPlayerLayer *playerLayer;
@property (nonatomic, strong) NSTimer *downloadTimer;
@property (nonatomic, strong) NSTimer *linkRefreshTimer;
@property (nonatomic, strong) NSData *lastSegmentData;
@property (nonatomic, assign) NSUInteger segmentIndex;
@property (nonatomic, strong) NSString *downloadLink;
@property (nonatomic, strong) NSString *decryptionKey;
@property (nonatomic, assign) BOOL decryptionFailedOnce;
@end

@implementation DeviceStreamViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor blackColor];

    self.player = [AVQueuePlayer queuePlayerWithItems:@[]];
    self.player.volume = 1.0;

    self.playerLayer = [AVPlayerLayer playerLayerWithPlayer:self.player];
    self.playerLayer.frame = self.view.bounds;
    self.playerLayer.videoGravity = AVLayerVideoGravityResizeAspect;
    [self.view.layer addSublayer:self.playerLayer];

    [self.player play];
    [self fetchDownloadLinkAndStartStreaming];

    NSLog(@"device streaming from = %@", self.deviceName);
}

- (void)fetchDownloadLinkAndStartStreaming {
    [self fetchDownloadLink];
    // Start timer to refresh download link every 60 seconds
    self.linkRefreshTimer = [NSTimer scheduledTimerWithTimeInterval:60.0
                                                             target:self
                                                           selector:@selector(fetchDownloadLink)
                                                           userInfo:nil
                                                            repeats:YES];
}

- (void)fetchDownloadLink {
    NSString *deviceName = self.deviceName;
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];

    if (!deviceName || !sessionToken) {
        NSLog(@"❌ Missing device name or session token");
        return;
    }

    // Retrieve decryption key from keychain
    NSString *keyIdentifier = [NSString stringWithFormat:@"decryption_key_%@", deviceName];
    NSError *keyError;
    self.decryptionKey = [[SecretManager sharedManager] retrieveDecryptionKeyWithIdentifier:keyIdentifier error:&keyError];
    if (keyError) {
        NSLog(@"⚠️ Keychain retrieval error: %@", keyError.localizedDescription);
    }

    NSString *encodedDeviceName = [deviceName stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];
    NSString *encodedSessionToken = [sessionToken stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];

    NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/get_stream_download_link"];
    components.queryItems = @[
        [NSURLQueryItem queryItemWithName:@"name" value:encodedDeviceName],
        [NSURLQueryItem queryItemWithName:@"session_token" value:encodedSessionToken]
    ];

    NSURL *url = components.URL;

    NSURLSessionDataTask *linkTask = [[NSURLSession sharedSession] dataTaskWithURL:url completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"❌ Error fetching stream link: %@", error.localizedDescription);
            return;
        }

        NSError *jsonError;
        NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError) {
            NSLog(@"⚠️ JSON Parsing Error: %@", jsonError.localizedDescription);
            return;
        }

        NSString *downloadLink = json[@"download_link"];
        if (![downloadLink isKindOfClass:[NSString class]] || downloadLink.length == 0) {
            NSLog(@"🚫 Invalid downloadLink link");
            return;
        } else {
            self.downloadLink = downloadLink;
        }

        NSLog(@"✅ Got download link: %@", self.downloadLink);

        dispatch_async(dispatch_get_main_queue(), ^{
            // Start downloading segments only if this is the first fetch
            if (!self.downloadTimer) [self startDownloadTimer];
        });
    }];
    [linkTask resume];
}

- (void)startDownloadTimer {
    self.downloadTimer = [NSTimer scheduledTimerWithTimeInterval:1.0
                                                          target:self
                                                        selector:@selector(downloadAndQueueSegment)
                                                        userInfo:nil
                                                         repeats:YES];
}

- (void)downloadAndQueueSegment {
    NSString *urlString = self.downloadLink;
    if (!urlString) {
        NSLog(@"❌ No download link available");
        return;
    }
    NSURL *remoteURL = [NSURL URLWithString:urlString];

    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithURL:remoteURL
                                                             completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error || !data) {
            NSLog(@"❌ Download error: %@", error.localizedDescription);
            return;
        }

        if ([self.lastSegmentData isEqualToData:data]) {
            NSLog(@"🔁 Same segment, skipping");
            return;
        }

        self.lastSegmentData = data;

        // Reset decryption failure flag when segment changes
        self.decryptionFailedOnce = NO;

        // Try decrypting the new segment
        [self decryptAndQueueSegment:data withCompletion:^(NSData *decryptedData) {
            if (!decryptedData) {
                NSLog(@"❌ Decryption failed");
                return;
            }

            NSString *fileName = [NSString stringWithFormat:@"segment_%lu.mp4", (unsigned long)(self.segmentIndex++ % 2)];
            NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:fileName];
            NSURL *localURL = [NSURL fileURLWithPath:tempPath];

            if ([decryptedData writeToURL:localURL atomically:YES]) {
                NSLog(@"✅ New segment saved: %@", fileName);
                dispatch_async(dispatch_get_main_queue(), ^{
                    AVURLAsset *asset = [AVURLAsset assetWithURL:localURL];
                    AVPlayerItem *item = [AVPlayerItem playerItemWithAsset:asset];
                    [self.player insertItem:item afterItem:nil];
                });
            } else {
                NSLog(@"❌ Write failed");
            }
        }];
    }];
    [task resume];
}

- (void)decryptAndQueueSegment:(NSData *)encryptedData withCompletion:(void (^)(NSData *))completion {
    if (self.decryptionKey) {
        NSData *decryptedData = [[SecretManager sharedManager] decryptData:encryptedData withKey:self.decryptionKey];
        if (decryptedData) {
            completion(decryptedData);
            self.decryptionFailedOnce = NO; // Reset on success
            return;
        } else {
            NSLog(@"⚠️ Decryption failed with stored key");
            if (self.decryptionFailedOnce) {
                // Second failure, prompt for key
                [self promptForKeyOnSecondFailure:encryptedData withCompletion:completion];
            } else {
                // First failure, mark and wait for next segment
                self.decryptionFailedOnce = YES;
                completion(nil);
            }
            return;
        }
    }

    // No key yet, mark first failure and wait for next segment
    if (!self.decryptionFailedOnce) {
        self.decryptionFailedOnce = YES;
        completion(nil);
        return;
    }

    // Second failure with no key, prompt user
    [self promptForKeyOnSecondFailure:encryptedData withCompletion:completion];
}

- (void)promptForKeyOnSecondFailure:(NSData *)encryptedData withCompletion:(void (^)(NSData *))completion {
    dispatch_async(dispatch_get_main_queue(), ^{
        [self promptUserForKeyWithCompletion:^(NSString *userProvidedKey) {
            if (userProvidedKey) {
                NSData *decryptedData = [[SecretManager sharedManager] decryptData:encryptedData withKey:userProvidedKey];
                if (decryptedData) {
                    // Save the valid key to keychain
                    NSString *keyIdentifier = [NSString stringWithFormat:@"decryption_key_%@", self.deviceName];
                    NSError *saveError;
                    [[SecretManager sharedManager] saveDecryptionKey:userProvidedKey withIdentifier:keyIdentifier error:&saveError];
                    if (saveError) {
                        NSLog(@"⚠️ Failed to save key to keychain: %@", saveError.localizedDescription);
                    }
                    self.decryptionKey = userProvidedKey;
                    self.decryptionFailedOnce = NO;
                    completion(decryptedData);
                } else {
                    [self showErrorAlertWithMessage:@"The provided key is incorrect. Please try again or cancel." completion:^{
                        [self decryptAndQueueSegment:encryptedData withCompletion:completion];
                    }];
                }
            } else {
                completion(nil); // User cancelled
            }
        }];
    });
}

- (void)promptUserForKeyWithCompletion:(void (^)(NSString *))completion {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Enter Decryption Key"
                                                                  message:@"Please enter the password used by your device to encrypt this data."
                                                           preferredStyle:UIAlertControllerStyleAlert];

    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Decryption Key";
        textField.secureTextEntry = YES;
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
        completion(nil);
    }];

    [alert addAction:okAction];
    [alert addAction:cancelAction];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)showErrorAlertWithMessage:(NSString *)message completion:(void (^)(void))completion {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Error"
                                                                  message:message
                                                           preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK"
                                                       style:UIAlertActionStyleDefault
                                                     handler:^(UIAlertAction * _Nonnull action) {
        if (completion) completion();
    }];
    [alert addAction:okAction];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    self.playerLayer.frame = self.view.bounds;
}

- (void)dealloc {
    [self.downloadTimer invalidate];
    [self.linkRefreshTimer invalidate];
}

@end
