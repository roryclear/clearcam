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
@property (nonatomic, strong) NSData *lastFailedEncryptedData;
@property (nonatomic, assign) NSInteger consecutiveDecryptionFailures;
@property (nonatomic, assign) BOOL decryptionFailedOnce;
@property (nonatomic, strong) UIActivityIndicatorView *loadingSpinner;
@property (nonatomic, assign) NSInteger successfullyQueuedSegmentCount;
@property (nonatomic, strong) NSMutableSet<NSData *> *seenSegmentHashes;
@end

@implementation DeviceStreamViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor blackColor];

    // Setup spinner
    self.loadingSpinner = [[UIActivityIndicatorView alloc] initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleLarge];
    self.loadingSpinner.center = self.view.center;
    self.loadingSpinner.color = [UIColor whiteColor];
    [self.view addSubview:self.loadingSpinner];
    [self.loadingSpinner startAnimating];

    self.player = [AVQueuePlayer queuePlayerWithItems:@[]];
    self.playerLayer = [AVPlayerLayer playerLayerWithPlayer:self.player];
    self.playerLayer.frame = self.view.bounds;
    self.playerLayer.videoGravity = AVLayerVideoGravityResizeAspect;
    [self.view.layer addSublayer:self.playerLayer];

    [self.player play];
    [self fetchDownloadLinkAndStartStreaming];
    
    self.successfullyQueuedSegmentCount = 0;
    self.seenSegmentHashes = [NSMutableSet set];
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

        // Only set lastSegmentData after successful decryption
        NSLog(@"📥 New segment downloaded, attempting decryption");

        [self decryptAndQueueSegment:data withCompletion:^(NSData *decryptedData) {
            if (!decryptedData) {
                NSLog(@"❌ Decryption failed");
                return;
            }

            // Success: now save the segment as "lastSegmentData"
            NSUInteger hash = data.hash;
            NSData *hashData = [NSData dataWithBytes:&hash length:sizeof(hash)];
            if ([self.seenSegmentHashes containsObject:hashData]) {
                NSLog(@"🔁 Duplicate decrypted segment, skipping queue");
                return;
            }
            [self.seenSegmentHashes addObject:hashData];
            self.successfullyQueuedSegmentCount += 1;
            self.lastSegmentData = data;
            self.decryptionFailedOnce = NO; // Only reset on success

            NSString *fileName = [NSString stringWithFormat:@"segment_%lu.mp4", (unsigned long)(self.segmentIndex++ % 2)];
            NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:fileName];
            NSURL *localURL = [NSURL fileURLWithPath:tempPath];

            // Delete existing file if any (from old streams)
            [[NSFileManager defaultManager] removeItemAtURL:localURL error:nil];

            if ([decryptedData writeToURL:localURL atomically:YES]) {
                NSLog(@"✅ New segment saved: %@", fileName);
                
                //thumbnail save
                AVAsset *videoAsset = [AVAsset assetWithURL:localURL];
                AVAssetImageGenerator *imageGenerator = [[AVAssetImageGenerator alloc] initWithAsset:videoAsset];
                imageGenerator.appliesPreferredTrackTransform = YES; // To avoid rotated images

                CMTime time = CMTimeMakeWithSeconds(0.0, 600); // Grab first frame (0s)
                [imageGenerator generateCGImagesAsynchronouslyForTimes:@[[NSValue valueWithCMTime:time]]
                                                     completionHandler:^(CMTime requestedTime, CGImageRef cgImage, CMTime actualTime, AVAssetImageGeneratorResult result, NSError *error) {
                    if (result == AVAssetImageGeneratorSucceeded && cgImage) {
                        UIImage *thumbnail = [UIImage imageWithCGImage:cgImage];
                        NSData *imageData = UIImageJPEGRepresentation(thumbnail, 0.7); // Compress a bit

                        NSString *filename = [NSString stringWithFormat:@"thumbnail_%@.jpg", self.deviceName];
                        NSString *documentsPath = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
                        NSString *filePath = [documentsPath stringByAppendingPathComponent:filename];
                        if ([imageData writeToFile:filePath atomically:YES]) {
                            NSLog(@"🖼️ Thumbnail saved: %@", filename);
                        } else {
                            NSLog(@"❌ Failed to save thumbnail");
                        }
                    } else {
                        NSLog(@"❌ Failed to generate thumbnail: %@", error.localizedDescription);
                    }
                }];
                
                
                dispatch_async(dispatch_get_main_queue(), ^{
                    if (self.successfullyQueuedSegmentCount >= 2) {
                        [self.loadingSpinner stopAnimating];
                        self.loadingSpinner.hidden = YES;
                    }

                    if (self.successfullyQueuedSegmentCount > 1) {
                        // Start queuing segments after the first one
                        AVURLAsset *asset = [AVURLAsset assetWithURL:localURL];
                        AVPlayerItem *item = [AVPlayerItem playerItemWithAsset:asset];
                        [self.player insertItem:item afterItem:nil];
                    } else {
                        NSLog(@"⏳ Skipping first segment from playback");
                    }
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
            self.consecutiveDecryptionFailures = 0; // ✅ Reset on success
            self.lastFailedEncryptedData = nil;
            return;
        } else {
            NSLog(@"⚠️ Decryption failed with stored key");
        }
    }

    // Handle failure
    BOOL isSameData = [self.lastFailedEncryptedData isEqualToData:encryptedData];
    if (isSameData) {
        NSLog(@"🔁 Same failed segment, not incrementing failure count");
        completion(nil);
        return;
    }

    self.consecutiveDecryptionFailures += 1;
    self.lastFailedEncryptedData = encryptedData;

    NSLog(@"❌ Decryption failure count (different segments): %ld", (long)self.consecutiveDecryptionFailures);

    if (self.consecutiveDecryptionFailures >= 2) {
        NSLog(@"🔑 Prompting for key after two different segment failures");
        [self promptForKeyOnSecondFailure:encryptedData withCompletion:completion];
        self.consecutiveDecryptionFailures = 0;
        self.lastFailedEncryptedData = nil;
    } else {
        NSLog(@"⏳ Waiting for another different failed segment...");
        completion(nil);
    }
}


- (void)promptForKeyOnSecondFailure:(NSData *)encryptedData withCompletion:(void (^)(NSData *))completion {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[SecretManager sharedManager] promptUserForKeyFromViewController:self completion:^(NSString *userProvidedKey) {
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

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    
    [self.downloadTimer invalidate];
    self.downloadTimer = nil;
    [self.linkRefreshTimer invalidate];
    self.linkRefreshTimer = nil;
    
    [self sendDeleteStreamDownloadLink];
}

- (void)sendDeleteStreamDownloadLink {
    NSString *deviceName = self.deviceName;
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];

    if (!deviceName || !sessionToken) {
        NSLog(@"❌ Missing device name or session token for delete request");
        return;
    }

    NSString *encodedDeviceName = [deviceName stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];
    NSString *encodedSessionToken = [sessionToken stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];

    NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/delete_stream_download_link"];
    components.queryItems = @[
        [NSURLQueryItem queryItemWithName:@"name" value:encodedDeviceName],
        [NSURLQueryItem queryItemWithName:@"session_token" value:encodedSessionToken]
    ];

    NSURL *url = components.URL;

    NSURLSessionDataTask *deleteTask = [[NSURLSession sharedSession] dataTaskWithURL:url completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"❌ Error sending delete request: %@", error.localizedDescription);
            return;
        }
        
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
        if (httpResponse.statusCode == 200) {
            NSLog(@"✅ Successfully deleted stream download link");
        } else {
            NSLog(@"⚠️ Delete request failed with status code: %ld", (long)httpResponse.statusCode);
        }
    }];
    [deleteTask resume];
}

@end
