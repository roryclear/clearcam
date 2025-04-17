#import "GalleryViewController.h"
#import "StoreManager.h"
#import "SecretManager.h"
#import <AVKit/AVKit.h>
#import <AVFoundation/AVFoundation.h>

@interface VideoTableViewCell : UITableViewCell
@property (nonatomic, strong) UIImageView *thumbnailView;
@property (nonatomic, strong) UILabel *titleLabel;
@property (nonatomic, strong) UIButton *menuButton;
@end

@implementation VideoTableViewCell

- (instancetype)initWithStyle:(UITableViewCellStyle)style reuseIdentifier:(NSString *)reuseIdentifier {
    self = [super initWithStyle:style reuseIdentifier:reuseIdentifier];
    if (self) {
        [self setupUI];
    }
    return self;
}

- (void)setupUI {
    self.thumbnailView = [[UIImageView alloc] init];
    self.thumbnailView.contentMode = UIViewContentModeScaleAspectFill;
    self.thumbnailView.clipsToBounds = YES;
    self.thumbnailView.layer.cornerRadius = 4;
    [self.contentView addSubview:self.thumbnailView];
    
    self.titleLabel = [[UILabel alloc] init];
    self.titleLabel.numberOfLines = 1;
    self.titleLabel.font = [UIFont systemFontOfSize:16 weight:UIFontWeightMedium];
    [self.contentView addSubview:self.titleLabel];
    
    self.menuButton = [UIButton buttonWithType:UIButtonTypeSystem];
    UIImage *ellipsisImage = [UIImage systemImageNamed:@"ellipsis"];
    UIImageSymbolConfiguration *config = [UIImageSymbolConfiguration configurationWithScale:UIImageSymbolScaleLarge];
    UIImage *verticalEllipsis = [ellipsisImage imageByApplyingSymbolConfiguration:config];
    [self.menuButton setImage:verticalEllipsis forState:UIControlStateNormal];
    self.menuButton.tintColor = [UIColor systemGrayColor];
    self.menuButton.transform = CGAffineTransformMakeRotation(M_PI_2);
    [self.contentView addSubview:self.menuButton];
    
    self.thumbnailView.translatesAutoresizingMaskIntoConstraints = NO;
    self.titleLabel.translatesAutoresizingMaskIntoConstraints = NO;
    self.menuButton.translatesAutoresizingMaskIntoConstraints = NO;
    
    [NSLayoutConstraint activateConstraints:@[
        [self.thumbnailView.leadingAnchor constraintEqualToAnchor:self.contentView.leadingAnchor constant:16],
        [self.thumbnailView.centerYAnchor constraintEqualToAnchor:self.contentView.centerYAnchor],
        [self.thumbnailView.widthAnchor constraintEqualToConstant:120],
        [self.thumbnailView.heightAnchor constraintEqualToConstant:80],
        
        [self.titleLabel.leadingAnchor constraintEqualToAnchor:self.thumbnailView.trailingAnchor constant:12],
        [self.titleLabel.trailingAnchor constraintEqualToAnchor:self.menuButton.leadingAnchor constant:-12],
        [self.titleLabel.centerYAnchor constraintEqualToAnchor:self.contentView.centerYAnchor],
        
        [self.menuButton.trailingAnchor constraintEqualToAnchor:self.contentView.trailingAnchor constant:-16],
        [self.menuButton.centerYAnchor constraintEqualToAnchor:self.contentView.centerYAnchor],
        [self.menuButton.widthAnchor constraintEqualToConstant:44],
        [self.menuButton.heightAnchor constraintEqualToConstant:44]
    ]];
}

@end

@interface GalleryViewController () <UITableViewDelegate, UITableViewDataSource>
@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSMutableArray<NSString *> *videoFiles;
@property (nonatomic, strong) NSString *downloadDirectory;
@property (nonatomic, strong) NSURLSession *downloadSession;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSMutableArray<NSString *> *> *groupedVideos;
@property (nonatomic, strong) NSMutableArray<NSString *> *sectionTitles;
@property (nonatomic, strong) UIRefreshControl *refreshControl;
@property (nonatomic, assign) BOOL isLoadingVideos;
@property (nonatomic, strong) NSMutableSet<NSString *> *loadedFilenames;
@end

@implementation GalleryViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Events";
    self.navigationController.navigationBarHidden = NO;
    
    self.videoFiles = [NSMutableArray array];
    self.groupedVideos = [NSMutableDictionary dictionary];
    self.sectionTitles = [NSMutableArray array];
    self.isLoadingVideos = NO;
    
    NSURLSessionConfiguration *config = [NSURLSessionConfiguration defaultSessionConfiguration];
    config.timeoutIntervalForRequest = 60.0;
    config.timeoutIntervalForResource = 600.0;
    self.downloadSession = [NSURLSession sessionWithConfiguration:config];
    self.loadedFilenames = [NSMutableSet set];
    [self setupDownloadDirectory];
    [self setupTableView];
    [self setupRefreshControl];
    [self loadExistingVideos];
    [self getEvents];
}

- (void)setupRefreshControl {
    self.refreshControl = [[UIRefreshControl alloc] init];
    [self.refreshControl addTarget:self action:@selector(handleRefresh) forControlEvents:UIControlEventValueChanged];
    if (@available(iOS 10.0, *)) {
        self.tableView.refreshControl = self.refreshControl;
    } else {
        [self.tableView addSubview:self.refreshControl];
    }
}

- (void)handleRefresh {
    if (!self.isLoadingVideos) {
        [self getEvents];
    } else {
        [self.refreshControl endRefreshing];
    }
}

- (void)setIsLoadingVideos:(BOOL)isLoadingVideos {
    _isLoadingVideos = isLoadingVideos;
    
    if (!isLoadingVideos && self.refreshControl.isRefreshing) {
        [self.refreshControl endRefreshing];
    }
}

- (void)setupDownloadDirectory {
    NSString *documentsDir = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    self.downloadDirectory = [documentsDir stringByAppendingPathComponent:@"downloaded-events"];
    
    if (![[NSFileManager defaultManager] fileExistsAtPath:self.downloadDirectory]) {
        NSError *error;
        [[NSFileManager defaultManager] createDirectoryAtPath:self.downloadDirectory
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:&error];
    }
}

- (void)setupTableView {
    self.tableView = [[UITableView alloc] initWithFrame:CGRectZero style:UITableViewStylePlain];
    self.tableView.delegate = self;
    self.tableView.dataSource = self;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor];
    [self.tableView registerClass:[VideoTableViewCell class] forCellReuseIdentifier:@"VideoCell"];
    self.tableView.separatorInset = UIEdgeInsetsMake(0, 16, 0, 16);

    if (@available(iOS 11.0, *)) {
        self.tableView.contentInsetAdjustmentBehavior = UIScrollViewContentInsetAdjustmentNever;
    }
    
    [self.view addSubview:self.tableView];
    
    self.tableView.translatesAutoresizingMaskIntoConstraints = NO;

    [NSLayoutConstraint activateConstraints:@[
        [self.tableView.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor constant:0],
        [self.tableView.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor constant:0],
        [self.tableView.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor constant:0],
        [self.tableView.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor constant:0]
    ]];
}

- (NSDate *)latestDownloadedFileDate {
    NSArray *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:self.downloadDirectory error:nil];
    NSDate *latestDate = nil;

    for (NSString *file in files) {
        NSString *filePath = [self.downloadDirectory stringByAppendingPathComponent:file];
        NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:filePath error:nil];
        NSDate *creationDate = attributes[NSFileCreationDate];
        if (creationDate && (!latestDate || [creationDate compare:latestDate] == NSOrderedDescending)) {
            latestDate = creationDate;
        }
    }
    return latestDate;
}

- (void)getEvents {
    if (self.isLoadingVideos) return;
    
    self.isLoadingVideos = YES;
        
    NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/events"];
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    
    NSMutableArray<NSURLQueryItem *> *queryItems = [NSMutableArray array];
    
    if (sessionToken) [queryItems addObject:[NSURLQueryItem queryItemWithName:@"session_token" value:sessionToken]];
    
    NSDate *latestDate = [self latestDownloadedFileDate];
    if (latestDate) {
        NSTimeInterval timestamp = [latestDate timeIntervalSince1970];
        NSString *timestampString = [NSString stringWithFormat:@"%.0f", timestamp];
        [queryItems addObject:[NSURLQueryItem queryItemWithName:@"newest_creation_time" value:timestampString]];
    }

    components.queryItems = queryItems;
    NSURL *url = components.URL;
    
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    [request setHTTPMethod:@"GET"];
    
    [[self.downloadSession dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            self.isLoadingVideos = NO;
        });
        
        if (error) return;
        
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
        if (httpResponse.statusCode == 200) {
            NSError *jsonError;
            NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
            
            if (!jsonError && json[@"files"]) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self downloadFiles:json[@"files"]];
                });
            }
        }
    }] resume];
}

- (void)processVideoFileAtPath:(NSString *)filePath withFilename:(NSString *)filename {
    NSString *extension = filePath.pathExtension.lowercaseString;
    
    // Only process mp4 and aes files
    if (!([extension isEqualToString:@"mp4"] || [extension isEqualToString:@"aes"])) return;
    
    if ([self.loadedFilenames containsObject:filename]) {
        return; // Skip if already processed
    }
    
    if ([extension isEqualToString:@"aes"]) {
        // Handle AES decryption
        NSError *decryptionError = nil;
        NSData *encryptedData = [NSData dataWithContentsOfURL:[NSURL fileURLWithPath:filePath] options:0 error:&decryptionError];
        
        if (encryptedData) {
            NSArray<NSString *> *storedKeys = [[SecretManager sharedManager] getAllDecryptionKeys];
            NSData *decryptedData = nil;
            
            // Try all available keys
            for (NSString *key in storedKeys) {
                decryptedData = [[SecretManager sharedManager] decryptData:encryptedData withKey:key];
                if (decryptedData) break;
            }
            
            if (decryptedData) {
                // Handle the decrypted data
                NSURL *decryptedURL = [self handleDecryptedData:decryptedData fromURL:[NSURL fileURLWithPath:filePath]];
                if (decryptedURL) {
                    filePath = decryptedURL.path;
                    filename = [filePath lastPathComponent];
                } else {
                    [self addVideoFileAtPath:filePath];
                    return;
                }
            } else {
                [self addVideoFileAtPath:filePath];
                return;
            }
        } else {
            return;
        }
    }
    
    [self addVideoFileAtPath:filePath];
}

- (void)downloadFiles:(NSArray<NSString *> *)fileURLs {
    if (fileURLs.count == 0) {
        [self loadExistingVideos];
        return;
    }
    
    self.isLoadingVideos = YES;
    dispatch_queue_t serialQueue = dispatch_queue_create("com.yourapp.serialDownloadQueue", DISPATCH_QUEUE_SERIAL);
    
    dispatch_async(serialQueue, ^{
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        
        for (NSString *fileURL in fileURLs) {
            NSURL *url = [NSURL URLWithString:fileURL];
            NSString *saveFileName = [url lastPathComponent];
            NSString *destPath = [self.downloadDirectory stringByAppendingPathComponent:saveFileName];
            
            NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
            [request setHTTPMethod:@"GET"];
            
            [[self.downloadSession downloadTaskWithRequest:request completionHandler:^(NSURL *location, NSURLResponse *response, NSError *error) {
                if (!error && [(NSHTTPURLResponse *)response statusCode] == 200) {
                    NSError *moveError;
                    NSURL *destURL = [NSURL fileURLWithPath:destPath];
                    
                    // Move the downloaded file to destination
                    [[NSFileManager defaultManager] moveItemAtURL:location toURL:destURL error:&moveError];
                    
                    // Process the file on main thread
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [self processVideoFileAtPath:destPath withFilename:saveFileName];
                    });
                }
                dispatch_semaphore_signal(semaphore);
            }] resume];
            
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        }
        
        // All downloads completed
        dispatch_async(dispatch_get_main_queue(), ^{
            self.isLoadingVideos = NO;
            [self loadExistingVideos];
        });
    });
}

- (void)loadExistingVideos {
    NSError *error;
    NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:self.downloadDirectory error:&error];

    if (!self.loadedFilenames) {
        self.loadedFilenames = [NSMutableSet set];
    }

    for (NSString *file in contents) {
        NSString *filePath = [self.downloadDirectory stringByAppendingPathComponent:file];
        [self processVideoFileAtPath:filePath withFilename:file];
    }
}

- (void)addVideoFileAtPath:(NSString *)filePath {
    if (!filePath) return;
    NSString *filename = [filePath lastPathComponent];
    if ([self.loadedFilenames containsObject:filename]) return;
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:filePath error:nil];
    NSDate *creationDate = [attributes fileCreationDate] ?: [NSDate distantPast];
    [self.videoFiles addObject:filePath];
    [self.loadedFilenames addObject:filename];
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    dateFormatter.dateStyle = NSDateFormatterMediumStyle;
    dateFormatter.timeStyle = NSDateFormatterNoStyle;
    NSDateFormatter *inputFormatter = [[NSDateFormatter alloc] init];
    [inputFormatter setDateFormat:@"yyyy-MM-dd"];
    NSString *datePart = [[filename componentsSeparatedByString:@"_"] firstObject];
    NSDate *date = [inputFormatter dateFromString:datePart];
    NSString *sectionTitle = [dateFormatter stringFromDate:date] ?: @"Unknown Date";
    if (!self.groupedVideos[sectionTitle]) {
        self.groupedVideos[sectionTitle] = [NSMutableArray array];
        [self.sectionTitles addObject:sectionTitle];
    }
    [self.groupedVideos[sectionTitle] addObject:filePath];
    [self.sectionTitles sortUsingComparator:^NSComparisonResult(NSString *obj1, NSString *obj2) {
        NSDate *date1 = [dateFormatter dateFromString:obj1];
        NSDate *date2 = [dateFormatter dateFromString:obj2];
        return [date2 compare:date1];
    }];
    for (NSString *section in self.groupedVideos) {
        [self.groupedVideos[section] sortUsingComparator:^NSComparisonResult(NSString *path1, NSString *path2) {
            NSDictionary *attr1 = [[NSFileManager defaultManager] attributesOfItemAtPath:path1 error:nil];
            NSDictionary *attr2 = [[NSFileManager defaultManager] attributesOfItemAtPath:path2 error:nil];
            return [attr2.fileCreationDate compare:attr1.fileCreationDate];
        }];
    }
    [self.tableView reloadData];
}

- (UIImage *)generateThumbnailForVideoAtPath:(NSString *)videoPath {
    NSURL *videoURL = [NSURL fileURLWithPath:videoPath];
    AVAsset *asset = [AVAsset assetWithURL:videoURL];
    AVAssetImageGenerator *generator = [[AVAssetImageGenerator alloc] initWithAsset:asset];
    generator.appliesPreferredTrackTransform = YES;

    CMTime duration = asset.duration;
    Float64 durationInSeconds = CMTimeGetSeconds(duration);
    Float64 middleTime = durationInSeconds / 1.75;

    CMTime time = CMTimeMakeWithSeconds(middleTime, 600);
    NSError *error = nil;
    CGImageRef imageRef = [generator copyCGImageAtTime:time actualTime:NULL error:&error];

    if (!imageRef) return nil;

    UIImage *thumbnail = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    return thumbnail;
}

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return self.sectionTitles.count;
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    NSString *sectionTitle = self.sectionTitles[indexPath.section];
    NSString *filePath = self.groupedVideos[sectionTitle][indexPath.row];
    NSString *extension = filePath.pathExtension.lowercaseString;
    
    if ([extension isEqualToString:@"mp4"]) {
        AVPlayerViewController *playerVC = [[AVPlayerViewController alloc] init];
        playerVC.player = [AVPlayer playerWithURL:[NSURL fileURLWithPath:filePath]];
        [self presentViewController:playerVC animated:YES completion:^{
            [playerVC.player play];
        }];
    } else if ([extension isEqualToString:@"aes"]) {
        NSError *readError = nil;
        NSData *encryptedData = [NSData dataWithContentsOfURL:[NSURL fileURLWithPath:filePath] options:0 error:&readError];
        if (!encryptedData) {
            [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to read the file: %@", readError.localizedDescription]];
            return;
        }
        
        NSArray<NSString *> *storedKeys = [[SecretManager sharedManager] getAllDecryptionKeys];
        __block NSData *decryptedData = nil;
        __block NSString *successfulKey = nil;

        for (NSString *key in storedKeys) {
            decryptedData = [[SecretManager sharedManager] decryptData:encryptedData withKey:key];
            if (decryptedData) {
                successfulKey = key;
                break;
            }
        }

        if (decryptedData) {
            // Decryption succeeded with a stored key, proceed to play
            NSURL *decryptedURL = [self handleDecryptedData:decryptedData fromURL:[NSURL fileURLWithPath:filePath]];
            if (decryptedURL) {
                AVPlayerViewController *playerVC = [[AVPlayerViewController alloc] init];
                playerVC.player = [AVPlayer playerWithURL:decryptedURL];
                [self presentViewController:playerVC animated:YES completion:^{
                    [playerVC.player play];
                }];
            }
        } else {
            // No stored key worked, prompt the user
            [self promptUserForKeyWithAESFileURL:[NSURL fileURLWithPath:filePath] encryptedData:encryptedData];
        }
    }
    
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
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
                [[SecretManager sharedManager] saveDecryptionKey:userProvidedKey withIdentifier:keyIdentifier error:&saveError];
                
                // Save the decrypted file
                NSURL *decryptedURL = [self handleDecryptedData:decryptedData fromURL:aesFileURL];
                if (decryptedURL) {
                    // Clear current state
                    [self.videoFiles removeAllObjects];
                    [self.groupedVideos removeAllObjects];
                    [self.sectionTitles removeAllObjects];
                    [self.loadedFilenames removeAllObjects];
                    
                    // Reload all videos to reprocess with the new key
                    [self loadExistingVideos];
                    
                    // Play the decrypted video
                    AVPlayerViewController *playerVC = [[AVPlayerViewController alloc] init];
                    playerVC.player = [AVPlayer playerWithURL:decryptedURL];
                    [self presentViewController:playerVC animated:YES completion:^{
                        [playerVC.player play];
                    }];
                }
            } else {
                [self showErrorAlertWithMessage:@"The provided key is incorrect. Please try again or cancel." completion:^{
                    [self promptUserForKeyWithAESFileURL:aesFileURL encryptedData:encryptedData];
                }];
            }
        }
    }];
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

- (void)showErrorAlertWithMessage:(NSString *)message {
    [self showErrorAlertWithMessage:message completion:nil];
}

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
    [self presentViewController:alert animated:YES completion:nil];
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    NSString *sectionTitle = self.sectionTitles[section];
    return self.groupedVideos[sectionTitle].count;
}

- (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    return self.sectionTitles[section];
}

- (UIView *)tableView:(UITableView *)tableView viewForHeaderInSection:(NSInteger)section {
    UIView *headerView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, tableView.bounds.size.width, 44)];
    headerView.backgroundColor = [UIColor systemGroupedBackgroundColor];
    
    UILabel *label = [[UILabel alloc] initWithFrame:CGRectMake(16, 0, tableView.bounds.size.width - 32, 44)];
    label.text = [self tableView:tableView titleForHeaderInSection:section];
    label.font = [UIFont systemFontOfSize:15 weight:UIFontWeightSemibold];
    label.textColor = [UIColor darkGrayColor];
    
    [headerView addSubview:label];
    return headerView;
}

- (CGFloat)tableView:(UITableView *)tableView heightForHeaderInSection:(NSInteger)section {
    return 44;
}

- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath {
    return 100;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    VideoTableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"VideoCell" forIndexPath:indexPath];
    
    NSString *sectionTitle = self.sectionTitles[indexPath.section];
    NSString *filePath = self.groupedVideos[sectionTitle][indexPath.row];
    NSString *filename = [filePath lastPathComponent];
    NSString *extension = filePath.pathExtension.lowercaseString;
    
    NSArray *components = [filename componentsSeparatedByString:@"_"];
    if (components.count >= 2) {
        NSString *timePart = components[1];
        NSArray *timeComponents = [timePart componentsSeparatedByString:@"-"];
        if (timeComponents.count >= 3) {
            NSString *hour = timeComponents[0];
            NSString *minute = timeComponents[1];
            NSString *second = timeComponents[2];
            cell.titleLabel.text = [NSString stringWithFormat:@"%@:%@:%@", hour, minute, second];
        } else {
            cell.titleLabel.text = filename;
        }
    } else {
        cell.titleLabel.text = filename;
    }
    
    cell.thumbnailView.image = nil;
    
    if ([extension isEqualToString:@"mp4"]) {
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            UIImage *thumbnail = [self generateThumbnailForVideoAtPath:filePath];
            dispatch_async(dispatch_get_main_queue(), ^{
                VideoTableViewCell *updateCell = [tableView cellForRowAtIndexPath:indexPath];
                if (updateCell) {
                    updateCell.thumbnailView.image = thumbnail;
                }
            });
        });
    } else if ([extension isEqualToString:@"aes"]) {
        UIImage *lockImage = [UIImage systemImageNamed:@"lock.fill"];
        UIImageSymbolConfiguration *config = [UIImageSymbolConfiguration configurationWithPointSize:40];
        UIImage *scaledLockImage = [lockImage imageByApplyingSymbolConfiguration:config];
        cell.thumbnailView.image = scaledLockImage;
        cell.thumbnailView.contentMode = UIViewContentModeCenter;
        cell.thumbnailView.tintColor = [UIColor systemGrayColor];
    }
    
    [cell.menuButton addTarget:self action:@selector(menuTapped:forEvent:) forControlEvents:UIControlEventTouchUpInside];
    
    return cell;
}

- (void)menuTapped:(UIButton *)sender forEvent:(UIEvent *)event {
    CGPoint touchPoint = [sender convertPoint:CGPointZero toView:self.tableView];
    NSIndexPath *indexPath = [self.tableView indexPathForRowAtPoint:touchPoint];
    
    if (!indexPath)  return;
    
    NSInteger section = indexPath.section;
    NSInteger row = indexPath.row;
    
    if (section >= self.sectionTitles.count) return;
    
    NSString *sectionTitle = self.sectionTitles[section];
    
    if (row >= self.groupedVideos[sectionTitle].count) return;
    
    NSString *videoPath = self.groupedVideos[sectionTitle][row];
    NSString *filename = [videoPath lastPathComponent];
    NSString *originalAesPath = [self.downloadDirectory stringByAppendingPathComponent:[filename stringByAppendingPathExtension:@"aes"]];
    
    UIAlertController *actionSheet = [UIAlertController alertControllerWithTitle:nil
                                                                         message:nil
                                                                  preferredStyle:UIAlertControllerStyleActionSheet];
    
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Share"
                                                    style:UIAlertActionStyleDefault
                                                  handler:^(UIAlertAction * _Nonnull action) {
        NSURL *videoURL = [NSURL fileURLWithPath:videoPath];
        UIActivityViewController *activityVC = [[UIActivityViewController alloc] initWithActivityItems:@[videoURL]
                                                                             applicationActivities:nil];
        [self presentViewController:activityVC animated:YES completion:nil];
    }]];
    
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Delete"
                                                    style:UIAlertActionStyleDestructive
                                                  handler:^(UIAlertAction * _Nonnull action) {
        NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
        NSError *localError;
        NSFileManager *fileManager = [NSFileManager defaultManager];
        
        if ([fileManager fileExistsAtPath:videoPath]) {
            [fileManager removeItemAtPath:videoPath error:&localError];
            if (localError) return;
        }
        
        if ([fileManager fileExistsAtPath:originalAesPath]) {
            [fileManager removeItemAtPath:originalAesPath error:&localError];
            if (localError) return;
        }
        
        [self.videoFiles removeObject:videoPath];
        [self.groupedVideos[sectionTitle] removeObjectAtIndex:row];
        
        [self.tableView beginUpdates];
        if ([self.groupedVideos[sectionTitle] count] == 0) {
            [self.groupedVideos removeObjectForKey:sectionTitle];
            [self.sectionTitles removeObjectAtIndex:section];
            [self.tableView deleteSections:[NSIndexSet indexSetWithIndex:section]
                          withRowAnimation:UITableViewRowAnimationAutomatic];
        } else {
            [self.tableView deleteRowsAtIndexPaths:@[[NSIndexPath indexPathForRow:row inSection:section]]
                                  withRowAnimation:UITableViewRowAnimationAutomatic];
        }
        [self.tableView endUpdates];
        
        if (sessionToken) {
            NSString *backendFilename = [filename hasSuffix:@".aes"] ? filename : [filename stringByAppendingPathExtension:@"aes"];
            NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/video"];
            components.queryItems = @[
                [NSURLQueryItem queryItemWithName:@"session_token" value:sessionToken],
                [NSURLQueryItem queryItemWithName:@"name" value:backendFilename]
            ];
            
            NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:components.URL];
            [request setHTTPMethod:@"DELETE"];
            
            [[self.downloadSession dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
                NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
            }] resume];
        }
    }]];
    
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Cancel"
                                                    style:UIAlertActionStyleCancel
                                                  handler:nil]];
    
    if (UIDevice.currentDevice.userInterfaceIdiom == UIUserInterfaceIdiomPad) {
        actionSheet.popoverPresentationController.sourceView = sender;
        actionSheet.popoverPresentationController.sourceRect = sender.bounds;
    }
    
    [self presentViewController:actionSheet animated:YES completion:nil];
}

- (NSURL *)handleDecryptedData:(NSData *)decryptedData fromURL:(NSURL *)aesFileURL {
    NSString *fileName = [aesFileURL lastPathComponent];
    if ([fileName hasSuffix:@".aes"]) {
        fileName = [fileName stringByReplacingOccurrencesOfString:@".aes" withString:@"" options:NSBackwardsSearch range:NSMakeRange(0, fileName.length)];
    }
    NSURL *decFileURL = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask][0] URLByAppendingPathComponent:fileName];
    
    NSError *writeError = nil;
    [decryptedData writeToURL:decFileURL options:NSDataWritingAtomic error:&writeError];
    if (writeError) return nil;
    
    NSError *attributesError = nil;
    NSDictionary *originalAttributes = [[NSFileManager defaultManager] attributesOfItemAtPath:[aesFileURL path] error:&attributesError];
    if (originalAttributes && !attributesError) {
        NSDate *originalCreationDate = originalAttributes[NSFileCreationDate];
        if (originalCreationDate) {
            NSDictionary *newAttributes = @{NSFileCreationDate: originalCreationDate};
            [[NSFileManager defaultManager] setAttributes:newAttributes ofItemAtPath:[decFileURL path] error:&attributesError];
        }
    }
    return decFileURL;
}

@end

