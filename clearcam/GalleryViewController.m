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
        if (error) {
            NSLog(@"Failed to create directory: %@", error.localizedDescription);
        }
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
    
    if (sessionToken) {
        [queryItems addObject:[NSURLQueryItem queryItemWithName:@"session_token" value:sessionToken]];
    } else {
        NSLog(@"No session token found in Keychain. Proceeding without it.");
    }
    
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
        
        if (error) {
            NSLog(@"Request failed: %@", error.localizedDescription);
            return;
        }
        
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
        if (httpResponse.statusCode == 200) {
            NSError *jsonError;
            NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
            
            if (!jsonError && json[@"files"]) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self downloadFiles:json[@"files"]];
                });
            } else {
                NSLog(@"JSON parsing error: %@ or no 'files' key in response.", jsonError.localizedDescription);
            }
        } else {
            NSLog(@"Request failed with status code: %ld", (long)httpResponse.statusCode);
        }
    }] resume];
}

- (void)downloadFiles:(NSArray<NSString *> *)fileNames {
    if (fileNames.count == 0) {
        [self loadExistingVideos];
        return;
    }
    
    self.isLoadingVideos = YES;
    
    dispatch_group_t downloadGroup = dispatch_group_create();
    for (NSString *fileName in fileNames) {
        dispatch_group_enter(downloadGroup);
        
        NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/video"];
        NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
        if (!sessionToken) {
            NSLog(@"No session token found in Keychain.");
            dispatch_group_leave(downloadGroup);
            return;
        }

        NSURLQueryItem *sessionTokenItem = [NSURLQueryItem queryItemWithName:@"session_token" value:sessionToken];
        NSURLQueryItem *nameItem = [NSURLQueryItem queryItemWithName:@"name" value:fileName];
        components.queryItems = @[sessionTokenItem, nameItem];
        
        NSURL *url = components.URL;
        NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
        [request setHTTPMethod:@"GET"];
        
        [[self.downloadSession downloadTaskWithRequest:request completionHandler:^(NSURL *location, NSURLResponse *response, NSError *error) {
            if (!error && [(NSHTTPURLResponse *)response statusCode] == 200) {
                NSString *destPath = [self.downloadDirectory stringByAppendingPathComponent:fileName];
                NSError *moveError;
                NSURL *destURL = [NSURL fileURLWithPath:destPath];
                
                [[NSFileManager defaultManager] moveItemAtURL:location toURL:destURL error:&moveError];
                if (!moveError) {
                    NSLog(@"File downloaded to: %@", destPath);
                } else {
                    NSLog(@"File move failed with error: %@", moveError.localizedDescription);
                }
            } else {
                NSLog(@"Download failed with error: %@, status code: %ld", error.localizedDescription, (long)[(NSHTTPURLResponse *)response statusCode]);
            }
            dispatch_group_leave(downloadGroup);
        }] resume];
    }
    
    dispatch_group_notify(downloadGroup, dispatch_get_main_queue(), ^{
        self.isLoadingVideos = NO;
        [self loadExistingVideos];
    });
}

- (void)loadExistingVideos {
    [self.videoFiles removeAllObjects];
    [self.groupedVideos removeAllObjects];
    [self.sectionTitles removeAllObjects];
    
    NSError *error;
    NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:self.downloadDirectory error:&error];
    
    NSMutableArray *filesWithDates = [NSMutableArray array];
    
    for (NSString *file in contents) {
        NSString *extension = file.pathExtension.lowercaseString;
        if ([extension isEqualToString:@"mp4"] || [extension isEqualToString:@"aes"]) {
            NSString *filePath = [self.downloadDirectory stringByAppendingPathComponent:file];
            
            if([extension isEqualToString:@"aes"]){
                NSData *encryptedData = [NSData dataWithContentsOfURL:[NSURL fileURLWithPath:filePath] options:0 error:&error];
                if (encryptedData) {
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

                    if (decryptedData) {
                        NSURL *url = [self handleDecryptedData:decryptedData fromURL:[NSURL fileURLWithPath:filePath]];
                        filePath = [url path];
                    }
                }
            }
            
            NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:filePath error:nil];
            NSDate *creationDate = [attributes fileCreationDate];
            
            [filesWithDates addObject:@{@"path": filePath,
                                        @"date": creationDate ?: [NSDate distantPast],
                                        @"filename": file}];
        }
    }
    
    [filesWithDates sortUsingComparator:^NSComparisonResult(NSDictionary *obj1, NSDictionary *obj2) {
        return [obj2[@"date"] compare:obj1[@"date"]];
    }];
    
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    dateFormatter.dateStyle = NSDateFormatterMediumStyle;
    dateFormatter.timeStyle = NSDateFormatterNoStyle;
    
    for (NSDictionary *fileInfo in filesWithDates) {
        NSString *filePath = fileInfo[@"path"];
        [self.videoFiles addObject:filePath];
        
        NSString *filename = fileInfo[@"filename"];
        NSString *datePart = [[filename componentsSeparatedByString:@"_"] firstObject];
        
        NSDateFormatter *inputFormatter = [[NSDateFormatter alloc] init];
        [inputFormatter setDateFormat:@"yyyy-MM-dd"];
        NSDate *date = [inputFormatter dateFromString:datePart];
        
        NSString *sectionTitle = [dateFormatter stringFromDate:date] ?: @"Unknown Date";
        
        if (!self.groupedVideos[sectionTitle]) {
            self.groupedVideos[sectionTitle] = [NSMutableArray array];
            [self.sectionTitles addObject:sectionTitle];
        }
        [self.groupedVideos[sectionTitle] addObject:filePath];
    }
    
    [self.sectionTitles sortUsingComparator:^NSComparisonResult(NSString *obj1, NSString *obj2) {
        NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
        formatter.dateStyle = NSDateFormatterMediumStyle;
        formatter.timeStyle = NSDateFormatterNoStyle;
        
        NSDate *date1 = [formatter dateFromString:obj1];
        NSDate *date2 = [formatter dateFromString:obj2];
        
        return [date2 compare:date1];
    }];
    
    [self.tableView reloadData];
}

- (UIImage *)generateThumbnailForVideoAtPath:(NSString *)videoPath {
    NSURL *videoURL = [NSURL fileURLWithPath:videoPath];
    AVAsset *asset = [AVAsset assetWithURL:videoURL];
    AVAssetImageGenerator *generator = [[AVAssetImageGenerator alloc] initWithAsset:asset];
    generator.appliesPreferredTrackTransform = YES;

    // Get the duration of the video
    CMTime duration = asset.duration;
    Float64 durationInSeconds = CMTimeGetSeconds(duration);
    Float64 middleTime = durationInSeconds / 1.75; //little after

    // Set the time to the middle of the video
    CMTime time = CMTimeMakeWithSeconds(middleTime, 600);
    NSError *error = nil;
    CGImageRef imageRef = [generator copyCGImageAtTime:time actualTime:NULL error:&error];

    if (!imageRef) {
        NSLog(@"Thumbnail generation failed: %@", error.localizedDescription);
        return nil;
    }

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
        UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Encrypted File"
                                                                      message:@"This file is encrypted (.aes) and cannot be played."
                                                               preferredStyle:UIAlertControllerStyleAlert];
        [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
        [self presentViewController:alert animated:YES completion:nil];
    }
    
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
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
    
    // Set title label (same logic as before)
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
        // Generate thumbnail for MP4 files
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
        // Use a lock icon for AES files
        UIImage *lockImage = [UIImage systemImageNamed:@"lock.fill"];
        UIImageSymbolConfiguration *config = [UIImageSymbolConfiguration configurationWithPointSize:40];
        UIImage *scaledLockImage = [lockImage imageByApplyingSymbolConfiguration:config];
        cell.thumbnailView.image = scaledLockImage;
        cell.thumbnailView.contentMode = UIViewContentModeCenter; // Center the lock icon
        cell.thumbnailView.tintColor = [UIColor systemGrayColor];
    }
    
    [cell.menuButton addTarget:self action:@selector(menuTapped:forEvent:) forControlEvents:UIControlEventTouchUpInside];
    
    return cell;
}

- (void)menuTapped:(UIButton *)sender forEvent:(UIEvent *)event {
    CGPoint touchPoint = [sender convertPoint:CGPointZero toView:self.tableView];
    NSIndexPath *indexPath = [self.tableView indexPathForRowAtPoint:touchPoint];
    
    if (!indexPath) {
        NSLog(@"Failed to get indexPath for menu button tap");
        return;
    }
    
    NSInteger section = indexPath.section;
    NSInteger row = indexPath.row;
    
    if (section >= self.sectionTitles.count) {
        NSLog(@"Invalid section: %ld, total sections: %lu", (long)section, (unsigned long)self.sectionTitles.count);
        return;
    }
    
    NSString *sectionTitle = self.sectionTitles[section];
    
    if (row >= self.groupedVideos[sectionTitle].count) {
        NSLog(@"Invalid row: %ld for section '%@', total rows: %lu", (long)row, sectionTitle, (unsigned long)self.groupedVideos[sectionTitle].count);
        return;
    }
    
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
        // Retrieve session token
        NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
        if (!sessionToken) {
            NSLog(@"No session token found in Keychain. Proceeding with local deletion only.");
        }
        
        // Perform local deletion for both decrypted .mp4 and original .aes (if they exist)
        NSError *localError;
        NSFileManager *fileManager = [NSFileManager defaultManager];
        
        // Delete decrypted .mp4 file (if it exists)
        if ([fileManager fileExistsAtPath:videoPath]) {
            [fileManager removeItemAtPath:videoPath error:&localError];
            if (localError) {
                NSLog(@"Failed to delete decrypted file at %@: %@", videoPath, localError.localizedDescription);
                return;
            }
        }
        
        // Delete original .aes file (if it exists)
        if ([fileManager fileExistsAtPath:originalAesPath]) {
            [fileManager removeItemAtPath:originalAesPath error:&localError];
            if (localError) {
                NSLog(@"Failed to delete original .aes file at %@: %@", originalAesPath, localError.localizedDescription);
                return;
            }
        }
        
        // Update data source and table view
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
        
        // Send DELETE request to backend using the original .aes filename
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
                if (!error && httpResponse.statusCode == 200) {
                    NSLog(@"Successfully deleted %@ from backend", backendFilename);
                } else {
                    NSLog(@"Failed to delete from backend: %@, status code: %ld", error.localizedDescription, (long)httpResponse.statusCode);
                }
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
    // Remove only the .aes extension
    NSString *fileName = [aesFileURL lastPathComponent];
    if ([fileName hasSuffix:@".aes"]) {
        fileName = [fileName stringByReplacingOccurrencesOfString:@".aes" withString:@"" options:NSBackwardsSearch range:NSMakeRange(0, fileName.length)];
    }
    NSURL *decFileURL = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask][0] URLByAppendingPathComponent:fileName];
    
    NSError *writeError = nil;
    [decryptedData writeToURL:decFileURL options:NSDataWritingAtomic error:&writeError];
    if (writeError) {
        NSLog(@"Failed to write decrypted data: %@", writeError.localizedDescription);
        return nil;
    }
    
    // Preserve the original creation date
    NSError *attributesError = nil;
    NSDictionary *originalAttributes = [[NSFileManager defaultManager] attributesOfItemAtPath:[aesFileURL path] error:&attributesError];
    if (originalAttributes && !attributesError) {
        NSDate *originalCreationDate = originalAttributes[NSFileCreationDate];
        if (originalCreationDate) {
            NSDictionary *newAttributes = @{NSFileCreationDate: originalCreationDate};
            [[NSFileManager defaultManager] setAttributes:newAttributes ofItemAtPath:[decFileURL path] error:&attributesError];
            if (attributesError) {
                NSLog(@"Failed to set creation date: %@", attributesError.localizedDescription);
            }
        }
    } else {
        NSLog(@"Failed to get original attributes: %@", attributesError.localizedDescription);
    }
    
    return decFileURL;
}

@end
