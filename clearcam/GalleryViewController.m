#import "GalleryViewController.h"
#import "StoreManager.h"
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
    
    NSURLSessionConfiguration *config = [NSURLSessionConfiguration defaultSessionConfiguration];
    config.timeoutIntervalForRequest = 60.0;
    config.timeoutIntervalForResource = 600.0;
    self.downloadSession = [NSURLSession sessionWithConfiguration:config];
    
    [self setupDownloadDirectory];
    [self setupTableView];
    
    [self loadExistingVideos];
    [self getEvents];
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
    [self.view addSubview:self.tableView];
    
    // Enable Auto Layout
    self.tableView.translatesAutoresizingMaskIntoConstraints = NO;
    
    // Pin to safe area
    [NSLayoutConstraint activateConstraints:@[
        [self.tableView.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor],
        [self.tableView.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor],
        [self.tableView.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor],
        [self.tableView.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor]
    ]];
}

- (void)viewWillLayoutSubviews {
    [super viewWillLayoutSubviews];
    self.tableView.frame = self.view.safeAreaLayoutGuide.layoutFrame;
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
        if ([file.pathExtension isEqualToString:@"mp4"]) {
            NSString *filePath = [self.downloadDirectory stringByAppendingPathComponent:file];
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
    
    // Group videos by date
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
    
    // Sort section titles in descending order
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
    
    CMTime time = CMTimeMakeWithSeconds(0.0, 600);
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
    NSString *videoPath = self.groupedVideos[sectionTitle][indexPath.row];
    NSString *filename = [videoPath lastPathComponent];
    
    // Extract and format the full time from filename (format: yyyy-MM-dd_HH-mm-ss)
    NSArray *components = [filename componentsSeparatedByString:@"_"];
    if (components.count >= 2) {
        NSString *timePart = components[1]; // Gets "HH-mm-ss" part
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
    
    [cell.menuButton addTarget:self action:@selector(menuTapped:forEvent:) forControlEvents:UIControlEventTouchUpInside];
    cell.menuButton.tag = indexPath.section * 1000 + indexPath.row;
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        UIImage *thumbnail = [self generateThumbnailForVideoAtPath:videoPath];
        dispatch_async(dispatch_get_main_queue(), ^{
            VideoTableViewCell *updateCell = [tableView cellForRowAtIndexPath:indexPath];
            if (updateCell) {
                updateCell.thumbnailView.image = thumbnail;
            }
        });
    });
    
    return cell;
}

- (void)menuTapped:(UIButton *)sender forEvent:(UIEvent *)event {
    NSInteger combinedTag = sender.tag;
    NSInteger section = combinedTag / 1000;
    NSInteger row = combinedTag % 1000;
    
    NSString *sectionTitle = self.sectionTitles[section];
    NSString *videoPath = self.groupedVideos[sectionTitle][row];
    NSString *filename = [videoPath lastPathComponent];
    
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
            NSLog(@"No session token found in Keychain. Cannot send delete request to backend.");
            // Optionally proceed with local deletion or show an error
            return;
        }
        
        // Prepare DELETE request to backend
        NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/video"];
        components.queryItems = @[
            [NSURLQueryItem queryItemWithName:@"session_token" value:sessionToken],
            [NSURLQueryItem queryItemWithName:@"name" value:filename]
        ];
        
        NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:components.URL];
        [request setHTTPMethod:@"DELETE"];
        
        // Send DELETE request to backend
        [[self.downloadSession dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
            NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
            if (!error && httpResponse.statusCode == 200) {
                NSLog(@"Successfully deleted %@ from backend", filename);
            } else {
                NSLog(@"Failed to delete from backend: %@, status code: %ld", error.localizedDescription, (long)httpResponse.statusCode);
                // Optionally show an alert here if backend deletion fails
            }
            
            // Proceed with local deletion regardless of backend success (or adjust logic as needed)
            NSError *localError;
            [[NSFileManager defaultManager] removeItemAtPath:videoPath error:&localError];
            if (localError) {
                NSLog(@"Failed to delete local video: %@", localError.localizedDescription);
            } else {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self.videoFiles removeObject:videoPath];
                    [self.groupedVideos[sectionTitle] removeObjectAtIndex:row];
                    
                    if ([self.groupedVideos[sectionTitle] count] == 0) {
                        [self.groupedVideos removeObjectForKey:sectionTitle];
                        [self.sectionTitles removeObjectAtIndex:section];
                        [self.tableView deleteSections:[NSIndexSet indexSetWithIndex:section]
                                      withRowAnimation:UITableViewRowAnimationAutomatic];
                    } else {
                        [self.tableView deleteRowsAtIndexPaths:@[[NSIndexPath indexPathForRow:row inSection:section]]
                                              withRowAnimation:UITableViewRowAnimationAutomatic];
                    }
                });
            }
        }] resume];
    }]];
    
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Cancel"
                                                   style:UIAlertActionStyleCancel
                                                 handler:nil]];
    
    // For iPad
    if (UIDevice.currentDevice.userInterfaceIdiom == UIUserInterfaceIdiomPad) {
        actionSheet.popoverPresentationController.sourceView = sender;
        actionSheet.popoverPresentationController.sourceRect = sender.bounds;
    }
    
    [self presentViewController:actionSheet animated:YES completion:nil];
}

@end

