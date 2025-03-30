#import "GalleryViewController.h"
#import "StoreManager.h"
#import <AVKit/AVKit.h>
#import <AVFoundation/AVFoundation.h>

@interface VideoTableViewCell : UITableViewCell
@property (nonatomic, strong) UIImageView *thumbnailView;
@property (nonatomic, strong) UILabel *titleLabel;
@property (nonatomic, strong) UIButton *menuButton; // Three-dot menu button (vertical)
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
    [self.contentView addSubview:self.thumbnailView];
    
    self.titleLabel = [[UILabel alloc] init];
    self.titleLabel.numberOfLines = 2;
    [self.contentView addSubview:self.titleLabel];
    
    self.menuButton = [UIButton buttonWithType:UIButtonTypeSystem];
    // Create a vertical ellipsis (three dots) using SF Symbols
    UIImage *ellipsisImage = [UIImage systemImageNamed:@"ellipsis"];
    UIImageSymbolConfiguration *config = [UIImageSymbolConfiguration configurationWithScale:UIImageSymbolScaleLarge];
    UIImage *verticalEllipsis = [ellipsisImage imageByApplyingSymbolConfiguration:config];
    [self.menuButton setImage:verticalEllipsis forState:UIControlStateNormal];
    self.menuButton.tintColor = [UIColor systemGrayColor];
    self.menuButton.transform = CGAffineTransformMakeRotation(M_PI_2); // Rotate 90 degrees to make it vertical
    [self.contentView addSubview:self.menuButton];
    
    self.thumbnailView.translatesAutoresizingMaskIntoConstraints = NO;
    self.titleLabel.translatesAutoresizingMaskIntoConstraints = NO;
    self.menuButton.translatesAutoresizingMaskIntoConstraints = NO;
    
    [NSLayoutConstraint activateConstraints:@[
        [self.thumbnailView.leadingAnchor constraintEqualToAnchor:self.contentView.safeAreaLayoutGuide.leadingAnchor constant:15],
        [self.thumbnailView.centerYAnchor constraintEqualToAnchor:self.contentView.centerYAnchor],
        [self.thumbnailView.widthAnchor constraintEqualToConstant:150], // Bigger thumbnail
        [self.thumbnailView.heightAnchor constraintEqualToConstant:90],
        
        [self.titleLabel.leadingAnchor constraintEqualToAnchor:self.thumbnailView.trailingAnchor constant:10],
        [self.titleLabel.trailingAnchor constraintEqualToAnchor:self.menuButton.leadingAnchor constant:-10],
        [self.titleLabel.centerYAnchor constraintEqualToAnchor:self.contentView.centerYAnchor],
        
        [self.menuButton.trailingAnchor constraintEqualToAnchor:self.contentView.safeAreaLayoutGuide.trailingAnchor constant:-15],
        [self.menuButton.centerYAnchor constraintEqualToAnchor:self.contentView.centerYAnchor],
        [self.menuButton.widthAnchor constraintEqualToConstant:30],
        [self.menuButton.heightAnchor constraintEqualToConstant:30]
    ]];
}

@end

@interface GalleryViewController () <UITableViewDelegate, UITableViewDataSource>
@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSMutableArray<NSString *> *videoFiles;
@property (nonatomic, strong) NSString *downloadDirectory;
@property (nonatomic, strong) NSURLSession *downloadSession;
@end

@implementation GalleryViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Events";
    self.navigationController.navigationBarHidden = NO;
    
    self.videoFiles = [NSMutableArray array];
    
    NSURLSessionConfiguration *config = [NSURLSessionConfiguration defaultSessionConfiguration];
    config.timeoutIntervalForRequest = 60.0;
    config.timeoutIntervalForResource = 600.0;
    self.downloadSession = [NSURLSession sessionWithConfiguration:config];
    
    [self setupDownloadDirectory];
    [self setupTableView];
    
    [self loadExistingVideos];
    [self getEvents];
}

- (void)viewWillLayoutSubviews {
    [super viewWillLayoutSubviews];
    self.tableView.frame = self.view.bounds;
}

- (void)deleteDownloadDirectory {
    NSString *documentsDir = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    self.downloadDirectory = [documentsDir stringByAppendingPathComponent:@"downloaded-events"];
    
    NSError *error;
    if ([[NSFileManager defaultManager] fileExistsAtPath:self.downloadDirectory]) {
        [[NSFileManager defaultManager] removeItemAtPath:self.downloadDirectory error:&error];
        if (error) {
            NSLog(@"Failed to clear directory: %@", error.localizedDescription);
        }
    }
    
    [[NSFileManager defaultManager] createDirectoryAtPath:self.downloadDirectory
                          withIntermediateDirectories:YES
                                           attributes:nil
                                                error:&error];
    if (error) {
        NSLog(@"Failed to create directory: %@", error.localizedDescription);
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
    self.tableView = [[UITableView alloc] initWithFrame:self.view.bounds style:UITableViewStylePlain];
    self.tableView.delegate = self;
    self.tableView.dataSource = self;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor];
    [self.tableView registerClass:[VideoTableViewCell class] forCellReuseIdentifier:@"VideoCell"];
    self.tableView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.tableView];
    
    [NSLayoutConstraint activateConstraints:@[
        [self.tableView.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [self.tableView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.tableView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.tableView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor]
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
                    
                    NSDictionary *fileAttributes = [[NSFileManager defaultManager] attributesOfItemAtPath:destPath error:nil];
                    if (fileAttributes) {
                        NSDate *creationDate = fileAttributes[NSFileCreationDate];
                        if (creationDate) {
                            NSLog(@"File %@ creation date: %@", fileName, creationDate);
                        } else {
                            NSLog(@"File %@ has no creation date available", fileName);
                        }
                    } else {
                        NSLog(@"Could not retrieve attributes for file: %@", fileName);
                    }
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
    
    NSError *error;
    NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:self.downloadDirectory error:&error];
    
    NSMutableArray *filesWithDates = [NSMutableArray array];
    
    for (NSString *file in contents) {
        if ([file.pathExtension isEqualToString:@"mp4"]) {
            NSString *filePath = [self.downloadDirectory stringByAppendingPathComponent:file];
            NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:filePath error:nil];
            NSDate *creationDate = [attributes fileCreationDate];
            
            [filesWithDates addObject:@{@"path": filePath,
                                      @"date": creationDate ?: [NSDate distantPast]}];
        }
    }
    
    [filesWithDates sortUsingComparator:^NSComparisonResult(NSDictionary *obj1, NSDictionary *obj2) {
        return [obj2[@"date"] compare:obj1[@"date"]];
    }];
    
    for (NSDictionary *fileInfo in filesWithDates) {
        [self.videoFiles addObject:fileInfo[@"path"]];
    }
    
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

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return self.videoFiles.count;
}

- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath {
    return 110; // Adjusted for bigger thumbnail
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    VideoTableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"VideoCell" forIndexPath:indexPath];
    
    NSString *videoPath = self.videoFiles[indexPath.row];
    cell.titleLabel.text = [videoPath lastPathComponent];
    cell.thumbnailView.image = nil;
    
    [cell.menuButton addTarget:self action:@selector(menuTapped:forEvent:) forControlEvents:UIControlEventTouchUpInside];
    cell.menuButton.tag = indexPath.row;
    
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

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    AVPlayerViewController *playerVC = [[AVPlayerViewController alloc] init];
    playerVC.player = [AVPlayer playerWithURL:[NSURL fileURLWithPath:self.videoFiles[indexPath.row]]];
    [self presentViewController:playerVC animated:YES completion:^{ [playerVC.player play]; }];
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

- (void)menuTapped:(UIButton *)sender forEvent:(UIEvent *)event {
    NSInteger row = sender.tag;
    NSString *videoPath = self.videoFiles[row];
    
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
        NSError *error;
        [[NSFileManager defaultManager] removeItemAtPath:videoPath error:&error];
        if (error) {
            NSLog(@"Failed to delete video: %@", error.localizedDescription);
        } else {
            [self.videoFiles removeObjectAtIndex:row];
            [self.tableView deleteRowsAtIndexPaths:@[[NSIndexPath indexPathForRow:row inSection:0]]
                                  withRowAnimation:UITableViewRowAnimationAutomatic];
        }
    }]];
    
    [actionSheet addAction:[UIAlertAction actionWithTitle:@"Cancel"
                                                    style:UIAlertActionStyleCancel
                                                  handler:nil]];
    
    [self presentViewController:actionSheet animated:YES completion:nil];
}

@end
