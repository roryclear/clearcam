#import "GalleryViewController.h"
#import "StoreManager.h"
#import "SecretManager.h"
#import <AVKit/AVKit.h>
#import <AVFoundation/AVFoundation.h>
#import "LiveViewController.h"
#import "ViewController.h"
#import "FileServer.h"
#import "SettingsViewController.h"

@interface VideoTableViewCell : UITableViewCell
@property (nonatomic, strong) UIImageView *thumbnailView;
@property (nonatomic, strong) UIButton *menuButton;
@property (nonatomic, strong) UILabel *titleLabel;
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

// Modified GalleryViewController with tabs
@interface GalleryViewController () <UITableViewDelegate, UITableViewDataSource>
@property (nonatomic, strong) UITabBarController *tabController;
@property (nonatomic, strong) UIViewController *eventsViewController;
@property (nonatomic, strong) LiveViewController *liveViewController;
@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSMutableArray<NSString *> *videoFiles;
@property (nonatomic, strong) NSString *downloadDirectory;
@property (nonatomic, strong) NSURLSession *downloadSession;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSMutableArray<NSString *> *> *groupedVideos;
@property (nonatomic, strong) NSMutableArray<NSString *> *sectionTitles;
@property (nonatomic, strong) UIRefreshControl *refreshControl;
@property (nonatomic, assign) BOOL isLoadingVideos;
@property (nonatomic, strong) NSMutableSet<NSString *> *loadedFilenames;
@property (nonatomic, strong) NSTimer *refreshTimer;
@property (nonatomic, strong) UIView *headerView;
@property (nonatomic, strong) FileServer *fileServer;
@end

@implementation GalleryViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.fileServer = [FileServer sharedInstance];
    [self.fileServer start];
    [self.navigationController setNavigationBarHidden:YES animated:NO];
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    
    // Create header view
    self.headerView = [[UIView alloc] init];
    self.headerView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.headerView];
    
    // Camera button
    self.cameraButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.cameraButton setImage:[UIImage systemImageNamed:@"camera.fill"] forState:UIControlStateNormal];
    self.cameraButton.tintColor = [[UIColor grayColor] colorWithAlphaComponent:0.9];
    [self.cameraButton addTarget:self action:@selector(cameraTapped) forControlEvents:UIControlEventTouchUpInside];
    self.cameraButton.translatesAutoresizingMaskIntoConstraints = NO;
    
    // App icon
    NSArray *iconFiles = [[[NSBundle mainBundle] infoDictionary] valueForKeyPath:@"CFBundleIcons.CFBundlePrimaryIcon.CFBundleIconFiles"];
    NSString *iconName = [iconFiles lastObject];
    UIImageView *appIconView = [[UIImageView alloc] initWithImage:[UIImage imageNamed:iconName]];
    appIconView.contentMode = UIViewContentModeScaleAspectFit;
    appIconView.translatesAutoresizingMaskIntoConstraints = NO;
    appIconView.layer.cornerRadius = 4;
    appIconView.layer.masksToBounds = YES;
    [appIconView.widthAnchor constraintEqualToConstant:24].active = YES;
    [appIconView.heightAnchor constraintEqualToConstant:24].active = YES;
    
    // Settings button
    self.settingsButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.settingsButton setImage:[UIImage systemImageNamed:@"gearshape.fill"] forState:UIControlStateNormal];
    self.settingsButton.tintColor = [[UIColor grayColor] colorWithAlphaComponent:0.9];
    [self.settingsButton addTarget:self action:@selector(settingsTapped) forControlEvents:UIControlEventTouchUpInside];
    self.settingsButton.translatesAutoresizingMaskIntoConstraints = NO;
    
    // IP label
    self.ipLabel = [[UILabel alloc] init];
    self.ipLabel.font = [UIFont systemFontOfSize:14 weight:UIFontWeightMedium];
    self.ipLabel.textColor = [UIColor secondaryLabelColor];
    self.ipLabel.textAlignment = NSTextAlignmentCenter;
    self.ipLabel.numberOfLines = 1;
    self.ipLabel.translatesAutoresizingMaskIntoConstraints = NO;

    UIStackView *titleStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.cameraButton, appIconView, self.settingsButton]];
    titleStack.axis = UILayoutConstraintAxisHorizontal;
    titleStack.distribution = UIStackViewDistributionEqualCentering;
    titleStack.alignment = UIStackViewAlignmentCenter;
    titleStack.spacing = 16;
    titleStack.translatesAutoresizingMaskIntoConstraints = NO;
    
    // Vertical stack for entire header
    UIStackView *headerStack = [[UIStackView alloc] initWithArrangedSubviews:@[titleStack, self.ipLabel]];
    headerStack.axis = UILayoutConstraintAxisVertical;
    headerStack.spacing = 8;
    headerStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.headerView addSubview:headerStack];

    self.eventsViewController = [[UIViewController alloc] init];
    self.eventsViewController.title = NSLocalizedString(@"events", @"Title for events screen");
    self.eventsViewController.tabBarItem = [[UITabBarItem alloc] initWithTitle:NSLocalizedString(@"events", @"Title for events screen")
                                                                         image:[UIImage systemImageNamed:@"photo.on.rectangle"]
                                                                           tag:0];
    
    self.liveViewController = [[LiveViewController alloc] init];
    self.liveViewController.title = @"Live Cameras";
    self.liveViewController.tabBarItem = [[UITabBarItem alloc] initWithTitle:NSLocalizedString(@"live", @"Title for live screen")
                                                                       image:[UIImage systemImageNamed:@"video"]
                                                                         tag:1];
    
    self.tabController = [[UITabBarController alloc] init];
    self.tabController.viewControllers = @[self.eventsViewController, self.liveViewController];
    
    [self addChildViewController:self.tabController];
    [self.view addSubview:self.tabController.view];
    self.tabController.view.translatesAutoresizingMaskIntoConstraints = NO;
    [self.tabController didMoveToParentViewController:self];

    [NSLayoutConstraint activateConstraints:@[
        [self.headerView.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor],
        [self.headerView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.headerView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.headerView.heightAnchor constraintEqualToConstant:90],

        [headerStack.topAnchor constraintEqualToAnchor:self.headerView.topAnchor constant:12],
        [headerStack.leadingAnchor constraintEqualToAnchor:self.headerView.leadingAnchor constant:16],
        [headerStack.trailingAnchor constraintEqualToAnchor:self.headerView.trailingAnchor constant:-16],
        [headerStack.bottomAnchor constraintEqualToAnchor:self.headerView.bottomAnchor constant:-8],
        
        [self.cameraButton.widthAnchor constraintEqualToConstant:44],
        [self.cameraButton.heightAnchor constraintEqualToConstant:44],
        [self.settingsButton.widthAnchor constraintEqualToConstant:44],
        [self.settingsButton.heightAnchor constraintEqualToConstant:44],

        [self.tabController.view.topAnchor constraintEqualToAnchor:self.headerView.bottomAnchor],
        [self.tabController.view.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.tabController.view.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.tabController.view.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor]
    ]];
    
    [self setupEventsViewController];

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
    [self loadExistingVideos];
    
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(updateIPAddressLabel)
                                                 name:@"DeviceIPAddressDidChangeNotification"
                                               object:nil];
    
    [self updateIPAddressLabel];
    [self updateTableViewBackground];
    [self setupRefreshTimer];
}

- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    [self.navigationController setNavigationBarHidden:YES animated:NO];
}

- (void)updateIPAddressLabel {
    BOOL streamViaWifiEnabled = [[NSUserDefaults standardUserDefaults] boolForKey:@"stream_via_wifi_enabled"];
    NSString *ipAddress = [[NSUserDefaults standardUserDefaults] stringForKey:@"DeviceIPAddress"];

    if (streamViaWifiEnabled) {
        self.ipLabel.hidden = NO;
        self.ipLabel.text = (ipAddress.length > 0) ? [NSString stringWithFormat:NSLocalizedString(@"streaming_over_wifi", nil), ipAddress] : NSLocalizedString(@"waiting_for_ip", nil);
        self.mainStackViewTopToSafeAreaConstraint.active = NO;
        self.mainStackViewTopToIPLabelConstraint.active = YES;
    } else {
        self.ipLabel.hidden = YES;
        self.ipLabel.text = nil;
        self.mainStackViewTopToIPLabelConstraint.active = NO;
        self.mainStackViewTopToSafeAreaConstraint.active = YES;
    }

    [self.view layoutIfNeeded];
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [self.navigationController setNavigationBarHidden:NO animated:animated];
}

- (void)dealloc {
    [self.refreshTimer invalidate];
    self.refreshTimer = nil;
}

- (void)setupRefreshTimer {
    self.refreshTimer = [NSTimer scheduledTimerWithTimeInterval:10.0
                                                        target:self
                                                      selector:@selector(refreshTimerFired)
                                                      userInfo:nil
                                                       repeats:YES];
}

- (void)refreshTimerFired {
    if (!self.isLoadingVideos) {
        [self getEvents];
    }
}


- (void)updateTableViewBackground {
    if (self.videoFiles.count == 0) {
        UILabel *messageLabel = [[UILabel alloc] initWithFrame:self.eventsViewController.view.bounds];
        messageLabel.numberOfLines = 0;
        messageLabel.lineBreakMode = NSLineBreakByWordWrapping;

        // Create the full attributed message
        NSMutableAttributedString *message = [[NSMutableAttributedString alloc] init];

        // Centered title
        NSMutableParagraphStyle *centeredStyle = [[NSMutableParagraphStyle alloc] init];
        centeredStyle.alignment = NSTextAlignmentCenter;
        centeredStyle.paragraphSpacing = 12.0;

        NSDictionary *titleAttrs = @{
            NSFontAttributeName: [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline],
            NSForegroundColorAttributeName: [UIColor labelColor],
            NSParagraphStyleAttributeName: centeredStyle
        };
        [message appendAttributedString:[[NSAttributedString alloc] initWithString:NSLocalizedString(@"no_videos_available", "no videos available") attributes:titleAttrs]];

        // Left-aligned steps with hanging indent
        NSMutableParagraphStyle *stepsStyle = [[NSMutableParagraphStyle alloc] init];
        stepsStyle.alignment = NSTextAlignmentLeft;
        stepsStyle.headIndent = 20.0;
        stepsStyle.firstLineHeadIndent = 0.0;
        stepsStyle.paragraphSpacing = 8.0;
        stepsStyle.lineBreakMode = NSLineBreakByWordWrapping;

        NSDictionary *stepAttrs = @{
            NSFontAttributeName: [UIFont preferredFontForTextStyle:UIFontTextStyleBody],
            NSForegroundColorAttributeName: [UIColor secondaryLabelColor],
            NSParagraphStyleAttributeName: stepsStyle
        };

        NSString *stepsText = NSLocalizedString(@"steps_text_clips", "steps to setup clips");

        [message appendAttributedString:[[NSAttributedString alloc] initWithString:stepsText attributes:stepAttrs]];

        messageLabel.attributedText = message;
        [messageLabel sizeToFit];

        self.tableView.backgroundView = messageLabel;
        self.tableView.separatorStyle = UITableViewCellSeparatorStyleNone;
    } else {
        self.tableView.backgroundView = nil;
        self.tableView.separatorStyle = UITableViewCellSeparatorStyleSingleLine;
    }
}


- (void)setupEventsViewController {
    // Move all the existing table view setup to the events view controller
    self.tableView = [[UITableView alloc] initWithFrame:CGRectZero style:UITableViewStylePlain];
    self.tableView.delegate = self;
    self.tableView.dataSource = self;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor];
    [self.tableView registerClass:[VideoTableViewCell class] forCellReuseIdentifier:@"VideoCell"];
    self.tableView.separatorInset = UIEdgeInsetsMake(0, 16, 0, 16);

    if (@available(iOS 11.0, *)) {
        self.tableView.contentInsetAdjustmentBehavior = UIScrollViewContentInsetAdjustmentNever;
    }
    
    [self.eventsViewController.view addSubview:self.tableView];
    
    self.tableView.translatesAutoresizingMaskIntoConstraints = NO;
    [NSLayoutConstraint activateConstraints:@[
        [self.tableView.topAnchor constraintEqualToAnchor:self.eventsViewController.view.safeAreaLayoutGuide.topAnchor constant:0],
        [self.tableView.leadingAnchor constraintEqualToAnchor:self.eventsViewController.view.safeAreaLayoutGuide.leadingAnchor constant:0],
        [self.tableView.trailingAnchor constraintEqualToAnchor:self.eventsViewController.view.safeAreaLayoutGuide.trailingAnchor constant:0],
        [self.tableView.bottomAnchor constraintEqualToAnchor:self.eventsViewController.view.safeAreaLayoutGuide.bottomAnchor constant:0]
    ]];
    
    // Setup refresh control for events
    self.refreshControl = [[UIRefreshControl alloc] init];
    [self.refreshControl addTarget:self action:@selector(handleRefresh) forControlEvents:UIControlEventValueChanged];
    if (@available(iOS 10.0, *)) {
        self.tableView.refreshControl = self.refreshControl;
    } else {
        [self.tableView addSubview:self.refreshControl];
    }
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
                    [[NSFileManager defaultManager] moveItemAtURL:location toURL:destURL error:&moveError];
                }
                dispatch_semaphore_signal(semaphore);
            }] resume];
            
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            self.isLoadingVideos = NO;
            [self loadExistingVideos];
        });
    });
}

- (void)loadExistingVideos {
    NSError *error;
    NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:self.downloadDirectory error:&error];
    if (!self.loadedFilenames) self.loadedFilenames = [NSMutableSet set];
    for (NSString *file in contents) {
        NSString *filePath = [self.downloadDirectory stringByAppendingPathComponent:file];
        NSString *outputFilePath = filePath;
        if ([file hasSuffix:@".aes"]) {
            NSData *encryptedData = [NSData dataWithContentsOfURL:[NSURL fileURLWithPath:filePath] options:0 error:&error];
            if (!encryptedData) {
                [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to read the file: %@", error.localizedDescription]];
                return;
            }
            NSArray<NSString *> *storedKeys = [[SecretManager sharedManager] getAllDecryptionKeys];
            NSData *decryptedData = nil;
            for (NSString *key in storedKeys) {
                decryptedData = [[SecretManager sharedManager] decryptData:encryptedData withKey:key];
                if (decryptedData) break;
            }
            if(decryptedData){
                NSString *outputFilename = [file stringByDeletingPathExtension];
                outputFilePath = [self.downloadDirectory stringByAppendingPathComponent:outputFilename];
                NSError *writeError = nil;
                BOOL success = [decryptedData writeToFile:outputFilePath options:NSDataWritingAtomic error:&writeError];
                if (!success) {
                    [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to write decrypted file: %@", writeError.localizedDescription]];
                    return;
                }
                NSError *deleteError = nil;
                BOOL removed = [[NSFileManager defaultManager] removeItemAtPath:filePath error:&deleteError];
                if (!removed) {
                    [self showErrorAlertWithMessage:[NSString stringWithFormat:@"Failed to delete original encrypted file: %@", deleteError.localizedDescription]];
                    return;
                }
            }
            
        }
        [self addVideoFileAtPath:outputFilePath];
    }
}

- (void)addVideoFileAtPath:(NSString *)filePath {
    if (!filePath) return;
    NSString *filename = [filePath lastPathComponent];
    if ([self.loadedFilenames containsObject:filename]) return;
    [self.videoFiles addObject:filePath];
    [self.loadedFilenames addObject:filename];
    NSError *error = nil;
    NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:@"(\\d{4}-\\d{2}-\\d{2})_(\\d{2}-\\d{2}-\\d{2})"
                                                                           options:0
                                                                             error:&error];
    NSTextCheckingResult *match = [regex firstMatchInString:filename options:0 range:NSMakeRange(0, filename.length)];
    if (!match) return;
    NSString *dateStr = [filename substringWithRange:[match rangeAtIndex:1]];
    NSString *timeStr = [filename substringWithRange:[match rangeAtIndex:2]];
    NSString *fullTimestamp = [NSString stringWithFormat:@"%@_%@", dateStr, timeStr];
    NSDateFormatter *fullFormatter = [[NSDateFormatter alloc] init];
    [fullFormatter setDateFormat:@"yyyy-MM-dd_HH-mm-ss"];
    NSDate *timestampDate = [fullFormatter dateFromString:fullTimestamp];
    if (!timestampDate) return;
    NSDateFormatter *sectionFormatter = [[NSDateFormatter alloc] init];
    sectionFormatter.dateStyle = NSDateFormatterMediumStyle;
    sectionFormatter.timeStyle = NSDateFormatterNoStyle;
    NSString *sectionTitle = [sectionFormatter stringFromDate:timestampDate] ?: @"Unknown Date";
    if (!self.groupedVideos[sectionTitle]) {
        self.groupedVideos[sectionTitle] = [NSMutableArray array];
        [self.sectionTitles addObject:sectionTitle];
    }
    [self.groupedVideos[sectionTitle] addObject:filePath];
    [self.sectionTitles sortUsingComparator:^NSComparisonResult(NSString *obj1, NSString *obj2) {
        NSDate *d1 = [sectionFormatter dateFromString:obj1] ?: [NSDate distantPast];
        NSDate *d2 = [sectionFormatter dateFromString:obj2] ?: [NSDate distantPast];
        return [d2 compare:d1];
    }];
    for (NSString *section in self.groupedVideos) {
        [self.groupedVideos[section] sortUsingComparator:^NSComparisonResult(NSString *path1, NSString *path2) {
            NSString *f1 = [path1 lastPathComponent];
            NSString *f2 = [path2 lastPathComponent];
            NSTextCheckingResult *m1 = [regex firstMatchInString:f1 options:0 range:NSMakeRange(0, f1.length)];
            NSTextCheckingResult *m2 = [regex firstMatchInString:f2 options:0 range:NSMakeRange(0, f2.length)];
            NSString *ts1 = (m1 ?
                [NSString stringWithFormat:@"%@_%@",
                 [f1 substringWithRange:[m1 rangeAtIndex:1]],
                 [f1 substringWithRange:[m1 rangeAtIndex:2]]] :
                @"1970-01-01_00-00-00");

            NSString *ts2 = (m2 ?
                [NSString stringWithFormat:@"%@_%@",
                 [f2 substringWithRange:[m2 rangeAtIndex:1]],
                 [f2 substringWithRange:[m2 rangeAtIndex:2]]] :
                @"1970-01-01_00-00-00");

            NSDate *d1 = [fullFormatter dateFromString:ts1] ?: [NSDate distantPast];
            NSDate *d2 = [fullFormatter dateFromString:ts2] ?: [NSDate distantPast];

            return [d2 compare:d1];
        }];
    }
    [self.tableView reloadData];
    [self updateTableViewBackground];
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
        [self promptUserForKeyWithAESFileURL:[NSURL fileURLWithPath:filePath] encryptedData:encryptedData];
    }
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

- (void)promptUserForKeyWithAESFileURL:(NSURL *)aesFileURL encryptedData:(NSData *)encryptedData {
    [[SecretManager sharedManager] promptUserForKeyFromViewController:self completion:^(NSString *userProvidedKey) {
        if (userProvidedKey) {
            NSData *decryptedData = [[SecretManager sharedManager] decryptData:encryptedData withKey:userProvidedKey];
            if (decryptedData) {
                NSError *saveError = nil;
                NSString *fileName = [[aesFileURL lastPathComponent] stringByDeletingPathExtension];
                NSString *keyPrefix = [userProvidedKey substringToIndex:MIN(6, userProvidedKey.length)];
                NSString *keyIdentifier = [NSString stringWithFormat:@"decryption_key_%@_%@", fileName, keyPrefix];
                [[SecretManager sharedManager] saveDecryptionKey:userProvidedKey withIdentifier:keyIdentifier error:&saveError];
                NSURL *decryptedURL = [self handleDecryptedData:decryptedData fromURL:aesFileURL];
                if (decryptedURL) {
                    [self.videoFiles removeAllObjects];
                    [self.groupedVideos removeAllObjects];
                    [self.sectionTitles removeAllObjects];
                    [self.loadedFilenames removeAllObjects];
                    [self loadExistingVideos];
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
    
    [actionSheet addAction:[UIAlertAction actionWithTitle:NSLocalizedString(@"share", "share")
                                                    style:UIAlertActionStyleDefault
                                                  handler:^(UIAlertAction * _Nonnull action) {
        NSURL *videoURL = [NSURL fileURLWithPath:videoPath];
        UIActivityViewController *activityVC = [[UIActivityViewController alloc] initWithActivityItems:@[videoURL]
                                                                             applicationActivities:nil];
        [self presentViewController:activityVC animated:YES completion:nil];
    }]];
    
    [actionSheet addAction:[UIAlertAction actionWithTitle:NSLocalizedString(@"delete", "delete")
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
    
    [actionSheet addAction:[UIAlertAction actionWithTitle:NSLocalizedString(@"cancel", "cancel")
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

- (void)cameraTapped {
    ViewController *cameraVC = [[ViewController alloc] init];
    [self.navigationController pushViewController:cameraVC animated:YES];
}

- (void)settingsTapped {
    SettingsViewController *settingsVC = [[SettingsViewController alloc] init];
    [self.navigationController pushViewController:settingsVC animated:YES];
}

@end
