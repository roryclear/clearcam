#import "SettingsViewController.h"
#import "SettingsManager.h"
#import "SecretManager.h"
#import "StoreManager.h"
#import "NumberSelectionViewController.h"
#import "notification.h"
#import "FileServer.h"
#import "ScheduleManagementViewController.h"
#import <UserNotifications/UserNotifications.h>
#import <StoreKit/StoreKit.h>

@interface SettingsViewController () <UITableViewDelegate, UITableViewDataSource>

@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSString *selectedResolution;
@property (nonatomic, strong) NSString *selectedPresetKey; // For YOLO indexes key
@property (nonatomic, assign) BOOL isPresetsSectionExpanded; // Track if presets section is expanded
@property (nonatomic, assign) BOOL sendNotifEnabled;
@property (nonatomic, assign) BOOL receiveNotifEnabled; // New property for receiving notifications
@property (nonatomic, assign) BOOL useOwnServerEnabled;
@property (nonatomic, strong) NSString *notificationServerAddress; // Store the notification server address
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *notificationSchedules; // Array to store notification schedules
@property (nonatomic, assign) BOOL streamViaWiFiEnabled;
@property (nonatomic, strong) id ipAddressObserver;
@property (nonatomic, assign) NSInteger threshold; // New property for threshold
@property (nonatomic, assign) BOOL liveStreamInternetEnabled; // New property for live stream toggle
@property (nonatomic, strong) NSString *deviceName; // New property for device name
@property (nonatomic, assign) BOOL useOwnInferenceServerEnabled;
@property (nonatomic, strong) NSString *inferenceServerAddress;

@end

@implementation SettingsViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = NSLocalizedString(@"settings", @"Title for settings screen");
    [[StoreManager sharedInstance] verifySubscriptionWithCompletionIfSubbed:^(BOOL isActive, NSDate *expiryDate) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.tableView reloadData];
        });
    }];
    
    UIBarButtonItem *helpButton = [[UIBarButtonItem alloc] initWithTitle:NSLocalizedString(@"docs", nil) style:UIBarButtonItemStylePlain target:self action:@selector(helpButtonTapped)];
    [helpButton setTitleTextAttributes:@{NSForegroundColorAttributeName: [UIColor systemBlueColor]}
                              forState:UIControlStateNormal];
    self.navigationItem.rightBarButtonItem = helpButton;
    
    // Register for subscription status change notifications
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(subscriptionStatusDidChange:)
                                                 name:StoreManagerSubscriptionStatusDidChangeNotification
                                               object:nil];
    
    // Initialize properties from NSUserDefaults
    SettingsManager *settingsManager = [SettingsManager sharedManager];
    NSString *height = settingsManager.height ?: @"720";
    self.selectedResolution = [NSString stringWithFormat:@"%@p", height];
    self.selectedPresetKey = [[NSUserDefaults standardUserDefaults] stringForKey:@"yolo_preset_idx"] ?: @"all";
    
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    
    [self updateIPAddressDisplay]; // Set initial value
    [defaults addObserver:self
               forKeyPath:@"DeviceIPAddress"
                  options:NSKeyValueObservingOptionNew
                  context:nil];
    
    if ([defaults objectForKey:@"stream_via_wifi_enabled"] != nil) {
        self.streamViaWiFiEnabled = [defaults boolForKey:@"stream_via_wifi_enabled"];
    } else {
        self.streamViaWiFiEnabled = NO;
        [defaults setBool:NO forKey:@"stream_via_wifi_enabled"];
    }
    
    if ([defaults objectForKey:@"send_notif_enabled"] != nil) {
        self.sendNotifEnabled = [defaults boolForKey:@"send_notif_enabled"];
    } else {
        self.sendNotifEnabled = NO;
        [defaults setBool:NO forKey:@"send_notif_enabled"];
    }
    
    if ([defaults objectForKey:@"receive_notif_enabled"] != nil) {
        self.receiveNotifEnabled = [defaults boolForKey:@"receive_notif_enabled"];
    } else {
        self.receiveNotifEnabled = NO; // Off by default
        [defaults setBool:NO forKey:@"receive_notif_enabled"];
    }
    
    if ([defaults objectForKey:@"use_own_server_enabled"] != nil) {
        self.useOwnServerEnabled = [defaults boolForKey:@"use_own_server_enabled"];
    } else {
        self.useOwnServerEnabled = NO;
        [defaults setBool:NO forKey:@"use_own_server_enabled"];
    }
    
    if ([defaults objectForKey:@"threshold"] != nil) {
        self.threshold = [defaults integerForKey:@"threshold"];
    } else {
        self.threshold = 25; // Default value of 25%
        [defaults setInteger:25 forKey:@"threshold"];
    }
    
    if ([defaults objectForKey:@"live_stream_internet_enabled"] != nil) {
        self.liveStreamInternetEnabled = [defaults boolForKey:@"live_stream_internet_enabled"];
    } else {
        self.liveStreamInternetEnabled = NO;
        [defaults setBool:NO forKey:@"live_stream_internet_enabled"];
    }
    
    self.deviceName = [defaults stringForKey:@"device_name"] ?: NSLocalizedString(@"default_device_name", @"Default device name");
    [defaults setObject:self.deviceName forKey:@"device_name"];
    
    self.notificationSchedules = [[defaults arrayForKey:@"notification_schedules"] mutableCopy];
    if (!self.notificationSchedules) {
        self.notificationSchedules = [@[@{
            @"days": @[@"Mon", @"Tue", @"Wed", @"Thu", @"Fri", @"Sat", @"Sun"],
            @"startHour": @0,
            @"startMinute": @0,
            @"endHour": @23,
            @"endMinute": @59,
            @"enabled": @YES
        }] mutableCopy];
        [defaults setObject:self.notificationSchedules forKey:@"notification_schedules"];
    }
    
    // Initialize notification server settings
    self.notificationServerAddress = [defaults stringForKey:@"own_notification_server_address"] ?: @"http://192.168.1.1:8080";
    
    // Initialize inference server settings
    if ([defaults objectForKey:@"useOwnInferenceServerEnabled"] != nil) {
        self.useOwnInferenceServerEnabled = [defaults boolForKey:@"useOwnInferenceServerEnabled"];
    } else {
        self.useOwnInferenceServerEnabled = NO;
        [defaults setBool:NO forKey:@"useOwnInferenceServerEnabled"];
    }
    self.inferenceServerAddress = [defaults stringForKey:@"own_inference_server_address"] ?: @"http://192.168.1.1:6667";
    [defaults synchronize];
    
    // Create table view
    self.tableView = [[UITableView alloc] initWithFrame:CGRectZero style:UITableViewStyleInsetGrouped];
    self.tableView.delegate = self;
    self.tableView.dataSource = self;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor];
    self.tableView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.tableView];
    
    [NSLayoutConstraint activateConstraints:@[
        [self.tableView.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [self.tableView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.tableView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.tableView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor]
    ]];
    
    // Ensure switch reflects current permission state
    UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
    [center getNotificationSettingsWithCompletionHandler:^(UNNotificationSettings * _Nonnull settings) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (settings.authorizationStatus == UNAuthorizationStatusAuthorized) {
                self.receiveNotifEnabled = [defaults boolForKey:@"receive_notif_enabled"];
            } else {
                self.receiveNotifEnabled = NO;
                [defaults setBool:NO forKey:@"receive_notif_enabled"];
                [defaults synchronize];
            }
            [self.tableView reloadData];
        });
    }];
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    [[NSUserDefaults standardUserDefaults] removeObserver:self forKeyPath:@"DeviceIPAddress"];
}

- (void)helpButtonTapped {
    NSURL *helpURL = [NSURL URLWithString:@"https://github.com/roryclear/clearcam/?tab=readme-ov-file#clearcam"];
    if ([[UIApplication sharedApplication] canOpenURL:helpURL]) {
        if (@available(iOS 10.0, *)) {
            [[UIApplication sharedApplication] openURL:helpURL options:@{} completionHandler:nil];
        } else {
            [[UIApplication sharedApplication] openURL:helpURL];
        }
    }
}

- (void)updateIPAddressDisplay {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    NSString *ipAddress = [defaults stringForKey:@"DeviceIPAddress"];
    if (ipAddress && ipAddress.length > 0) {
        self.notificationServerAddress = [NSString stringWithFormat:@"http://%@", ipAddress];
    } else {
        self.notificationServerAddress = NSLocalizedString(@"waiting_for_ip", @"Placeholder when IP address is not available");
    }
}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary *)change
                       context:(void *)context {
    if ([keyPath isEqualToString:@"DeviceIPAddress"]) {
        [self updateIPAddressDisplay];
        [self.tableView reloadData];
    }
}

- (void)streamViaWiFiSwitchToggled:(UISwitch *)sender {
    self.streamViaWiFiEnabled = sender.on;
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    [defaults setBool:self.streamViaWiFiEnabled forKey:@"stream_via_wifi_enabled"];
    [defaults synchronize];
}

- (void)liveStreamInternetSwitchToggled:(UISwitch *)sender {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    BOOL isPremium = [defaults boolForKey:@"isSubscribed"];
    
    if (isPremium) {
        if (sender.on) {
            NSString *password = [self retrievePasswordFromSecretsManager];
            if (!password) {
                [self promptForPasswordWithCompletion:^(BOOL success) {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        if (success) {
                            self.liveStreamInternetEnabled = YES;
                            [defaults setBool:YES forKey:@"live_stream_internet_enabled"];
                        } else {
                            self.liveStreamInternetEnabled = NO;
                            [defaults setBool:NO forKey:@"live_stream_internet_enabled"];
                            sender.on = NO;
                        }
                        [defaults synchronize];
                        [self.tableView reloadData];
                    });
                }];
            } else {
                self.liveStreamInternetEnabled = sender.on;
                [defaults setBool:self.liveStreamInternetEnabled forKey:@"live_stream_internet_enabled"];
                [defaults synchronize];
            }
        } else {
            self.liveStreamInternetEnabled = sender.on;
            [defaults setBool:self.liveStreamInternetEnabled forKey:@"live_stream_internet_enabled"];
            [defaults synchronize];
        }
    } else {
        sender.on = NO;
        self.liveStreamInternetEnabled = NO;
        [defaults setBool:NO forKey:@"live_stream_internet_enabled"];
        [defaults synchronize];
        [self showUpgradePopup];
    }
}

- (void)useOwnInferenceServerSwitchToggled:(UISwitch *)sender {
    self.useOwnInferenceServerEnabled = sender.isOn;
    [[NSUserDefaults standardUserDefaults] setBool:sender.isOn forKey:@"useOwnInferenceServerEnabled"];
    [[NSUserDefaults standardUserDefaults] synchronize];
    
    if (sender.isOn) {
        [self showInferenceServerAddressInputDialog];
    }
    
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:0] withRowAnimation:UITableViewRowAnimationAutomatic];
}

- (void)receiveNotifSwitchToggled:(UISwitch *)sender {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    BOOL hasRequestedPermission = [defaults boolForKey:@"hasRequestedNotificationPermission"];
    
    self.receiveNotifEnabled = sender.on;
    [defaults setBool:self.receiveNotifEnabled forKey:@"receive_notif_enabled"];
    [defaults synchronize];

    if (sender.on) {
        UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
        [center getNotificationSettingsWithCompletionHandler:^(UNNotificationSettings * _Nonnull settings) {
            dispatch_async(dispatch_get_main_queue(), ^{
                if (settings.authorizationStatus == UNAuthorizationStatusAuthorized) {
                    [FileServer sendDeviceTokenToServer]; // Send token when toggled ON
                } else if (!hasRequestedPermission) {
                    [center requestAuthorizationWithOptions:(UNAuthorizationOptionAlert | UNAuthorizationOptionSound | UNAuthorizationOptionBadge)
                                          completionHandler:^(BOOL granted, NSError * _Nullable error) {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            if (granted) {
                                [[UIApplication sharedApplication] registerForRemoteNotifications];
                                [defaults setBool:YES forKey:@"hasRequestedNotificationPermission"];
                                [defaults synchronize];
                                
                                [FileServer sendDeviceTokenToServer]; // Send token after granting permission
                            } else {
                                self.receiveNotifEnabled = NO;
                                [defaults setBool:NO forKey:@"receive_notif_enabled"];
                                [defaults synchronize];
                                sender.on = NO;
                                
                                UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"permission_denied", @"Title for permission denied alert")
                                                                                              message:NSLocalizedString(@"notification_permission_denied", @"Message when notification permission is denied")
                                                                                       preferredStyle:UIAlertControllerStyleAlert];
                                [alert addAction:[UIAlertAction actionWithTitle:NSLocalizedString(@"ok", @"OK button") style:UIAlertActionStyleDefault handler:nil]];
                                [self presentViewController:alert animated:YES completion:nil];
                            }
                        });
                    }];
                } else {
                    self.receiveNotifEnabled = NO;
                    [defaults setBool:NO forKey:@"receive_notif_enabled"];
                    [defaults synchronize];
                    sender.on = NO;
                    
                    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"enable_in_settings", @"Title for enabling notifications in settings")
                                                                                  message:NSLocalizedString(@"notifications_disabled", @"Message when notifications are disabled")
                                                                           preferredStyle:UIAlertControllerStyleAlert];
                    [alert addAction:[UIAlertAction actionWithTitle:NSLocalizedString(@"ok", @"OK button") style:UIAlertActionStyleDefault handler:nil]];
                    [self presentViewController:alert animated:YES completion:nil];
                }
            });
        }];
    } else {
        // Clear any pending notifications when turning off
        UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
        [center removeAllPendingNotificationRequests];
        [center removeAllDeliveredNotifications];
        
        // Delete device token from server
        [self deleteDeviceTokenFromServer];
    }
}

- (void)deleteDeviceTokenFromServer {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    NSString *deviceToken = [defaults stringForKey:@"device_token"];
    
    if (!deviceToken || deviceToken.length == 0) return;
    
    // Retrieve session token from Keychain
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken || sessionToken.length == 0) return;
    NSURL *url = [NSURL URLWithString:@"https://rors.ai/delete_device"];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    request.HTTPMethod = @"DELETE";
    [request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
    
    NSDictionary *body = @{
        @"device_token": deviceToken,
        @"session_token": sessionToken
    };
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:body options:0 error:nil];
    request.HTTPBody = jsonData;
    
    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithRequest:request
                                                                 completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) return;
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
    }];
    
    [task resume];
}

- (void)subscriptionStatusDidChange:(NSNotification *)notification {
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.tableView reloadData];
    });
}

#pragma mark - Orientation Control

- (BOOL)shouldAutorotate {
    return NO;
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
    return UIInterfaceOrientationMaskPortrait;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
    return UIInterfaceOrientationPortrait;
}

#pragma mark - UITableView DataSource

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return 6;
}

- (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    if (section == 0) {
        return NSLocalizedString(@"camera_settings", @"Header for camera settings section");
    } else if (section == 1) {
        return NSLocalizedString(@"live_stream_settings", @"Header for live stream settings section");
    } else if (section == 2) {
        return NSLocalizedString(@"viewer_settings", @"Header for viewer settings section");
    } else if (section == 3) {
        return NSLocalizedString(@"experimental_features", @"Header for advanced settings section");
    } else if (section == 4) {
        BOOL isSubscribed = [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"];
        return isSubscribed ? NSLocalizedString(@"subscription", @"Header for subscription section when subscribed") : NSLocalizedString(@"upgrade_to_premium", @"Header for subscription section when not subscribed");
    } else if (section == 5) {
        return nil; // No header for Terms and Privacy section
    }
    return nil;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"SettingsCell"];
    if (!cell) {
        cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleSubtitle reuseIdentifier:@"SettingsCell"];
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
    }

    cell.backgroundColor = [UIColor secondarySystemBackgroundColor];
    cell.textLabel.textColor = [UIColor labelColor];
    cell.detailTextLabel.textColor = [UIColor secondaryLabelColor];
    cell.textLabel.text = nil;
    cell.detailTextLabel.text = nil;
    cell.imageView.image = nil;
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
    cell.accessoryView = nil;

    BOOL isPremium = [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"];

    if (indexPath.section == 0) { // Camera Settings
        if (indexPath.row == 0) {
            cell.textLabel.text = NSLocalizedString(@"stream_via_wifi", @"Label for stream via Wi-Fi setting");
            NSString *ipAddress = [[NSUserDefaults standardUserDefaults] stringForKey:@"DeviceIPAddress"];
            cell.detailTextLabel.text = ipAddress && ipAddress.length > 0 ?
                                       [NSString stringWithFormat:@"http://%@", ipAddress] :
                                       NSLocalizedString(@"waiting_for_ip", @"Placeholder when IP address is not available");
            cell.accessoryType = UITableViewCellAccessoryNone;
            UISwitch *wifiSwitch = [[UISwitch alloc] init];
            wifiSwitch.on = self.streamViaWiFiEnabled;
            [wifiSwitch addTarget:self action:@selector(streamViaWiFiSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = wifiSwitch;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 1) {
            cell.textLabel.text = NSLocalizedString(@"resolution", @"Label for resolution setting");
            cell.detailTextLabel.text = self.selectedResolution;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 2) {
            cell.textLabel.text = NSLocalizedString(@"detect_objects", @"Label for detect objects setting");
            cell.detailTextLabel.text = self.selectedPresetKey;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 3) {
            cell.textLabel.text = NSLocalizedString(@"manage_detection_presets", @"Label for manage detection presets setting");
            cell.detailTextLabel.text = nil;
            cell.accessoryType = self.isPresetsSectionExpanded ? UITableViewCellAccessoryNone : UITableViewCellAccessoryDisclosureIndicator;
            cell.userInteractionEnabled = YES;
        } else if (self.isPresetsSectionExpanded && indexPath.row >= 4 && indexPath.row < 4 + [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys].count + 1) {
            NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
            if (indexPath.row - 4 < presetKeys.count) {
                NSString *presetKey = presetKeys[indexPath.row - 4];
                cell.textLabel.text = presetKey;
                cell.detailTextLabel.text = nil;
                cell.imageView.image = nil;
                cell.userInteractionEnabled = YES;
            } else {
                cell.textLabel.text = nil;
                cell.detailTextLabel.text = nil;
                cell.imageView.image = [UIImage systemImageNamed:@"plus.circle.fill"];
                cell.imageView.tintColor = [UIColor systemGreenColor];
                cell.accessoryType = UITableViewCellAccessoryNone;
                cell.userInteractionEnabled = YES;
            }
        } else {
            NSInteger presetOffset = self.isPresetsSectionExpanded ? [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys].count + 1 : 0;
            if (indexPath.row == 4 + presetOffset) {
                cell.textLabel.text = NSLocalizedString(@"detection_certainty_threshold", @"Label for detection certainty threshold setting");
                cell.detailTextLabel.text = [NSString stringWithFormat:@"%ld%%", (long)self.threshold];
                cell.userInteractionEnabled = YES;
            } else if (indexPath.row == 5 + presetOffset) {
                cell.textLabel.text = NSLocalizedString(@"send_videos_on_detection", @"Label for send videos on detection setting");
                cell.detailTextLabel.text = NSLocalizedString(@"send_videos_description", @"Description for send videos on detection setting");
                cell.accessoryType = UITableViewCellAccessoryNone;
                UISwitch *sendNotifSwitch = [[UISwitch alloc] init];
                sendNotifSwitch.on = self.sendNotifEnabled;
                [sendNotifSwitch addTarget:self action:@selector(sendNotifSwitchToggled:) forControlEvents:UIControlEventValueChanged];
                cell.accessoryView = sendNotifSwitch;
                BOOL isEnabled = isPremium || self.useOwnServerEnabled;
                sendNotifSwitch.enabled = isEnabled;
                cell.textLabel.textColor = isEnabled ? [UIColor labelColor] : [UIColor grayColor];
                cell.userInteractionEnabled = YES;
            } else if (indexPath.row == 6 + presetOffset) {
                cell.textLabel.text = NSLocalizedString(@"change_encryption_password", @"Label for change encryption password setting");
                cell.detailTextLabel.text = nil;
                cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
                cell.textLabel.textColor = isPremium ? [UIColor labelColor] : [UIColor grayColor];
                cell.userInteractionEnabled = YES;
            } else if (indexPath.row == 7 + presetOffset) {
                cell.textLabel.text = NSLocalizedString(@"manage_detection_schedules", @"Label for manage detection schedules setting");
                cell.detailTextLabel.text = NSLocalizedString(@"manage_detection_schedules_description", @"Description for manage detection schedules setting");
                cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
                cell.userInteractionEnabled = YES;
            }
        }
    } else if (indexPath.section == 1) { // Live Stream Settings
        if (indexPath.row == 0) {
            cell.textLabel.text = NSLocalizedString(@"live_stream_network", @"Label for live stream over network setting");
            cell.accessoryType = UITableViewCellAccessoryNone;
            UISwitch *liveStreamSwitch = [[UISwitch alloc] init];
            liveStreamSwitch.on = isPremium ? self.liveStreamInternetEnabled : NO;
            [liveStreamSwitch addTarget:self action:@selector(liveStreamInternetSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = liveStreamSwitch;
            liveStreamSwitch.enabled = isPremium;
            cell.textLabel.textColor = isPremium ? [UIColor labelColor] : [UIColor grayColor];
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 1) {
            cell.textLabel.text = NSLocalizedString(@"device_name", @"Label for device name setting");
            cell.detailTextLabel.text = self.deviceName;
            cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
            cell.textLabel.textColor = isPremium ? [UIColor labelColor] : [UIColor grayColor];
            cell.userInteractionEnabled = YES;
        }
    } else if (indexPath.section == 2) { // Viewer Settings
        if (indexPath.row == 0) {
            cell.textLabel.text = NSLocalizedString(@"receive_notifications", @"Label for receive notifications setting");
            cell.detailTextLabel.text = NSLocalizedString(@"receive_notifications_description", @"Description for receive notifications setting");
            cell.accessoryType = UITableViewCellAccessoryNone;
            UISwitch *receiveNotifSwitch = [[UISwitch alloc] init];
            receiveNotifSwitch.on = isPremium ? self.receiveNotifEnabled : NO;
            [receiveNotifSwitch addTarget:self action:@selector(receiveNotifSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = receiveNotifSwitch;
            receiveNotifSwitch.enabled = isPremium;
            cell.textLabel.textColor = isPremium ? [UIColor labelColor] : [UIColor grayColor];
            cell.userInteractionEnabled = YES;
        }
    } else if (indexPath.section == 3) { // Advanced Settings
        if (indexPath.row == 0) {
            cell.textLabel.text = NSLocalizedString(@"use_own_notification_server", @"Label for use own notification server setting");
            cell.detailTextLabel.text = self.useOwnServerEnabled ?
                (self.notificationServerAddress ?: NSLocalizedString(@"enter_notification_server_address", @"Placeholder for notification server address")) :
                NSLocalizedString(@"use_own_notification_server_description", @"Description for use own notification server setting");
            cell.accessoryType = UITableViewCellAccessoryNone;
            UISwitch *useOwnNotificationServerSwitch = [[UISwitch alloc] init];
            useOwnNotificationServerSwitch.on = self.useOwnServerEnabled;
            [useOwnNotificationServerSwitch addTarget:self action:@selector(useOwnnotificationServerSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = useOwnNotificationServerSwitch;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 1) {
            cell.textLabel.text = NSLocalizedString(@"use_own_inference_server", @"Label for use own inference server setting");
            cell.detailTextLabel.text = self.useOwnInferenceServerEnabled ?
                (self.inferenceServerAddress ?: NSLocalizedString(@"enter_inference_server_address", @"Placeholder for inference server address")) :
                NSLocalizedString(@"use_own_inference_server_description", @"Description for use own inference server setting");
            cell.accessoryType = UITableViewCellAccessoryNone;
            UISwitch *useOwnInferenceServerSwitch = [[UISwitch alloc] init];
            useOwnInferenceServerSwitch.on = self.useOwnInferenceServerEnabled;
            [useOwnInferenceServerSwitch addTarget:self action:@selector(useOwnInferenceServerSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = useOwnInferenceServerSwitch;
            cell.userInteractionEnabled = YES;
        }
    } else if (indexPath.section == 4) { // Upgrade to Premium / Subscription
        if (!isPremium && indexPath.row == 0) {
            cell.textLabel.text = NSLocalizedString(@"upgrade_to_premium", @"Label for upgrade to premium button");
            cell.textLabel.textColor = [UIColor systemBlueColor];
            cell.textLabel.textAlignment = NSTextAlignmentCenter;
            cell.accessoryType = UITableViewCellAccessoryNone;
            cell.userInteractionEnabled = YES;
        } else if ((isPremium && indexPath.row == 0) || (!isPremium && indexPath.row == 1)) {
            cell.textLabel.text = NSLocalizedString(@"restore_purchases", @"Label for restore purchases button");
            cell.textLabel.textColor = [UIColor systemBlueColor];
            cell.textLabel.textAlignment = NSTextAlignmentCenter;
            cell.accessoryType = UITableViewCellAccessoryNone;
            cell.userInteractionEnabled = YES;
        }
    } else if (indexPath.section == 5) { // Terms and Privacy
        if (indexPath.row == 0) {
            cell.textLabel.text = NSLocalizedString(@"terms_of_use", @"Label for terms of use link");
            cell.textLabel.textColor = [UIColor systemBlueColor];
            cell.textLabel.textAlignment = NSTextAlignmentCenter;
            cell.accessoryType = UITableViewCellAccessoryNone;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 1) {
            cell.textLabel.text = NSLocalizedString(@"privacy_policy", @"Label for privacy policy link");
            cell.textLabel.textColor = [UIColor systemBlueColor];
            cell.textLabel.textAlignment = NSTextAlignmentCenter;
            cell.accessoryType = UITableViewCellAccessoryNone;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 2) {
            cell.textLabel.text = NSLocalizedString(@"delete_encryption_keys", @"Label for delete encryption keys button");
            cell.textLabel.textColor = [UIColor systemRedColor];
            cell.textLabel.textAlignment = NSTextAlignmentCenter;
            cell.accessoryType = UITableViewCellAccessoryNone;
            cell.userInteractionEnabled = YES;
        }
    }

    return cell;
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    if (section == 0) { // Camera Settings
        NSInteger baseRows = 8; // Up to "Manage Detection Schedules"
        if (self.isPresetsSectionExpanded) {
            NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
            baseRows += presetKeys.count + 1; // Preset rows + "Add Preset"
        }
        return baseRows;
    } else if (section == 1) { // Live Stream Settings
        return 2; // Live Stream over the internet and Device name
    } else if (section == 2) { // Viewer Settings
        return 1; // Just Receive Notifications
    } else if (section == 3) { // Advanced Settings
        return 2; // Use Own Notification Server, Use Own Inference Server
    } else if (section == 4) { // Upgrade to Premium / Subscription
        BOOL isSubscribed = [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"];
        return isSubscribed ? 1 : 2; // 1 row for "Restore Purchases" if subscribed, 2 rows ("Upgrade" + "Restore") if not
    } else if (section == 5) { // Terms and Privacy
        return 3; // Terms of Use, Privacy Policy, Delete Encryption Keys
    }
    return 0;
}

- (void)showInferenceServerAddressInputDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"enter_inference_server_address_title", @"Title for inference server address input dialog")
                                                                   message:NSLocalizedString(@"enter_inference_server_address_message", @"Message for inference server address input dialog")
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = NSLocalizedString(@"server_address_label", @"Placeholder for server address input");
        textField.text = self.inferenceServerAddress;
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"save", @"Save button")
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *address = alert.textFields.firstObject.text;
        if (address.length > 0) {
            self.inferenceServerAddress = address;
            NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
            [defaults setObject:address forKey:@"own_inference_server_address"];
            [defaults synchronize];
            [self.tableView reloadData];
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"cancel", @"Cancel button")
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];

    [self presentViewController:alert animated:YES completion:nil];
}

- (void)showThresholdInputDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"set_threshold_title", @"Title for threshold input dialog")
                                                                   message:NSLocalizedString(@"set_threshold_message", @"Message for threshold input dialog")
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = NSLocalizedString(@"threshold_placeholder", @"Placeholder for threshold input");
        textField.keyboardType = UIKeyboardTypeNumberPad;
        textField.text = [NSString stringWithFormat:@"%ld", (long)self.threshold];
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"save", @"Save button")
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *thresholdText = alert.textFields.firstObject.text;
        NSInteger newThreshold = [thresholdText integerValue];
        if (newThreshold >= 1 && newThreshold <= 100) {
            self.threshold = newThreshold;
            NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
            [defaults setInteger:self.threshold forKey:@"threshold"];
            [defaults synchronize];
            [self.tableView reloadData];
        } else {
            UIAlertController *errorAlert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"invalid_threshold_title", @"Title for invalid threshold error")
                                                                               message:NSLocalizedString(@"invalid_threshold_message", @"Message for invalid threshold error")
                                                                        preferredStyle:UIAlertControllerStyleAlert];
            [errorAlert addAction:[UIAlertAction actionWithTitle:NSLocalizedString(@"ok", @"OK button") style:UIAlertActionStyleDefault handler:nil]];
            [self presentViewController:errorAlert animated:YES completion:nil];
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"cancel", @"Cancel button")
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];
    
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)showDeviceNameInputDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"set_device_name_title", @"Title for device name input dialog")
                                                                   message:NSLocalizedString(@"set_device_name_message", @"Message for device name input dialog")
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = NSLocalizedString(@"device_name_placeholder", @"Placeholder for device name input");
        textField.text = self.deviceName;
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"save", @"Save button")
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *name = alert.textFields.firstObject.text;
        if (name.length > 0) {
            self.deviceName = name;
            NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
            [defaults setObject:name forKey:@"device_name"];
            [defaults synchronize];
            [self.tableView reloadData];
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"cancel", @"Cancel button")
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];
    
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)showDeleteEncryptionKeysPrompt {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"delete_encryption_keys_title", @"Title for delete encryption keys confirmation")
                                                                   message:NSLocalizedString(@"delete_encryption_keys_message", @"Message for delete encryption keys confirmation")
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    UIAlertAction *deleteAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"delete", @"Delete button")
                                                           style:UIAlertActionStyleDestructive
                                                         handler:^(UIAlertAction * _Nonnull action) {
        // Delete encryption keys
        NSError *error = nil;
        [[SecretManager sharedManager] deleteAllKeysWithError:&error];
        NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
        
        // Turn off "Receive Notifications on This Device" unless "Use Own Server" is on
        if (!self.useOwnServerEnabled) {
            self.sendNotifEnabled = NO;
            [defaults setBool:NO forKey:@"send_notif_enabled"];
        }
        
        // Turn off "Live Stream over the Internet"
        self.liveStreamInternetEnabled = NO;
        [defaults setBool:NO forKey:@"live_stream_internet_enabled"];
        
        [defaults synchronize];
        [self.tableView reloadData];
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"cancel", @"Cancel button")
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    
    [alert addAction:deleteAction];
    [alert addAction:cancelAction];
    
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    BOOL isPremium = [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"];
    
    if (indexPath.section == 0) { // Camera Settings
        if (indexPath.row == 0) {
            // Stream via Wi-Fi - handled by switch
        } else if (indexPath.row == 1) {
            [self showResolutionPicker];
        } else if (indexPath.row == 2) {
            [self showYoloIndexesPicker];
        } else if (indexPath.row == 3) {
            self.isPresetsSectionExpanded = !self.isPresetsSectionExpanded;
            [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:0] withRowAnimation:UITableViewRowAnimationAutomatic];
        } else if (self.isPresetsSectionExpanded && indexPath.row >= 4 && indexPath.row < 4 + [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys].count + 1) {
            NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
            if (indexPath.row - 4 < presetKeys.count) {
                NSString *presetKey = presetKeys[indexPath.row - 4];
                NSArray *selectedIndexes = [[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"][presetKey];
                [self showNumberSelectionForPreset:presetKey selectedIndexes:selectedIndexes];
            } else {
                [self showAddPresetDialog];
            }
        } else {
            NSInteger offset = self.isPresetsSectionExpanded ? [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys].count + 1 : 0;
            if (indexPath.row == 4 + offset) {
                [self showThresholdInputDialog];
            } else if (indexPath.row == 5 + offset) {
                if (!isPremium) {
                    [self showUpgradePopup];
                }
            } else if (indexPath.row == 6 + offset) {
                if (isPremium) {
                    [self promptForPasswordWithCompletion:^(BOOL success) {
                    }];
                } else {
                    [self showUpgradePopup];
                }
            } else if (indexPath.row == 7 + offset) {
                ScheduleManagementViewController *scheduleVC = [[ScheduleManagementViewController alloc] init];
                scheduleVC.notificationSchedules = [self.notificationSchedules mutableCopy];
                scheduleVC.completionHandler = ^(NSArray<NSDictionary *> *schedules) {
                    self.notificationSchedules = [schedules mutableCopy];
                    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
                    [defaults setObject:self.notificationSchedules forKey:@"notification_schedules"];
                    [defaults synchronize];
                    [self.tableView reloadData];
                };
                [self.navigationController pushViewController:scheduleVC animated:YES];
            }
        }
    } else if (indexPath.section == 1) { // Live Stream Settings
        if (indexPath.row == 0) {
            if (!isPremium) {
                [self showUpgradePopup];
            } // Switch handles toggle when premium
        } else if (indexPath.row == 1) {
            if (isPremium) {
                [self showDeviceNameInputDialog];
            } else {
                [self showUpgradePopup];
            }
        }
    } else if (indexPath.section == 2) { // Viewer Settings
        if (indexPath.row == 0) {
            if (!isPremium) {
                [self showUpgradePopup];
            } // Switch handles toggle when premium
        }
    } else if (indexPath.section == 3) { // Advanced Settings
        // Rows 0 and 1 (server switches) are handled by their toggles, so do nothing on tap
    } else if (indexPath.section == 4) { // Upgrade to Premium / Subscription
        if (!isPremium && indexPath.row == 0) {
            [self showUpgradePopup];
        } else if ((isPremium && indexPath.row == 0) || (!isPremium && indexPath.row == 1)) {
            UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"restoring_purchases_title", @"Title for restoring purchases alert")
                                                                          message:NSLocalizedString(@"restoring_purchases_message", @"Message for restoring purchases alert")
                                                                   preferredStyle:UIAlertControllerStyleAlert];
            [self presentViewController:alert animated:YES completion:nil];
            
            [[StoreManager sharedInstance] restorePurchases];
            
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2.0 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                [self dismissViewControllerAnimated:YES completion:nil];
            });
        }
    } else if (indexPath.section == 5) { // Terms and Privacy
        if (indexPath.row == 0) { // Terms of Use
            [[UIApplication sharedApplication] openURL:[NSURL URLWithString:@"https://www.apple.com/legal/internet-services/itunes/dev/stdeula/"] options:@{} completionHandler:nil];
        } else if (indexPath.row == 1) { // Privacy Policy
            [[UIApplication sharedApplication] openURL:[NSURL URLWithString:@"https://www.rors.ai/privacy"] options:@{} completionHandler:nil];
        } else if (indexPath.row == 2) { // Delete Encryption Keys
            [self showDeleteEncryptionKeysPrompt];
        }
    }
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

- (void)showUpgradePopup {
    [[StoreManager sharedInstance] getPremiumProductInfo:^(SKProduct * _Nullable product, NSError * _Nullable error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (!product) {
                NSLog(@"Error fetching product info: %@", error.localizedDescription);
                return;
            }

            // Format price
            NSNumberFormatter *formatter = [[NSNumberFormatter alloc] init];
            [formatter setNumberStyle:NSNumberFormatterCurrencyStyle];
            [formatter setLocale:product.priceLocale];
            NSString *localizedPrice = [formatter stringFromNumber:product.price] ?: NSLocalizedString(@"price_unknown", @"Fallback for unknown price");

            // Detect dark mode
            BOOL isDarkMode = (self.traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark);
            UIColor *cardBackground = isDarkMode ? [UIColor colorWithWhite:0.1 alpha:1.0] : [UIColor colorWithWhite:1.0 alpha:1.0];
            UIColor *textColor = isDarkMode ? UIColor.whiteColor : UIColor.blackColor;

            // Overlay
            UIView *overlay = [[UIView alloc] initWithFrame:self.view.bounds];
            overlay.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.7];
            overlay.tag = 999;
            [self.view addSubview:overlay];

            // Card
            UIView *card = [[UIView alloc] init];
            card.translatesAutoresizingMaskIntoConstraints = NO;
            card.backgroundColor = cardBackground;
            card.layer.cornerRadius = 20;
            card.clipsToBounds = YES;
            [overlay addSubview:card];

            // Title
            UILabel *title = [[UILabel alloc] init];
            title.translatesAutoresizingMaskIntoConstraints = NO;
            title.text = NSLocalizedString(@"get_premium_title", @"Title for premium upgrade popup");
            title.textColor = [UIColor colorWithRed:1.0 green:0.84 blue:0 alpha:1.0]; // gold
            title.font = [UIFont boldSystemFontOfSize:24];
            title.textAlignment = NSTextAlignmentCenter;

            // Features
            UIStackView *featureStack = [[UIStackView alloc] init];
            featureStack.translatesAutoresizingMaskIntoConstraints = NO;
            featureStack.axis = UILayoutConstraintAxisVertical;
            featureStack.spacing = 8;

            NSArray *features = @[
                NSLocalizedString(@"premium_feature_1", @"Feature 1 description for premium"),
                NSLocalizedString(@"premium_feature_2", @"Feature 2 description for premium"),
                NSLocalizedString(@"premium_feature_3", @"Feature 3 description for premium"),
                NSLocalizedString(@"premium_feature_4", @"Feature 4 description for premium")
            ];

            for (NSString *item in features) {
                UILabel *label = [[UILabel alloc] init];
                label.text = [NSString stringWithFormat:@"• %@", item];
                label.font = [UIFont systemFontOfSize:16 weight:UIFontWeightSemibold];
                label.textColor = textColor;
                label.numberOfLines = 0;
                [featureStack addArrangedSubview:label];
            }

            // Upgrade button
            UIButton *upgradeBtn = [UIButton buttonWithType:UIButtonTypeSystem];
            upgradeBtn.translatesAutoresizingMaskIntoConstraints = NO;
            NSString *upgradeTitle = [NSString stringWithFormat:NSLocalizedString(@"upgrade_button", @"Upgrade button text with price"), localizedPrice];
            [upgradeBtn setTitle:upgradeTitle forState:UIControlStateNormal];
            [upgradeBtn setTitleColor:UIColor.blackColor forState:UIControlStateNormal];
            upgradeBtn.backgroundColor = [UIColor colorWithRed:1.0 green:0.84 blue:0 alpha:1.0];
            upgradeBtn.titleLabel.font = [UIFont boldSystemFontOfSize:17];
            upgradeBtn.layer.cornerRadius = 12;
            [upgradeBtn addTarget:self action:@selector(handleUpgradeTap) forControlEvents:UIControlEventTouchUpInside];

            // Cancel button
            UIButton *cancelBtn = [UIButton buttonWithType:UIButtonTypeSystem];
            cancelBtn.translatesAutoresizingMaskIntoConstraints = NO;
            [cancelBtn setTitle:NSLocalizedString(@"not_now", @"Not now button") forState:UIControlStateNormal];
            [cancelBtn setTitleColor:textColor forState:UIControlStateNormal];
            cancelBtn.titleLabel.font = [UIFont systemFontOfSize:15];
            [cancelBtn addTarget:self action:@selector(dismissUpgradePopup) forControlEvents:UIControlEventTouchUpInside];

            // Disclaimer label
            UILabel *disclaimer = [[UILabel alloc] init];
            disclaimer.translatesAutoresizingMaskIntoConstraints = NO;
            disclaimer.text = NSLocalizedString(@"premium_disclaimer", @"Disclaimer for premium subscription limits");
            disclaimer.font = [UIFont systemFontOfSize:13];
            disclaimer.textColor = [textColor colorWithAlphaComponent:0.6];
            disclaimer.numberOfLines = 0;
            disclaimer.textAlignment = NSTextAlignmentCenter;

            // Add all to card
            [card addSubview:title];
            [card addSubview:featureStack];
            [card addSubview:upgradeBtn];
            [card addSubview:cancelBtn];
            [card addSubview:disclaimer];

            // Constraints
            [NSLayoutConstraint activateConstraints:@[
                [card.centerXAnchor constraintEqualToAnchor:overlay.centerXAnchor],
                [card.centerYAnchor constraintEqualToAnchor:overlay.centerYAnchor],
                [card.widthAnchor constraintEqualToConstant:320],

                [title.topAnchor constraintEqualToAnchor:card.topAnchor constant:24],
                [title.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:20],
                [title.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-20],

                [featureStack.topAnchor constraintEqualToAnchor:title.bottomAnchor constant:20],
                [featureStack.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:20],
                [featureStack.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-20],

                [upgradeBtn.topAnchor constraintEqualToAnchor:featureStack.bottomAnchor constant:24],
                [upgradeBtn.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:20],
                [upgradeBtn.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-20],
                [upgradeBtn.heightAnchor constraintEqualToConstant:48],

                [cancelBtn.topAnchor constraintEqualToAnchor:upgradeBtn.bottomAnchor constant:16],
                [cancelBtn.centerXAnchor constraintEqualToAnchor:card.centerXAnchor],

                [disclaimer.topAnchor constraintEqualToAnchor:cancelBtn.bottomAnchor constant:14],
                [disclaimer.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:20],
                [disclaimer.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-20],
                [disclaimer.bottomAnchor constraintEqualToAnchor:card.bottomAnchor constant:-20]
            ]];
        });
    }];
}

- (void)handleUpgradeTap {
    // Create and show a loading spinner
    UIActivityIndicatorView *spinner = [[UIActivityIndicatorView alloc] initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleLarge];
    spinner.translatesAutoresizingMaskIntoConstraints = NO;
    spinner.color = UIColor.whiteColor;
    UIView *overlay = [self.view viewWithTag:999];
    [overlay addSubview:spinner];
    
    // Center the spinner in the overlay
    [NSLayoutConstraint activateConstraints:@[
        [spinner.centerXAnchor constraintEqualToAnchor:overlay.centerXAnchor],
        [spinner.centerYAnchor constraintEqualToAnchor:overlay.centerYAnchor]
    ]];
    
    [spinner startAnimating];
    
    // Disable user interaction on the overlay to prevent multiple taps
    overlay.userInteractionEnabled = NO;
    
    // Fetch and initiate the purchase
    [[StoreManager sharedInstance] fetchAndPurchaseProductWithCompletion:^(BOOL success, NSError * _Nullable error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            // Stop the spinner and re-enable interaction
            [spinner stopAnimating];
            [spinner removeFromSuperview];
            overlay.userInteractionEnabled = YES;
            
            // Dismiss the upgrade popup
            [self dismissUpgradePopup];
            
            if (!success && error) {
                // Show error alert if purchase failed
                UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"purchase_failed", @"Title for purchase failed alert")
                                                                               message:error.localizedDescription
                                                                        preferredStyle:UIAlertControllerStyleAlert];
                [alert addAction:[UIAlertAction actionWithTitle:NSLocalizedString(@"ok", @"OK button") style:UIAlertActionStyleDefault handler:nil]];
                [self presentViewController:alert animated:YES completion:nil];
            }
            
            // Reload table to reflect subscription status
            [self.tableView reloadData];
        });
    }];
}

- (void)dismissUpgradePopup {
    UIView *overlay = [self.view viewWithTag:999];
    [overlay removeFromSuperview];
}

- (void)useOwnnotificationServerSwitchToggled:(UISwitch *)sender {
    self.useOwnServerEnabled = sender.on;
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    [defaults setBool:self.useOwnServerEnabled forKey:@"use_own_server_enabled"];
    [defaults synchronize];
    
    // Check subscription status and password
    BOOL isPremium = [defaults boolForKey:@"isSubscribed"];
    NSString *password = [self retrievePasswordFromSecretsManager];
    
    if (!self.useOwnServerEnabled && (!isPremium || !password)) {
        self.sendNotifEnabled = NO;
        [defaults setBool:NO forKey:@"send_notif_enabled"];
        [defaults synchronize];
    }
    
    if (sender.on) {
        [self shownotificationServerAddressInputDialog];
    }
    
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:0] withRowAnimation:UITableViewRowAnimationAutomatic];
}

- (void)shownotificationServerAddressInputDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"enter_notification_server_address_title", @"Title for notification server address input dialog")
                                                                   message:NSLocalizedString(@"enter_notification_server_address_message", @"Message for notification server address input dialog")
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = NSLocalizedString(@"server_address_label", @"Placeholder for server address input");
        textField.text = self.notificationServerAddress;
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"save", @"Save button")
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *address = alert.textFields.firstObject.text;
        if (address.length > 0) {
            self.notificationServerAddress = address;
            NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
            [defaults setObject:address forKey:@"own_notification_server_address"];
            [defaults synchronize];
            [self.tableView reloadData];
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"cancel", @"Cancel button")
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];

    [self presentViewController:alert animated:YES completion:nil];
}

- (void)testnotificationServer {
    [[notification sharedInstance] sendNotification];
    UIAlertController *resultAlert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"test_notification_sent_title", @"Title for test notification sent alert")
                                                                        message:NSLocalizedString(@"test_notification_sent_message", @"Message for test notification sent alert")
                                                                 preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"ok", @"OK button")
                                                       style:UIAlertActionStyleDefault
                                                     handler:nil];
    [resultAlert addAction:okAction];
    [self presentViewController:resultAlert animated:YES completion:nil];
}

- (void)sendNotifSwitchToggled:(UISwitch *)sender {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    BOOL isPremium = [defaults boolForKey:@"isSubscribed"];
    
    if (isPremium || self.useOwnServerEnabled) {
        if (sender.on && !self.useOwnServerEnabled) {
            NSString *password = [self retrievePasswordFromSecretsManager];
            if (!password) {
                [self promptForPasswordWithCompletion:^(BOOL success) {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        if (success) {
                            self.sendNotifEnabled = YES;
                            [defaults setBool:YES forKey:@"send_notif_enabled"];
                        } else {
                            self.sendNotifEnabled = NO;
                            [defaults setBool:NO forKey:@"send_notif_enabled"];
                            sender.on = NO;
                        }
                        [defaults synchronize];
                        [self.tableView reloadData];
                    });
                }];
            } else {
                self.sendNotifEnabled = sender.on;
                [defaults setBool:self.sendNotifEnabled forKey:@"send_notif_enabled"];
                [defaults synchronize];
            }
        } else {
            self.sendNotifEnabled = sender.on;
            [defaults setBool:self.sendNotifEnabled forKey:@"send_notif_enabled"];
            [defaults synchronize];
        }
    } else {
        sender.on = NO;
        self.sendNotifEnabled = NO;
        [defaults setBool:NO forKey:@"send_notif_enabled"];
        [defaults synchronize];
        [self showUpgradePopup];
    }
}

- (void)promptForPasswordWithCompletion:(void (^)(BOOL success))completion {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"set_password_title", @"Title for password input dialog")
                                                                   message:NSLocalizedString(@"set_password_message", @"Message for password input dialog")
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = NSLocalizedString(@"password_placeholder", @"Placeholder for password input");
        textField.secureTextEntry = YES;
    }];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = NSLocalizedString(@"confirm_password_placeholder", @"Placeholder for confirm password input");
        textField.secureTextEntry = YES;
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"save", @"Save button")
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *password = alert.textFields[0].text;
        NSString *confirmPassword = alert.textFields[1].text;
        
        if ([password isEqualToString:confirmPassword] && password.length > 0) {
            [self savePasswordToSecretsManager:password];
            completion(YES);
        } else {
            [self showInvalidPasswordAlert];
            completion(NO);
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"cancel", @"Cancel button")
                                                           style:UIAlertActionStyleCancel
                                                         handler:^(UIAlertAction * _Nonnull action) {
        completion(NO);
    }];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];
    
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)savePasswordToSecretsManager:(NSString *)password {
    if (!password) return;
    
    NSError *error = nil;
    [[SecretManager sharedManager] saveEncryptionKey:password error:&error];
}

- (NSString *)retrievePasswordFromSecretsManager {
    NSString *storedKey = [[SecretManager sharedManager] getEncryptionKey];
    if (storedKey) {
        return storedKey;
    }
    return nil;
}

- (void)showInvalidPasswordAlert {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"invalid_password_title", @"Title for invalid password alert")
                                                                   message:NSLocalizedString(@"invalid_password_message", @"Message for invalid password alert")
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"ok", @"OK button")
                                                       style:UIAlertActionStyleDefault
                                                     handler:nil];
    
    [alert addAction:okAction];
    [self presentViewController:alert animated:YES completion:nil];
}

#pragma mark - UITableView Delegate

- (UITableViewCellEditingStyle)tableView:(UITableView *)tableView editingStyleForRowAtIndexPath:(NSIndexPath *)indexPath {
    if (indexPath.section == 0 && self.isPresetsSectionExpanded) { // Only in Camera Settings
        NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
        if (indexPath.row >= 4 && indexPath.row < 4 + presetKeys.count) {
            NSString *presetKey = presetKeys[indexPath.row - 4];
            if (![presetKey isEqualToString:@"all"]) {
                return UITableViewCellEditingStyleDelete;
            }
        }
    }
    return UITableViewCellEditingStyleNone;
}

- (void)tableView:(UITableView *)tableView commitEditingStyle:(UITableViewCellEditingStyle)editingStyle forRowAtIndexPath:(NSIndexPath *)indexPath {
    if (editingStyle == UITableViewCellEditingStyleDelete && indexPath.section == 0 && self.isPresetsSectionExpanded) {
        NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
        if (indexPath.row >= 4 && indexPath.row < 4 + presetKeys.count) {
            NSString *presetKey = presetKeys[indexPath.row - 4];

            if ([presetKey isEqualToString:@"all"]) return;

            NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
            NSMutableDictionary *yoloPresets = [[defaults objectForKey:@"yolo_presets"] mutableCopy];
            [yoloPresets removeObjectForKey:presetKey];
            [defaults setObject:yoloPresets forKey:@"yolo_presets"];

            if ([self.selectedPresetKey isEqualToString:presetKey]) {
                self.selectedPresetKey = @"all";
                [defaults setObject:@"all" forKey:@"yolo_preset_idx"];
            }

            [defaults synchronize];

            [tableView beginUpdates];
            [tableView deleteRowsAtIndexPaths:@[indexPath] withRowAnimation:UITableViewRowAnimationAutomatic];
            [tableView endUpdates];

            [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:0] withRowAnimation:UITableViewRowAnimationAutomatic];
        }
    }
}

#pragma mark - Resolution Picker

- (void)showResolutionPicker {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"select_resolution_title", @"Title for resolution picker")
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];

    NSArray *resolutions = @[@"720p", @"1080p"];

    for (NSString *resolution in resolutions) {
        UIAlertAction *action = [UIAlertAction actionWithTitle:resolution
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
            self.selectedResolution = resolution;
            SettingsManager *settingsManager = [SettingsManager sharedManager];
            if ([resolution isEqualToString:@"720p"]) {
                [settingsManager updateResolutionWithWidth:@"1280" height:@"720" textSize:@"2" preset:@"AVCaptureSessionPreset1280x720"];
            } else if ([resolution isEqualToString:@"1080p"]) {
                [settingsManager updateResolutionWithWidth:@"1920" height:@"1080" textSize:@"3" preset:@"AVCaptureSessionPreset1920x1080"];
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.tableView reloadData];
            });
        }];
        [alert addAction:action];
    }

    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"cancel", @"Cancel button")
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    [alert addAction:cancelAction];

    dispatch_async(dispatch_get_main_queue(), ^{
        [self presentViewController:alert animated:YES completion:nil];
    });
}

#pragma mark - YOLO Indexes Picker

- (void)showYoloIndexesPicker {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"select_objects_preset_title", @"Title for YOLO preset picker")
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    SettingsManager *settingsManager = [SettingsManager sharedManager];
    NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
    for (NSString *presetKey in presetKeys) {
        UIAlertAction *action = [UIAlertAction actionWithTitle:presetKey
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
            self.selectedPresetKey = presetKey;
            [settingsManager updateYoloIndexesKey:presetKey];
            [self.tableView reloadData];
        }];
        [alert addAction:action];
    }

    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"cancel", @"Cancel button")
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    [alert addAction:cancelAction];

    [self presentViewController:alert animated:YES completion:nil];
}

#pragma mark - Preset Management

- (void)showAddPresetDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"add_preset_title", @"Title for add preset dialog")
                                                                   message:NSLocalizedString(@"add_preset_message", @"Message for add preset name input")
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = NSLocalizedString(@"preset_name_placeholder", @"Placeholder for preset name input");
    }];
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:NSLocalizedString(@"save", @"Save button")
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *presetName = alert.textFields.firstObject.text;
        if (presetName.length > 0) {
            [self showNumberSelectionForPreset:presetName selectedIndexes:@[]];
        }
    }];
    [alert addAction:saveAction];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)showNumberSelectionForPreset:(NSString *)presetKey selectedIndexes:(NSArray<NSNumber *> *)selectedIndexes {
    NumberSelectionViewController *numberSelectionVC = [[NumberSelectionViewController alloc] init];
    numberSelectionVC.presetKey = presetKey;
    numberSelectionVC.selectedIndexes = [selectedIndexes mutableCopy];
    numberSelectionVC.completionHandler = ^(NSArray<NSNumber *> *selectedIndexes) {
        NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
        NSMutableDictionary *yoloPresets = [[defaults objectForKey:@"yolo_presets"] mutableCopy];

        if (!yoloPresets) {
            yoloPresets = [NSMutableDictionary dictionary];
        }
        yoloPresets[presetKey] = selectedIndexes;
        [defaults setObject:yoloPresets forKey:@"yolo_presets"];
        self.selectedPresetKey = presetKey;
        [defaults setObject:presetKey forKey:@"yolo_preset_idx"];
        [defaults synchronize];
        [self.tableView reloadData];
    };
    [self.navigationController pushViewController:numberSelectionVC animated:YES];
}

@end
