#import "SettingsViewController.h"
#import "SettingsManager.h"
#import "SecretManager.h"
#import "StoreManager.h"
#import "NumberSelectionViewController.h"
#import "Email.h"
#import "ScheduleManagementViewController.h"
#import <UserNotifications/UserNotifications.h>

@interface SettingsViewController () <UITableViewDelegate, UITableViewDataSource>

@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSString *selectedResolution;
@property (nonatomic, strong) NSString *selectedPresetKey; // For YOLO indexes key
@property (nonatomic, assign) BOOL isPresetsSectionExpanded; // Track if presets section is expanded
@property (nonatomic, assign) BOOL sendNotifEnabled;
@property (nonatomic, assign) BOOL receiveNotifEnabled; // New property for receiving notifications
@property (nonatomic, assign) BOOL useOwnServerEnabled;
@property (nonatomic, assign) BOOL isEmailServerSectionExpanded; // Track if email server section is expanded
@property (nonatomic, strong) NSString *emailServerAddress; // Store the email server address
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *notificationSchedules; // Array to store notification schedules
@property (nonatomic, assign) BOOL streamViaWiFiEnabled;
@property (nonatomic, strong) id ipAddressObserver;
@property (nonatomic, assign) NSInteger threshold; // New property for threshold

@end

@implementation SettingsViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Settings";
    if (![[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"] || ![[NSDate date] compare:[[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"]] || [[NSDate date] compare:[[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"]] == NSOrderedDescending) {
        [[StoreManager sharedInstance] verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.tableView reloadData];
            });
        }];
    }
    NSLog(@"%d %@", [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"], [[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"]);
    
    // Register for subscription status change notifications
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(subscriptionStatusDidChange:)
                                                 name:StoreManagerSubscriptionStatusDidChangeNotification
                                               object:nil];
    
    // Initialize properties from NSUserDefaults
    SettingsManager *settingsManager = [SettingsManager sharedManager];
    self.selectedResolution = [NSString stringWithFormat:@"%@p", settingsManager.height];
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
    
    self.isEmailServerSectionExpanded = self.useOwnServerEnabled;
    self.emailServerAddress = [defaults stringForKey:@"own_email_server_address"] ?: @"http://192.168.1.1";
    
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

- (void)updateIPAddressDisplay {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    NSString *ipAddress = [defaults stringForKey:@"DeviceIPAddress"];
    if (ipAddress && ipAddress.length > 0) {
        self.emailServerAddress = [NSString stringWithFormat:@"http://%@", ipAddress];
    } else {
        self.emailServerAddress = @"Waiting for IP...";
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

- (void)receiveNotifSwitchToggled:(UISwitch *)sender {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    BOOL hasRequestedPermission = [defaults boolForKey:@"hasRequestedNotificationPermission"];
    
    self.receiveNotifEnabled = sender.on;
    [defaults setBool:self.receiveNotifEnabled forKey:@"receive_notif_enabled"];
    [defaults synchronize];

    if (sender.on) {
        NSLog(@"Receive Notifications on This Device turned ON");

        UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
        [center getNotificationSettingsWithCompletionHandler:^(UNNotificationSettings * _Nonnull settings) {
            dispatch_async(dispatch_get_main_queue(), ^{
                if (settings.authorizationStatus == UNAuthorizationStatusAuthorized) {
                    NSLog(@"Notifications already authorized, sending token...");
                    [self sendDeviceTokenToServer]; // ✅ Always send token when toggled ON
                } else if (!hasRequestedPermission) {
                    NSLog(@"Requesting permission for notifications...");
                    [center requestAuthorizationWithOptions:(UNAuthorizationOptionAlert | UNAuthorizationOptionSound | UNAuthorizationOptionBadge)
                                          completionHandler:^(BOOL granted, NSError * _Nullable error) {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            if (granted) {
                                NSLog(@"Notification permission granted");
                                [[UIApplication sharedApplication] registerForRemoteNotifications];
                                [defaults setBool:YES forKey:@"hasRequestedNotificationPermission"];
                                [defaults synchronize];
                                
                                [self sendDeviceTokenToServer]; // ✅ Send token after granting permission
                            } else {
                                NSLog(@"Notification permission denied: %@", error);
                                self.receiveNotifEnabled = NO;
                                [defaults setBool:NO forKey:@"receive_notif_enabled"];
                                [defaults synchronize];
                                sender.on = NO;
                                
                                UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Permission Denied"
                                                                                              message:@"Notification permission was denied. You can enable it in Settings."
                                                                                       preferredStyle:UIAlertControllerStyleAlert];
                                [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
                                [self presentViewController:alert animated:YES completion:nil];
                            }
                        });
                    }];
                } else {
                    NSLog(@"Notifications not authorized, showing settings alert.");
                    self.receiveNotifEnabled = NO;
                    [defaults setBool:NO forKey:@"receive_notif_enabled"];
                    [defaults synchronize];
                    sender.on = NO;
                    
                    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Enable in Settings"
                                                                                  message:@"Notifications are disabled. Please enable them in the Settings app."
                                                                           preferredStyle:UIAlertControllerStyleAlert];
                    [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
                    [self presentViewController:alert animated:YES completion:nil];
                }
            });
        }];
    } else {
        NSLog(@"Receive Notifications on This Device turned OFF");
        // Clear any pending notifications when turning off
        UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
        [center removeAllPendingNotificationRequests];
        [center removeAllDeliveredNotifications];
    }
}


- (void)sendDeviceTokenToServer {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    NSString *deviceToken = [defaults stringForKey:@"device_token"];
    
    if (!deviceToken || deviceToken.length == 0) {
        NSLog(@"No device token found, skipping API call.");
        return;
    }
    
    // Retrieve session token from Keychain
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken || sessionToken.length == 0) {
        NSLog(@"No session token found in Keychain. Skipping API call.");
        return;
    }
    
    NSURL *url = [NSURL URLWithString:@"https://rors.ai/add_device"];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    request.HTTPMethod = @"POST";
    [request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
    
    NSDictionary *body = @{
        @"device_token": deviceToken,
        @"session_token": sessionToken
    };
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:body options:0 error:nil];
    request.HTTPBody = jsonData;
    
    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithRequest:request
                                                                 completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"Error sending device token: %@", error.localizedDescription);
            return;
        }
        
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
        if (httpResponse.statusCode == 200) {
            NSLog(@"Device token successfully sent to server");
        } else {
            NSLog(@"Failed to send device token, server responded with status code: %ld", (long)httpResponse.statusCode);
        }
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
    return 2;
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    if (section == 0) {
        NSInteger baseRows = 10;
        if (self.useOwnServerEnabled && self.isEmailServerSectionExpanded) {
            baseRows += 2;
        }
        if (self.isPresetsSectionExpanded) {
            NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
            baseRows += presetKeys.count + 1;
        }
        return baseRows;
    } else if (section == 1) {
        return [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"] ? 0 : 1;
    }
    return 0;
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

    if (indexPath.section == 0) {
        if (indexPath.row == 0) {
            cell.textLabel.text = @"Stream via Wi-Fi";
            NSString *ipAddress = [[NSUserDefaults standardUserDefaults] stringForKey:@"DeviceIPAddress"];
            cell.detailTextLabel.text = ipAddress && ipAddress.length > 0 ?
                                       [NSString stringWithFormat:@"http://%@", ipAddress] :
                                       @"Waiting for IP...";
            cell.accessoryType = UITableViewCellAccessoryNone;
            UISwitch *wifiSwitch = [[UISwitch alloc] init];
            wifiSwitch.on = self.streamViaWiFiEnabled;
            [wifiSwitch addTarget:self action:@selector(streamViaWiFiSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = wifiSwitch;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 1) {
            cell.textLabel.text = @"Resolution";
            cell.detailTextLabel.text = self.selectedResolution;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 2) {
            cell.textLabel.text = @"Detect Objects";
            cell.detailTextLabel.text = self.selectedPresetKey;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 3) {
            cell.textLabel.text = @"Manage Detection Presets";
            cell.detailTextLabel.text = nil;
            cell.accessoryType = self.isPresetsSectionExpanded ? UITableViewCellAccessoryNone : UITableViewCellAccessoryDisclosureIndicator;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 4 && !self.isPresetsSectionExpanded) {
            cell.textLabel.text = @"Detection Certainty Threshold";
            cell.detailTextLabel.text = [NSString stringWithFormat:@"%ld%%", (long)self.threshold];
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
            NSInteger offset = self.isPresetsSectionExpanded ? [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys].count + 1 : 0;
            if (indexPath.row == 4 + offset) {
                cell.textLabel.text = @"Detection Certainty Threshold";
                cell.detailTextLabel.text = [NSString stringWithFormat:@"%ld%%", (long)self.threshold];
                cell.userInteractionEnabled = YES;
            } else if (indexPath.row == 5 + offset) {
                cell.textLabel.text = @"Send Notifications on Events";
                cell.accessoryType = UITableViewCellAccessoryNone;
                UISwitch *sendNotifSwitch = [[UISwitch alloc] init];
                sendNotifSwitch.on = isPremium ? self.sendNotifEnabled : NO;
                [sendNotifSwitch addTarget:self action:@selector(sendNotifSwitchToggled:) forControlEvents:UIControlEventValueChanged];
                cell.accessoryView = sendNotifSwitch;
                sendNotifSwitch.enabled = isPremium;
                cell.textLabel.textColor = isPremium ? [UIColor labelColor] : [UIColor grayColor];
                cell.userInteractionEnabled = YES;
            } else if (indexPath.row == 6 + offset) {
                cell.textLabel.text = @"Receive Notifications on This Device";
                cell.accessoryType = UITableViewCellAccessoryNone;
                UISwitch *receiveNotifSwitch = [[UISwitch alloc] init];
                receiveNotifSwitch.on = self.receiveNotifEnabled;
                [receiveNotifSwitch addTarget:self action:@selector(receiveNotifSwitchToggled:) forControlEvents:UIControlEventValueChanged];
                cell.accessoryView = receiveNotifSwitch;
                cell.userInteractionEnabled = YES;
            } else if (indexPath.row == 7 + offset) {
                cell.textLabel.text = @"Change Encryption Password";
                cell.detailTextLabel.text = nil;
                cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
                cell.textLabel.textColor = isPremium ? [UIColor labelColor] : [UIColor grayColor];
                cell.userInteractionEnabled = YES;
            } else if (indexPath.row == 8 + offset) {
                cell.textLabel.text = @"Manage Notification Schedules";
                cell.detailTextLabel.text = nil;
                cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
                cell.userInteractionEnabled = YES;
            } else if (indexPath.row == 9 + offset) {
                cell.textLabel.text = @"Use Own Notification Server";
                cell.accessoryType = UITableViewCellAccessoryNone;
                UISwitch *useOwnEmailServerSwitch = [[UISwitch alloc] init];
                useOwnEmailServerSwitch.on = self.useOwnServerEnabled;
                [useOwnEmailServerSwitch addTarget:self action:@selector(useOwnEmailServerSwitchToggled:) forControlEvents:UIControlEventValueChanged];
                cell.accessoryView = useOwnEmailServerSwitch;
                cell.userInteractionEnabled = YES;
            } else if (self.useOwnServerEnabled && self.isEmailServerSectionExpanded && indexPath.row == 10 + offset) {
                cell.textLabel.text = @"Server Address";
                cell.detailTextLabel.text = self.emailServerAddress;
                cell.userInteractionEnabled = YES;
            } else if (self.useOwnServerEnabled && self.isEmailServerSectionExpanded && indexPath.row == 11 + offset) {
                cell.textLabel.text = @"Test own server";
                cell.textLabel.textColor = [UIColor systemBlueColor];
                cell.detailTextLabel.text = nil;
                cell.accessoryType = UITableViewCellAccessoryNone;
                cell.userInteractionEnabled = YES;
            }
        }
    } else if (indexPath.section == 1) {
        cell.textLabel.text = @"Upgrade to Premium";
        cell.textLabel.textColor = [UIColor systemBlueColor];
        cell.textLabel.textAlignment = NSTextAlignmentCenter;
        cell.accessoryType = UITableViewCellAccessoryNone;
        cell.userInteractionEnabled = YES;
    }

    return cell;
}

- (void)showThresholdInputDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Set Threshold"
                                                                   message:@"Enter a % value between 1 and 100 (default is 25)."
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Threshold (1-100)";
        textField.keyboardType = UIKeyboardTypeNumberPad;
        textField.text = [NSString stringWithFormat:@"%ld", (long)self.threshold];
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:@"Save"
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
            UIAlertController *errorAlert = [UIAlertController alertControllerWithTitle:@"Invalid Threshold"
                                                                               message:@"Please enter a valid value between 1 and 100"
                                                                        preferredStyle:UIAlertControllerStyleAlert];
            [errorAlert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
            [self presentViewController:errorAlert animated:YES completion:nil];
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];
    
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    BOOL isPremium = [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"];
    
    if (indexPath.section == 0) {
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
                    [self showPremiumRequiredAlert];
                }
            } else if (indexPath.row == 6 + offset) {
                // Receive Notifications - handled by switch
            } else if (indexPath.row == 7 + offset) {
                if (isPremium) {
                    [self promptForPasswordWithCompletion:^(BOOL success) {
                        if (success) {
                            NSLog(@"Encryption password changed successfully.");
                        }
                    }];
                } else {
                    [self showPremiumRequiredAlert];
                }
            } else if (indexPath.row == 8 + offset) {
                ScheduleManagementViewController *scheduleVC = [[ScheduleManagementViewController alloc] init];
                scheduleVC.emailSchedules = [self.notificationSchedules mutableCopy];
                scheduleVC.completionHandler = ^(NSArray<NSDictionary *> *schedules) {
                    self.notificationSchedules = [schedules mutableCopy];
                    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
                    [defaults setObject:self.notificationSchedules forKey:@"notification_schedules"];
                    [defaults synchronize];
                    [self.tableView reloadData];
                };
                [self.navigationController pushViewController:scheduleVC animated:YES];
            } else if (indexPath.row == 9 + offset) {
                // Use Own Notification Server - handled by switch
            } else if (self.useOwnServerEnabled && self.isEmailServerSectionExpanded && indexPath.row == 10 + offset) {
                [self showEmailServerAddressInputDialog];
            } else if (self.useOwnServerEnabled && self.isEmailServerSectionExpanded && indexPath.row == 11 + offset) {
                [self testEmailServer];
            }
        }
    } else if (indexPath.section == 1) {
        [[StoreManager sharedInstance] fetchAndPurchaseProduct];
    }
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

- (void)showPremiumRequiredAlert {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Premium Required"
                                                                   message:@"This feature requires a premium subscription."
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK"
                                                       style:UIAlertActionStyleDefault
                                                     handler:nil];
    
    [alert addAction:okAction];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)useOwnEmailServerSwitchToggled:(UISwitch *)sender {
    self.useOwnServerEnabled = sender.on;
    self.isEmailServerSectionExpanded = sender.on;
    
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    [defaults setBool:self.useOwnServerEnabled forKey:@"use_own_server_enabled"];
    [defaults synchronize];
    
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:0] withRowAnimation:UITableViewRowAnimationAutomatic];
}

- (void)showEmailServerAddressInputDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Enter Server Address"
                                                                   message:@"Please enter the address of your email server."
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Server Address";
        textField.text = self.emailServerAddress;
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:@"Save"
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *address = alert.textFields.firstObject.text;
        if (address.length > 0) {
            self.emailServerAddress = address;
            NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
            [defaults setObject:address forKey:@"own_email_server_address"];
            [defaults synchronize];
            [self.tableView reloadData];
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];

    [self presentViewController:alert animated:YES completion:nil];
}

- (void)testEmailServer {
    [[Email sharedInstance] sendEmailWithImageAtPath:@""];
    UIAlertController *resultAlert = [UIAlertController alertControllerWithTitle:@"Test Initiated"
                                                                        message:@"Test email has been initiated. Check your server logs for results."
                                                                 preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK"
                                                       style:UIAlertActionStyleDefault
                                                     handler:nil];
    [resultAlert addAction:okAction];
    [self presentViewController:resultAlert animated:YES completion:nil];
}

- (void)sendNotifSwitchToggled:(UISwitch *)sender {
    if (![[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"]) {
        sender.on = NO;
        [self showPremiumRequiredAlert];
        return;
    }
    
    if (sender.on) {
        NSString *password = [self retrievePasswordFromSecretsManager];
        if (!password) {
            [self promptForPasswordWithCompletion:^(BOOL success) {
                if (success) {
                    self.sendNotifEnabled = YES;
                    [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"send_notif_enabled"];
                    [[NSUserDefaults standardUserDefaults] synchronize];
                } else {
                    sender.on = NO;
                }
            }];
        } else {
            self.sendNotifEnabled = YES;
            [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"send_notif_enabled"];
            [[NSUserDefaults standardUserDefaults] synchronize];
        }
    } else {
        self.sendNotifEnabled = NO;
        [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"send_notif_enabled"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

- (void)promptForPasswordWithCompletion:(void (^)(BOOL success))completion {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Set Password"
                                                                   message:@"Enter a password to encrypt and decrypt your data"
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Password";
        textField.secureTextEntry = YES;
    }];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Confirm Password";
        textField.secureTextEntry = YES;
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:@"Save"
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
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
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
    BOOL success = [[SecretManager sharedManager] saveEncryptionKey:password error:&error];
    
    if (!success) NSLog(@"Failed to save password to Keychain: %@", error.localizedDescription);
}

- (NSString *)retrievePasswordFromSecretsManager {
    NSString *storedKey = [[SecretManager sharedManager] getEncryptionKey];
    if (storedKey) {
        return storedKey;
    }
    NSLog(@"No password found in Keychain.");
    return nil;
}

- (void)showInvalidPasswordAlert {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Invalid Password"
                                                                   message:@"Passwords do not match or are empty."
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK"
                                                       style:UIAlertActionStyleDefault
                                                     handler:nil];
    
    [alert addAction:okAction];
    [self presentViewController:alert animated:YES completion:nil];
}

#pragma mark - UITableView Delegate

- (UITableViewCellEditingStyle)tableView:(UITableView *)tableView editingStyleForRowAtIndexPath:(NSIndexPath *)indexPath {
    if (indexPath.section == 0 && self.isPresetsSectionExpanded) {
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

            if ([presetKey isEqualToString:@"all"]) {
                NSLog(@"Cannot delete the 'all' preset.");
                return;
            }

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
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Select Resolution"
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

    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    [alert addAction:cancelAction];

    dispatch_async(dispatch_get_main_queue(), ^{
        [self presentViewController:alert animated:YES completion:nil];
    });
}

#pragma mark - YOLO Indexes Picker

- (void)showYoloIndexesPicker {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Select objects preset"
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

    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    [alert addAction:cancelAction];

    [self presentViewController:alert animated:YES completion:nil];
}

#pragma mark - Preset Management

- (void)showAddPresetDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Add Preset"
                                                                   message:@"Enter a name for the new preset"
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Preset Name";
    }];
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:@"Save"
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
