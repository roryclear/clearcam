#import "SettingsViewController.h"
#import "SettingsManager.h"
#import "SecretManager.h"
#import "StoreManager.h"
#import "NumberSelectionViewController.h"
#import "Email.h"

@interface SettingsViewController () <UITableViewDelegate, UITableViewDataSource>

@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSString *selectedResolution;
@property (nonatomic, strong) NSString *selectedPresetKey; // For YOLO indexes key
@property (nonatomic, assign) BOOL isPresetsSectionExpanded; // Track if presets section is expanded
@property (nonatomic, assign) BOOL sendEmailAlertsEnabled;
@property (nonatomic, assign) BOOL encryptEmailDataEnabled;
@property (nonatomic, assign) BOOL useOwnEmailServerEnabled; // Track if "Use own email server" is enabled
@property (nonatomic, assign) BOOL isEmailServerSectionExpanded; // Track if email server section is expanded
@property (nonatomic, strong) NSString *emailServerAddress; // Store the email server address

@end

@implementation SettingsViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Basic setup
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Settings";
    
    // Check subscription status on view load
    [self checkSubscriptionStatus];
    
    // Register for subscription status change notifications
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(subscriptionStatusDidChange:)
                                                 name:StoreManagerSubscriptionStatusDidChangeNotification
                                               object:nil];
    
    // Initialize selectedResolution and selectedPresetKey based on SettingsManager
    SettingsManager *settingsManager = [SettingsManager sharedManager];
    self.selectedResolution = [NSString stringWithFormat:@"%@p", settingsManager.height];
    self.selectedPresetKey = [[NSUserDefaults standardUserDefaults] stringForKey:@"yolo_preset_idx"] ?: @"all"; // Default to "all" if no key is saved
    
    // Initialize sendEmailAlertsEnabled from NSUserDefaults
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    if ([defaults objectForKey:@"send_email_alerts_enabled"] != nil) {
        self.sendEmailAlertsEnabled = [defaults boolForKey:@"send_email_alerts_enabled"];
    } else {
        self.sendEmailAlertsEnabled = NO;
        [defaults setBool:NO forKey:@"send_email_alerts_enabled"];
        [defaults synchronize];
    }
    
    // Initialize encryptEmailDataEnabled from NSUserDefaults
    if ([defaults objectForKey:@"encrypt_email_data_enabled"] != nil) {
        self.encryptEmailDataEnabled = [defaults boolForKey:@"encrypt_email_data_enabled"];
    } else {
        self.encryptEmailDataEnabled = NO;
        [defaults setBool:NO forKey:@"encrypt_email_data_enabled"];
        [defaults synchronize];
    }
    
    // Initialize useOwnEmailServerEnabled from NSUserDefaults
    if ([defaults objectForKey:@"use_own_email_server_enabled"] != nil) {
        self.useOwnEmailServerEnabled = [defaults boolForKey:@"use_own_email_server_enabled"];
    } else {
        self.useOwnEmailServerEnabled = NO;
        [defaults setBool:NO forKey:@"use_own_email_server_enabled"];
        [defaults synchronize];
    }
    
    // Initialize isEmailServerSectionExpanded based on useOwnEmailServerEnabled
    self.isEmailServerSectionExpanded = self.useOwnEmailServerEnabled; // Add this line
    
    // Initialize emailServerAddress from NSUserDefaults
    self.emailServerAddress = [defaults stringForKey:@"own_email_server_address"] ?: @"http://192.168.1.1"; // Default value
    
    // Create table view with proper Auto Layout constraints
    self.tableView = [[UITableView alloc] initWithFrame:CGRectZero style:UITableViewStyleInsetGrouped];
    self.tableView.delegate = self;
    self.tableView.dataSource = self;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor]; // Matches dark/light mode
    self.tableView.translatesAutoresizingMaskIntoConstraints = NO; // Enable Auto Layout
    [self.view addSubview:self.tableView];
    
    // Set up constraints to pin the table view to all edges of the view
    [NSLayoutConstraint activateConstraints:@[
        [self.tableView.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [self.tableView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.tableView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.tableView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor]
    ]];
}

- (void)dealloc {
    // Unregister from notifications to prevent crashes
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

#pragma mark - Subscription Status

- (void)checkSubscriptionStatus {
    [[StoreManager sharedInstance] verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (isActive) {
                NSLog(@"Subscription is active. Expiry date: %@", expiryDate);
            } else {
                NSLog(@"No active subscription found.");
            }
            [self.tableView reloadData]; // Refresh the UI based on the subscription status
        });
    }];
}

- (void)subscriptionStatusDidChange:(NSNotification *)notification {
    // Called when the subscription status changes (e.g., after a successful purchase)
    dispatch_async(dispatch_get_main_queue(), ^{
        NSLog(@"Subscription status changed. Refreshing UI.");
        [self.tableView reloadData]; // Refresh the UI to reflect the new subscription status
    });
}

#pragma mark - Orientation Control

- (BOOL)shouldAutorotate {
    return NO; // Disable autorotation to lock orientation
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
    return UIInterfaceOrientationMaskPortrait; // Only allow portrait orientation
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
    return UIInterfaceOrientationPortrait; // Set the preferred orientation to portrait
}

#pragma mark - UITableView DataSource

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return 3; // One for general settings, one for presets, one for upgrade to premium
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    if (section == 0) {
        // Resolution, Detect objects, Email, Send Email Alerts, Encrypt Email Data, Use own email server
        NSInteger baseRows = 6; // Existing 6 rows
        if (self.useOwnEmailServerEnabled && self.isEmailServerSectionExpanded) {
            return baseRows + 2; // Add 2 rows: one for Address field and one for Test button
        }
        return baseRows;
    } else if (section == 1) {
        // Presets section: 1 row for "Manage Presets" header, plus rows for each preset, plus "Add New +"
        NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
        return self.isPresetsSectionExpanded ? (presetKeys.count + 2) : 1;
    } else if (section == 2) {
        return [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"] ? 0 : 1; // Show "Upgrade to Premium" button only if not subscribed
    }
    return 0;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"SettingsCell"];
    
    if (!cell) {
        cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleValue1 reuseIdentifier:@"SettingsCell"];
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

    BOOL isPremiumOrUsingOwnServer = [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"] || self.useOwnEmailServerEnabled;

    if (indexPath.section == 0) {
        if (indexPath.row == 0) {
            cell.textLabel.text = @"Resolution";
            cell.detailTextLabel.text = self.selectedResolution;
            cell.userInteractionEnabled = YES;
            NSLog(@"Resolution cell configured, tappable: YES");
        } else if (indexPath.row == 1) {
            cell.textLabel.text = @"Detect objects";
            cell.detailTextLabel.text = self.selectedPresetKey;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 2) {
            cell.textLabel.text = @"Email Address";
            NSString *email = [[NSUserDefaults standardUserDefaults] stringForKey:@"user_email"];
            cell.detailTextLabel.text = email ?: @"Not set";
            if (!isPremiumOrUsingOwnServer) {
                cell.textLabel.textColor = [UIColor grayColor];
                cell.detailTextLabel.textColor = [UIColor grayColor];
                cell.userInteractionEnabled = NO;
                NSLog(@"Email Address cell configured, tappable: NO (not premium or using own server)");
            } else {
                cell.textLabel.textColor = [UIColor labelColor];
                cell.detailTextLabel.textColor = [UIColor secondaryLabelColor];
                cell.userInteractionEnabled = YES;
                NSLog(@"Email Address cell configured, tappable: YES");
            }
        } else if (indexPath.row == 3) {
            cell.textLabel.text = @"Send Email Alerts";
            cell.accessoryType = UITableViewCellAccessoryNone;
            UISwitch *emailAlertsSwitch = [[UISwitch alloc] init];
            emailAlertsSwitch.on = self.sendEmailAlertsEnabled;
            [emailAlertsSwitch addTarget:self action:@selector(emailAlertsSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = emailAlertsSwitch;
            if (!isPremiumOrUsingOwnServer) {
                emailAlertsSwitch.enabled = NO;
                cell.textLabel.textColor = [UIColor grayColor];
                cell.userInteractionEnabled = NO;
            } else {
                emailAlertsSwitch.enabled = YES;
                cell.textLabel.textColor = [UIColor labelColor];
                cell.userInteractionEnabled = YES;
            }
        } else if (indexPath.row == 4) {
            cell.textLabel.text = @"Encrypt Email Data (Recommended)";
            cell.accessoryType = UITableViewCellAccessoryNone;
            UISwitch *encryptEmailDataSwitch = [[UISwitch alloc] init];
            encryptEmailDataSwitch.on = self.encryptEmailDataEnabled;
            [encryptEmailDataSwitch addTarget:self action:@selector(encryptEmailDataSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = encryptEmailDataSwitch;
            if (!isPremiumOrUsingOwnServer) {
                encryptEmailDataSwitch.enabled = NO;
                cell.textLabel.textColor = [UIColor grayColor];
                cell.userInteractionEnabled = NO;
            } else {
                encryptEmailDataSwitch.enabled = YES;
                cell.textLabel.textColor = [UIColor labelColor];
                cell.userInteractionEnabled = YES;
            }
        } else if (indexPath.row == 5) {
            cell.textLabel.text = @"Use own email server";
            cell.accessoryType = UITableViewCellAccessoryNone;
            UISwitch *useOwnEmailServerSwitch = [[UISwitch alloc] init];
            useOwnEmailServerSwitch.on = self.useOwnEmailServerEnabled;
            [useOwnEmailServerSwitch addTarget:self action:@selector(useOwnEmailServerSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = useOwnEmailServerSwitch;
            useOwnEmailServerSwitch.enabled = YES;
            cell.textLabel.textColor = [UIColor labelColor];
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 6 && self.useOwnEmailServerEnabled && self.isEmailServerSectionExpanded) {
                    cell.textLabel.text = @"Server Address";
                    cell.detailTextLabel.text = self.emailServerAddress;
                    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
                    cell.userInteractionEnabled = YES;
        } else if (indexPath.row == 7 && self.useOwnEmailServerEnabled && self.isEmailServerSectionExpanded) {
            cell.textLabel.text = @"Test own server";
            cell.textLabel.textColor = [UIColor systemBlueColor];
            cell.detailTextLabel.text = nil;
            cell.accessoryType = UITableViewCellAccessoryNone;
            cell.userInteractionEnabled = YES;
        }
    } else if (indexPath.section == 1) {
        NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
        if (indexPath.row == 0) {
            cell.textLabel.text = @"Manage Presets";
            cell.detailTextLabel.text = nil;
            cell.accessoryType = self.isPresetsSectionExpanded ? UITableViewCellAccessoryNone : UITableViewCellAccessoryDisclosureIndicator;
            cell.userInteractionEnabled = YES;
            NSLog(@"Manage Presets cell configured, tappable: YES");
        } else if (indexPath.row <= presetKeys.count) {
            NSString *presetKey = presetKeys[indexPath.row - 1];
            cell.textLabel.text = presetKey;
            cell.detailTextLabel.text = nil;
            cell.imageView.image = nil;
            cell.userInteractionEnabled = YES;
        } else if (indexPath.row == presetKeys.count + 1) {
            cell.textLabel.text = nil;
            cell.detailTextLabel.text = nil;
            cell.imageView.image = [UIImage systemImageNamed:@"plus.circle.fill"];
            cell.imageView.tintColor = [UIColor systemGreenColor];
            cell.accessoryType = UITableViewCellAccessoryNone;
            cell.userInteractionEnabled = YES;
        }
    } else if (indexPath.section == 2) {
        cell.textLabel.text = @"Upgrade to Premium";
        cell.textLabel.textColor = [UIColor systemBlueColor];
        cell.textLabel.textAlignment = NSTextAlignmentCenter;
        cell.accessoryType = UITableViewCellAccessoryNone;
        cell.userInteractionEnabled = YES;
    }

    return cell;
}
#pragma mark - UITableView Delegate

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    NSLog(@"Tapped section %ld, row %ld", (long)indexPath.section, (long)indexPath.row);
    if (indexPath.section == 0) {
        if (indexPath.row == 0) {
            NSLog(@"Tapped Resolution row");
            [self showResolutionPicker];
        } else if (indexPath.row == 1) {
            NSLog(@"Tapped Detect objects row");
            [self showYoloIndexesPicker];
        } else if (indexPath.row == 2 && ([[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"] || self.useOwnEmailServerEnabled)) {
            NSLog(@"Tapped Email Address row");
            [self showEmailInputDialog];
        } else if (indexPath.row == 6 && self.useOwnEmailServerEnabled && self.isEmailServerSectionExpanded) {
                    NSLog(@"Tapped Server Address row");
                    [self showEmailServerAddressInputDialog];
        } else if (indexPath.row == 7 && self.useOwnEmailServerEnabled && self.isEmailServerSectionExpanded) {
            NSLog(@"Tapped Test own server row");
            [self testEmailServer];
        }
    } else if (indexPath.section == 1) {
        NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
        if (indexPath.row == 0) {
            NSLog(@"Tapped Manage Presets row");
            self.isPresetsSectionExpanded = !self.isPresetsSectionExpanded;
            [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:1] withRowAnimation:UITableViewRowAnimationAutomatic];
        } else if (indexPath.row <= presetKeys.count) {
            NSString *presetKey = presetKeys[indexPath.row - 1];
            NSLog(@"Tapped preset: %@", presetKey);
            NSArray *selectedIndexes = [[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"][presetKey];
            [self showNumberSelectionForPreset:presetKey selectedIndexes:selectedIndexes];
        } else {
            NSLog(@"Tapped Add New Preset row");
            [self showAddPresetDialog];
        }
    } else if (indexPath.section == 2) {
        NSLog(@"Tapped Upgrade to Premium row");
        [[StoreManager sharedInstance] fetchAndPurchaseProduct];
    }
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

#pragma mark - Use Own Email Server

- (void)useOwnEmailServerSwitchToggled:(UISwitch *)sender {
    self.useOwnEmailServerEnabled = sender.on;
    self.isEmailServerSectionExpanded = sender.on; // Expand or collapse the section based on toggle state
    
    // Save the toggle state to NSUserDefaults
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    [defaults setBool:self.useOwnEmailServerEnabled forKey:@"use_own_email_server_enabled"];
    [defaults synchronize];
    
    // Reload section 0 to show or hide the expandable fields
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:0] withRowAnimation:UITableViewRowAnimationAutomatic];
    
    // Log the change
    NSLog(@"Use own email server: %@", self.useOwnEmailServerEnabled ? @"Enabled" : @"Disabled");
}

- (void)showEmailServerAddressInputDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Enter Server Address"
                                                                   message:@"Please enter the address of your email server."
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Server Address";
        textField.text = self.emailServerAddress; // Pre-fill with existing address
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:@"Save"
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *address = alert.textFields.firstObject.text;
        if (address.length > 0) {
            self.emailServerAddress = address;
            // Save the address to NSUserDefaults
            NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
            [defaults setObject:address forKey:@"own_email_server_address"];
            [defaults synchronize];
            // Reload the table view to show the updated address
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
    // You might want to add some feedback here, like an alert showing success/failure
    UIAlertController *resultAlert = [UIAlertController alertControllerWithTitle:@"Test Initiated"
                                                                        message:@"Test email has been initiated. Check your server logs for results."
                                                                 preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK"
                                                       style:UIAlertActionStyleDefault
                                                     handler:nil];
    [resultAlert addAction:okAction];
    [self presentViewController:resultAlert animated:YES completion:nil];
}

- (void)encryptEmailDataSwitchToggled:(UISwitch *)sender {
    if (sender.on) {
        // Check if a password already exists in the secrets manager
        NSString *password = [self retrievePasswordFromSecretsManager];
        if (!password) {
            // No password exists, prompt the user to set one
            [self promptForPasswordWithCompletion:^(BOOL success) {
                if (success) {
                    // Password set successfully, enable the toggle
                    self.encryptEmailDataEnabled = YES;
                    [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"encrypt_email_data_enabled"];
                    [[NSUserDefaults standardUserDefaults] synchronize];
                } else {
                    // User canceled, turn the toggle off
                    sender.on = NO;
                }
            }];
        } else {
            // Password exists, enable the toggle
            self.encryptEmailDataEnabled = YES;
            [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"encrypt_email_data_enabled"];
            [[NSUserDefaults standardUserDefaults] synchronize];
        }
    } else {
        // Turn off the toggle
        self.encryptEmailDataEnabled = NO;
        [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"encrypt_email_data_enabled"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

- (void)promptForPasswordWithCompletion:(void (^)(BOOL success))completion {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Set Password"
                                                                   message:@"Enter a password to encrypt email data"
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
            // Save the password to the secrets manager
            [self savePasswordToSecretsManager:password];
            completion(YES);
        } else {
            // Passwords do not match or are empty
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
    if (!password) {
        NSLog(@"Error: Password cannot be nil.");
        return;
    }
    
    NSError *error = nil;
    BOOL success = [[SecretManager sharedManager] saveEncryptionKey:password error:&error];
    
    if (!success) {
        NSLog(@"Failed to save password to Keychain: %@", error.localizedDescription);
    } else {
        NSLog(@"Password successfully saved to Keychain.");
    }
}

- (NSString *)retrievePasswordFromSecretsManager {
    NSString *storedKey = [[SecretManager sharedManager] getEncryptionKey];
    if (storedKey) {
        return storedKey; // Assuming only one password is stored
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

- (void)emailAlertsSwitchToggled:(UISwitch *)sender {
    if (sender.on) {
        // Check if an email address is already set
        NSString *email = [[NSUserDefaults standardUserDefaults] stringForKey:@"user_email"];
        if (!email || ![self isValidEmail:email]) {
            // No valid email address is set, prompt the user to enter one
            [self showEmailInputDialogWithCompletion:^(BOOL success) {
                if (success) {
                    // Email address was successfully set, keep the toggle on
                    self.sendEmailAlertsEnabled = YES;
                    [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"send_email_alerts_enabled"];
                    [[NSUserDefaults standardUserDefaults] synchronize];
                } else {
                    // User canceled or entered an invalid email, turn the toggle off
                    sender.on = NO;
                    self.sendEmailAlertsEnabled = NO;
                    [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"send_email_alerts_enabled"];
                    [[NSUserDefaults standardUserDefaults] synchronize];
                }
            }];
        } else {
            // Email address is valid, update the property and save the state
            self.sendEmailAlertsEnabled = YES;
            [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"send_email_alerts_enabled"];
            [[NSUserDefaults standardUserDefaults] synchronize];
        }
    } else {
        // Toggle is turned off, update the property and save the state
        self.sendEmailAlertsEnabled = NO;
        [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"send_email_alerts_enabled"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }

    // Log the change (optional)
    NSLog(@"Send Email Alerts: %@", self.sendEmailAlertsEnabled ? @"Enabled" : @"Disabled");
}

- (void)showEmailInputDialogWithCompletion:(void (^)(BOOL success))completion {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Enter Email Address"
                                                                   message:@"Please enter a valid email address to enable email alerts."
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Email Address";
        textField.keyboardType = UIKeyboardTypeEmailAddress;
        textField.text = [[NSUserDefaults standardUserDefaults] stringForKey:@"user_email"]; // Pre-fill with existing email
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:@"Save"
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *email = alert.textFields.firstObject.text;
        if ([self isValidEmail:email]) {
            // Save the email to UserDefaults
            [[NSUserDefaults standardUserDefaults] setObject:email forKey:@"user_email"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            
            // Reload the table view to show the updated email
            [self.tableView reloadData];
            
            // Call the completion handler with success
            completion(YES);
        } else {
            // Show an alert for invalid email
            [self showInvalidEmailAlert];
            
            // Call the completion handler with failure
            completion(NO);
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:^(UIAlertAction * _Nonnull action) {
        // Call the completion handler with failure
        completion(NO);
    }];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];
    
    [self presentViewController:alert animated:YES completion:nil];
}

- (BOOL)isValidEmail:(NSString *)email {
    NSString *emailRegex = @"[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}";
    NSPredicate *emailTest = [NSPredicate predicateWithFormat:@"SELF MATCHES %@", emailRegex];
    return [emailTest evaluateWithObject:email];
}

- (void)showInvalidEmailAlert {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Invalid Email"
                                                                   message:@"Please enter a valid email address."
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    UIAlertAction *okAction = [UIAlertAction actionWithTitle:@"OK"
                                                       style:UIAlertActionStyleDefault
                                                     handler:nil];
    
    [alert addAction:okAction];
    [self presentViewController:alert animated:YES completion:nil];
}

#pragma mark - UITableView Delegate

- (UITableViewCellEditingStyle)tableView:(UITableView *)tableView editingStyleForRowAtIndexPath:(NSIndexPath *)indexPath {
    if (indexPath.section == 1 && indexPath.row > 0 && indexPath.row <= [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys].count) {
        NSString *presetKey = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys][indexPath.row - 1];
        if (![presetKey isEqualToString:@"all"]) {
            return UITableViewCellEditingStyleDelete; // Enable swipe-to-delete for presets (except "all")
        }
    }
    return UITableViewCellEditingStyleNone;
}

- (void)tableView:(UITableView *)tableView commitEditingStyle:(UITableViewCellEditingStyle)editingStyle forRowAtIndexPath:(NSIndexPath *)indexPath {
    if (editingStyle == UITableViewCellEditingStyleDelete) {
        NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
        NSString *presetKey = presetKeys[indexPath.row - 1];

        // Prevent deletion of the "all" preset
        if ([presetKey isEqualToString:@"all"]) {
            NSLog(@"Cannot delete the 'all' preset.");
            return;
        }

        // Remove the preset
        NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
        NSMutableDictionary *yoloPresets = [[defaults objectForKey:@"yolo_presets"] mutableCopy];
        [yoloPresets removeObjectForKey:presetKey];
        [defaults setObject:yoloPresets forKey:@"yolo_presets"];

        // Check if the deleted preset was the currently selected one
        if ([self.selectedPresetKey isEqualToString:presetKey]) {
            // Switch to the "all" preset
            self.selectedPresetKey = @"all";
            [defaults setObject:@"all" forKey:@"yolo_preset_idx"];
        }

        [defaults synchronize];

        // Animate the deletion of the row
        [tableView beginUpdates];
        [tableView deleteRowsAtIndexPaths:@[indexPath] withRowAnimation:UITableViewRowAnimationAutomatic];
        [tableView endUpdates];

        // Reload both sections to ensure the UI is fully updated
        [self.tableView reloadSections:[NSIndexSet indexSetWithIndexesInRange:NSMakeRange(0, 2)] withRowAnimation:UITableViewRowAnimationAutomatic];
    }
}

- (void)showEmailInputDialog {
    NSLog(@"Entering showEmailInputDialog");
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Enter Email Address"
                                                                   message:@"Please enter a valid email address."
                                                            preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.placeholder = @"Email Address";
        textField.keyboardType = UIKeyboardTypeEmailAddress;
        textField.text = [[NSUserDefaults standardUserDefaults] stringForKey:@"user_email"];
    }];
    
    UIAlertAction *saveAction = [UIAlertAction actionWithTitle:@"Save"
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        NSString *email = alert.textFields.firstObject.text;
        NSLog(@"User entered email: %@", email);
        if ([self isValidEmail:email]) {
            [[NSUserDefaults standardUserDefaults] setObject:email forKey:@"user_email"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.tableView reloadData];
                NSLog(@"Table view reloaded with new email");
            });
        } else {
            [self showInvalidEmailAlert];
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:^(UIAlertAction * _Nonnull action) {
        NSLog(@"User canceled email input dialog");
    }];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];

    dispatch_async(dispatch_get_main_queue(), ^{
        NSLog(@"Presenting email input dialog");
        [self presentViewController:alert animated:YES completion:^{
            NSLog(@"Email input dialog presentation completed");
        }];
    });
}

#pragma mark - Resolution Picker

- (void)showResolutionPicker {
    NSLog(@"Entering showResolutionPicker");
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Select Resolution"
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];

    NSArray *resolutions = @[@"720p", @"1080p"];

    for (NSString *resolution in resolutions) {
        UIAlertAction *action = [UIAlertAction actionWithTitle:resolution
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
            NSLog(@"User selected resolution: %@", resolution);
            self.selectedResolution = resolution;
            SettingsManager *settingsManager = [SettingsManager sharedManager];
            if ([resolution isEqualToString:@"720p"]) {
                [settingsManager updateResolutionWithWidth:@"1280" height:@"720" textSize:@"2" preset:@"AVCaptureSessionPreset1280x720"];
            } else if ([resolution isEqualToString:@"1080p"]) {
                [settingsManager updateResolutionWithWidth:@"1920" height:@"1080" textSize:@"3" preset:@"AVCaptureSessionPreset1920x1080"];
            }
            NSLog(@"Updated SettingsManager: width=%@, height=%@, textSize=%@, preset=%@",
                  settingsManager.width, settingsManager.height, settingsManager.text_size, settingsManager.preset);
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.tableView reloadData];
                NSLog(@"Table view reloaded for resolution change");
            });
        }];
        [alert addAction:action];
    }

    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:^(UIAlertAction * _Nonnull action) {
        NSLog(@"User canceled resolution picker");
    }];
    [alert addAction:cancelAction];

    dispatch_async(dispatch_get_main_queue(), ^{
        NSLog(@"Presenting resolution picker");
        [self presentViewController:alert animated:YES completion:^{
            NSLog(@"Resolution picker presentation completed");
        }];
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
            NSLog(@"Selected YOLO indexes key: %@", self.selectedPresetKey);
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

- (void)showPresetManagementOptions {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Manage Presets"
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    UIAlertAction *addAction = [UIAlertAction actionWithTitle:@"Add Preset"
                                                        style:UIAlertActionStyleDefault
                                                      handler:^(UIAlertAction * _Nonnull action) {
        [self showAddPresetDialog];
    }];
    [alert addAction:addAction];
    UIAlertAction *editAction = [UIAlertAction actionWithTitle:@"Edit Preset"
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        [self showEditPresetDialog];
    }];
    [alert addAction:editAction];
    UIAlertAction *deleteAction = [UIAlertAction actionWithTitle:@"Delete Preset"
                                                           style:UIAlertActionStyleDestructive
                                                         handler:^(UIAlertAction * _Nonnull action) {
        [self showDeletePresetDialog];
    }];
    [alert addAction:deleteAction];
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    [alert addAction:cancelAction];
    [self presentViewController:alert animated:YES completion:nil];
}

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

- (void)showEditPresetDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Edit Preset"
                                                                   message:@"Select a preset to edit"
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
    for (NSString *presetKey in presetKeys) {
        UIAlertAction *action = [UIAlertAction actionWithTitle:presetKey
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
            NSArray *selectedIndexes = [[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"][presetKey];
            [self showNumberSelectionForPreset:presetKey selectedIndexes:selectedIndexes];
        }];
        [alert addAction:action];
    }
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    [alert addAction:cancelAction];
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

- (void)showDeletePresetDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Delete Preset"
                                                                   message:@"Select a preset to delete"
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
    for (NSString *presetKey in presetKeys) {
        if (![presetKey isEqualToString:@"all"]) { // Prevent deletion of the "all" preset
            UIAlertAction *action = [UIAlertAction actionWithTitle:presetKey
                                                             style:UIAlertActionStyleDestructive
                                                           handler:^(UIAlertAction * _Nonnull action) {
                NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
                NSMutableDictionary *yoloPresets = [[defaults objectForKey:@"yolo_presets"] mutableCopy];

                if (yoloPresets) [yoloPresets removeObjectForKey:presetKey];
                [defaults setObject:yoloPresets forKey:@"yolo_presets"];
                if ([self.selectedPresetKey isEqualToString:presetKey]) {
                    self.selectedPresetKey = @"all";
                    [defaults setObject:@"all" forKey:@"yolo_preset_idx"];
                }
                [defaults synchronize];
                [self.tableView reloadSections:[NSIndexSet indexSetWithIndexesInRange:NSMakeRange(0, 2)] withRowAnimation:UITableViewRowAnimationAutomatic];
            }];
            [alert addAction:action];
        }
    }
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    [alert addAction:cancelAction];
    [self presentViewController:alert animated:YES completion:nil];
}
@end
