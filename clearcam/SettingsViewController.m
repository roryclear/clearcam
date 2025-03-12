#import "SettingsViewController.h"
#import "SettingsManager.h"
#import "SecretManager.h"
#import "NumberSelectionViewController.h"

@interface SettingsViewController () <UITableViewDelegate, UITableViewDataSource>

@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSString *selectedResolution;
@property (nonatomic, strong) NSString *selectedPresetKey; // For YOLO indexes key
@property (nonatomic, assign) BOOL isPresetsSectionExpanded; // Track if presets section is expanded
@property (nonatomic, assign) BOOL sendEmailAlertsEnabled;
@property (nonatomic, assign) BOOL encryptEmailDataEnabled;

@end

@implementation SettingsViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    // Basic setup
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Settings";

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
    return 2; // One for general settings, one for presets
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    if (section == 0) {
        return 5; // Resolution, Detect objects, Email, Send Email Alerts, Encrypt Email Data
    } else if (section == 1) {
        // Presets section: 1 row for "Manage Presets" header, plus rows for each preset, plus "Add New +"
        NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
        return self.isPresetsSectionExpanded ? (presetKeys.count + 2) : 1;
    }
    return 0;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"SettingsCell"];
    
    if (!cell) {
        cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleValue1 reuseIdentifier:@"SettingsCell"];
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
    }

    // Make sure cell adapts to Dark Mode
    cell.backgroundColor = [UIColor secondarySystemBackgroundColor];
    cell.textLabel.textColor = [UIColor labelColor];
    cell.detailTextLabel.textColor = [UIColor secondaryLabelColor];

    // Clear cell state to prevent lingering text or images
    cell.textLabel.text = nil;
    cell.detailTextLabel.text = nil;
    cell.imageView.image = nil;
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;

    // Clear accessoryView for all cells initially
    cell.accessoryView = nil;

    if (indexPath.section == 0) {
        if (indexPath.row == 0) {
            // Resolution
            cell.textLabel.text = @"Resolution";
            cell.detailTextLabel.text = self.selectedResolution; // Show current resolution
        } else if (indexPath.row == 1) {
            // Detect objects
            cell.textLabel.text = @"Detect objects";
            cell.detailTextLabel.text = self.selectedPresetKey; // Show current YOLO preset
        } else if (indexPath.row == 2) {
            // Email Address
            cell.textLabel.text = @"Email Address";
            NSString *email = [[NSUserDefaults standardUserDefaults] stringForKey:@"user_email"];
            cell.detailTextLabel.text = email ?: @"Not set"; // Show current email or "Not set"
        } else if (indexPath.row == 3) {
            // Send Email Alerts
            cell.textLabel.text = @"Send Email Alerts";
            cell.accessoryType = UITableViewCellAccessoryNone; // No disclosure indicator for switches

            // Add a UISwitch to the cell
            UISwitch *emailAlertsSwitch = [[UISwitch alloc] init];
            emailAlertsSwitch.on = self.sendEmailAlertsEnabled;
            [emailAlertsSwitch addTarget:self action:@selector(emailAlertsSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = emailAlertsSwitch;
        } else if (indexPath.row == 4) {
            // Encrypt Email Data
            cell.textLabel.text = @"Encrypt Email Data (Recommended)";
            cell.accessoryType = UITableViewCellAccessoryNone; // No disclosure indicator for switches

            // Add a UISwitch to the cell
            UISwitch *encryptEmailDataSwitch = [[UISwitch alloc] init];
            encryptEmailDataSwitch.on = self.encryptEmailDataEnabled;
            [encryptEmailDataSwitch addTarget:self action:@selector(encryptEmailDataSwitchToggled:) forControlEvents:UIControlEventValueChanged];
            cell.accessoryView = encryptEmailDataSwitch;
        }
    } else if (indexPath.section == 1) {
        // Presets section (unchanged)
        NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
        if (indexPath.row == 0) {
            // "Manage Presets" header row
            cell.textLabel.text = @"Manage Presets";
            cell.detailTextLabel.text = nil;
            cell.accessoryType = self.isPresetsSectionExpanded ? UITableViewCellAccessoryNone : UITableViewCellAccessoryDisclosureIndicator;
        } else if (indexPath.row <= presetKeys.count) {
            // Preset rows (including "all")
            NSString *presetKey = presetKeys[indexPath.row - 1];
            cell.textLabel.text = presetKey;
            cell.detailTextLabel.text = nil;
            cell.imageView.image = nil; // Clear any existing image
        } else if (indexPath.row == presetKeys.count + 1) {
            // "Add New +" row (only show if expanded)
            cell.textLabel.text = nil; // Clear text
            cell.detailTextLabel.text = nil; // Clear detail text
            cell.imageView.image = [UIImage systemImageNamed:@"plus.circle.fill"]; // Use the system + icon
            cell.imageView.tintColor = [UIColor systemGreenColor]; // Set the icon color to green
            cell.accessoryType = UITableViewCellAccessoryNone;
        }
    }

    return cell;
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
    BOOL success = [[SecretManager sharedManager] saveKey:password error:&error];
    
    if (!success) {
        NSLog(@"Failed to save password to Keychain: %@", error.localizedDescription);
    } else {
        NSLog(@"Password successfully saved to Keychain.");
    }
}

- (NSString *)retrievePasswordFromSecretsManager {
    NSArray<NSString *> *storedKeys = [[SecretManager sharedManager] getAllStoredKeys];
    
    if (storedKeys.count > 0) {
        return storedKeys.firstObject; // Assuming only one password is stored
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

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    if (indexPath.section == 0) {
        if (indexPath.row == 0) {
            [self showResolutionPicker];
        } else if (indexPath.row == 1) {
            [self showYoloIndexesPicker];
        } else if (indexPath.row == 2) {
            [self showEmailInputDialog];
        }
        // Ignore taps on the "Send Email Alerts" row (indexPath.row == 3)
    } else if (indexPath.section == 1) {
        NSArray *presetKeys = [[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"] allKeys];
        if (indexPath.row == 0) {
            // Toggle presets section expansion
            self.isPresetsSectionExpanded = !self.isPresetsSectionExpanded;
            [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:1] withRowAnimation:UITableViewRowAnimationAutomatic];
        } else if (indexPath.row <= presetKeys.count) {
            // Edit preset
            NSString *presetKey = presetKeys[indexPath.row - 1];
            NSArray *selectedIndexes = [[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"][presetKey];
            [self showNumberSelectionForPreset:presetKey selectedIndexes:selectedIndexes];
        } else {
            // Add new preset
            [self showAddPresetDialog];
        }
    }

    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

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
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Enter Email Address"
                                                                   message:@"Please enter a valid email address."
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
            [[NSUserDefaults standardUserDefaults] setObject:email forKey:@"user_email"]; // Save email to UserDefaults
            [[NSUserDefaults standardUserDefaults] synchronize];
            [self.tableView reloadData]; // Refresh the table view to show the updated email
        } else {
            [self showInvalidEmailAlert];
        }
    }];
    
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel
                                                         handler:nil];
    
    [alert addAction:saveAction];
    [alert addAction:cancelAction];
    
    [self presentViewController:alert animated:YES completion:nil];
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
            // Update the selected resolution
            self.selectedResolution = resolution;
            
            // Log the selected resolution
            NSLog(@"Selected resolution: %@", self.selectedResolution);
            
            // Update SettingsManager with the new resolution
            SettingsManager *settingsManager = [SettingsManager sharedManager];
            if ([self.selectedResolution isEqualToString:@"720p"]) {
                [settingsManager updateResolutionWithWidth:@"1280" height:@"720" textSize:@"2" preset:@"AVCaptureSessionPreset1280x720"];
            } else if ([self.selectedResolution isEqualToString:@"1080p"]) {
                [settingsManager updateResolutionWithWidth:@"1920" height:@"1080" textSize:@"3" preset:@"AVCaptureSessionPreset1920x1080"];
            }
            
            // Log the updated resolution settings from SettingsManager
            NSLog(@"Updated resolution settings: %@x%@, text size: %@, preset: %@",
                  settingsManager.width, settingsManager.height, settingsManager.text_size, settingsManager.preset);
            
            // Reload the table view to update the UI
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

    // Add Preset
    UIAlertAction *addAction = [UIAlertAction actionWithTitle:@"Add Preset"
                                                        style:UIAlertActionStyleDefault
                                                      handler:^(UIAlertAction * _Nonnull action) {
        [self showAddPresetDialog];
    }];
    [alert addAction:addAction];

    // Edit Preset
    UIAlertAction *editAction = [UIAlertAction actionWithTitle:@"Edit Preset"
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
        [self showEditPresetDialog];
    }];
    [alert addAction:editAction];

    // Delete Preset
    UIAlertAction *deleteAction = [UIAlertAction actionWithTitle:@"Delete Preset"
                                                           style:UIAlertActionStyleDestructive
                                                         handler:^(UIAlertAction * _Nonnull action) {
        [self showDeletePresetDialog];
    }];
    [alert addAction:deleteAction];

    // Cancel
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
    SettingsManager *settingsManager = [SettingsManager sharedManager];
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

        // Update the dictionary with the new key-value pair
        yoloPresets[presetKey] = selectedIndexes;

        // Save the updated dictionary back to NSUserDefaults
        [defaults setObject:yoloPresets forKey:@"yolo_presets"];

        // Automatically select the new preset
        self.selectedPresetKey = presetKey;
        [defaults setObject:presetKey forKey:@"yolo_preset_idx"];

        // Synchronize to save changes immediately
        [defaults synchronize];

        // Reload the table view to update the UI
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

                if (yoloPresets) {
                    [yoloPresets removeObjectForKey:presetKey];
                }

                // Save the updated dictionary back to NSUserDefaults
                [defaults setObject:yoloPresets forKey:@"yolo_presets"];

                // Check if the deleted preset was the currently selected one
                if ([self.selectedPresetKey isEqualToString:presetKey]) {
                    // Switch to the "all" preset
                    self.selectedPresetKey = @"all";
                    [defaults setObject:@"all" forKey:@"yolo_preset_idx"];
                }

                // Synchronize to save changes immediately
                [defaults synchronize];

                // Reload both sections to ensure the UI is fully updated
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
