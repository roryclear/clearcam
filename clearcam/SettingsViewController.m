#import "SettingsViewController.h"
#import "SettingsManager.h"
#import "NumberSelectionViewController.h" // Import the new class

@interface SettingsViewController () <UITableViewDelegate, UITableViewDataSource>

@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSString *selectedResolution;
@property (nonatomic, strong) NSString *selectedPresetKey; // For YOLO indexes key

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
    self.selectedPresetKey = [[NSUserDefaults standardUserDefaults] stringForKey:@"yolo_indexes_key"] ?: @"all"; // Default to "all" if no key is saved

    // Create table view
    self.tableView = [[UITableView alloc] initWithFrame:self.view.bounds style:UITableViewStyleInsetGrouped];
    self.tableView.delegate = self;
    self.tableView.dataSource = self;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor]; // Matches dark/light mode
    [self.view addSubview:self.tableView];
}

#pragma mark - UITableView DataSource

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return 1;
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return 3; // One for resolution, one for YOLO indexes key, one for managing presets
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

    if (indexPath.row == 0) {
        cell.textLabel.text = @"Resolution";
        cell.detailTextLabel.text = self.selectedResolution;
    } else if (indexPath.row == 1) {
        cell.textLabel.text = @"Detect objects";
        cell.detailTextLabel.text = self.selectedPresetKey;
    } else if (indexPath.row == 2) {
        cell.textLabel.text = @"Manage Presets";
        cell.detailTextLabel.text = nil; // No detail text for this row
    }

    return cell;
}

#pragma mark - UITableView Delegate

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    if (indexPath.row == 0) {
        [self showResolutionPicker];
    } else if (indexPath.row == 1) {
        [self showYoloIndexesPicker];
    } else if (indexPath.row == 2) {
        [self showPresetManagementOptions];
    }

    [tableView deselectRowAtIndexPath:indexPath animated:YES];
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
    NSArray *presetKeys = [settingsManager.presets allKeys];
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
    NSArray *presetKeys = [settingsManager.presets allKeys];
    for (NSString *presetKey in presetKeys) {
        UIAlertAction *action = [UIAlertAction actionWithTitle:presetKey
                                                         style:UIAlertActionStyleDefault
                                                       handler:^(UIAlertAction * _Nonnull action) {
            NSArray *selectedIndexes = settingsManager.presets[presetKey];
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
        SettingsManager *settingsManager = [SettingsManager sharedManager];
        settingsManager.presets[presetKey] = selectedIndexes;
        [[NSUserDefaults standardUserDefaults] synchronize];
        [self.tableView reloadData];
    };
    [self.navigationController pushViewController:numberSelectionVC animated:YES];
}

- (void)showDeletePresetDialog {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Delete Preset"
                                                                   message:@"Select a preset to delete"
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    SettingsManager *settingsManager = [SettingsManager sharedManager];
    NSArray *presetKeys = [settingsManager.presets allKeys];
    for (NSString *presetKey in presetKeys) {
        if (![presetKey isEqualToString:@"all"]) { // Prevent deletion of the "all" preset
            UIAlertAction *action = [UIAlertAction actionWithTitle:presetKey
                                                             style:UIAlertActionStyleDestructive
                                                           handler:^(UIAlertAction * _Nonnull action) {
                [settingsManager.presets removeObjectForKey:presetKey];
                [[NSUserDefaults standardUserDefaults] synchronize];
                [self.tableView reloadData];
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
