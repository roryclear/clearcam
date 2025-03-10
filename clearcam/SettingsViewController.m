#import "SettingsViewController.h"
#import "SettingsManager.h" // Import SettingsManager

@interface SettingsViewController () <UITableViewDelegate, UITableViewDataSource>

@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSString *selectedResolution;

@end

@implementation SettingsViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    // Basic setup
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Settings";

    // Default resolution
    self.selectedResolution = @"1080p";

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
    return 1; // More settings can be added later
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
    }

    return cell;
}

#pragma mark - UITableView Delegate

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    if (indexPath.row == 0) {
        [self showResolutionPicker];
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

@end
