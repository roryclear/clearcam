#import "SettingsViewController.h"

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
            self.selectedResolution = resolution;
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
