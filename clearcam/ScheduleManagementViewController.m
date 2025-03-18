#import "ScheduleManagementViewController.h"

@interface ScheduleEditorViewController : UITableViewController
@property (nonatomic, strong) NSMutableDictionary *schedule;
@property (nonatomic, copy) void (^completionHandler)(NSDictionary *schedule);
@end

@implementation ScheduleManagementViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.title = @"Manage Schedules";
    self.tableView.allowsMultipleSelection = NO;
    [self.tableView registerClass:[UITableViewCell class] forCellReuseIdentifier:@"ScheduleCell"];
    
    UIBarButtonItem *addButton = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemAdd target:self action:@selector(addScheduleTapped:)];
    self.navigationItem.rightBarButtonItem = addButton;
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return self.emailSchedules.count;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"ScheduleCell" forIndexPath:indexPath];
    NSDictionary *schedule = self.emailSchedules[indexPath.row];
    
    cell.textLabel.text = [self scheduleDescriptionFor:schedule];
    cell.detailTextLabel.text = nil;
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
    
    UISwitch *toggleSwitch = [[UISwitch alloc] init];
    toggleSwitch.on = [schedule[@"enabled"] boolValue];
    toggleSwitch.tag = indexPath.row;
    [toggleSwitch addTarget:self action:@selector(toggleSwitchChanged:) forControlEvents:UIControlEventValueChanged];
    cell.accessoryView = toggleSwitch;
    
    return cell;
}

- (NSString *)scheduleDescriptionFor:(NSDictionary *)schedule {
    NSArray *days = schedule[@"days"];
    NSNumber *startHour = schedule[@"startHour"];
    NSNumber *startMinute = schedule[@"startMinute"];
    NSNumber *endHour = schedule[@"endHour"];
    NSNumber *endMinute = schedule[@"endMinute"];
    
    NSString *daysString = days.count == 7 ? @"Every day" : [days componentsJoinedByString:@", "];
    NSString *timeString = [NSString stringWithFormat:@"%02ld:%02ld-%02ld:%02ld",
                           [startHour longValue], [startMinute longValue],
                           [endHour longValue], [endMinute longValue]];
    return [NSString stringWithFormat:@"%@ %@", daysString, timeString];
}

- (void)toggleSwitchChanged:(UISwitch *)sender {
    NSInteger index = sender.tag;
    NSMutableDictionary *schedule = [self.emailSchedules[index] mutableCopy];
    schedule[@"enabled"] = @(sender.on);
    self.emailSchedules[index] = schedule;
    [self.tableView reloadRowsAtIndexPaths:@[[NSIndexPath indexPathForRow:index inSection:0]] withRowAnimation:UITableViewRowAnimationNone];
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    ScheduleEditorViewController *editorVC = [[ScheduleEditorViewController alloc] init];
    editorVC.schedule = [self.emailSchedules[indexPath.row] mutableCopy];
    editorVC.completionHandler = ^(NSDictionary *updatedSchedule) {
        self.emailSchedules[indexPath.row] = updatedSchedule;
        [self.tableView reloadRowsAtIndexPaths:@[indexPath] withRowAnimation:UITableViewRowAnimationNone];
    };
    [self.navigationController pushViewController:editorVC animated:YES];
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

- (UITableViewCellEditingStyle)tableView:(UITableView *)tableView editingStyleForRowAtIndexPath:(NSIndexPath *)indexPath {
    NSDictionary *schedule = self.emailSchedules[indexPath.row];
    return ([schedule[@"days"] count] == 7 && [schedule[@"startHour"] isEqual:@0] && [schedule[@"startMinute"] isEqual:@0] && [schedule[@"endHour"] isEqual:@23] && [schedule[@"endMinute"] isEqual:@59]) ? UITableViewCellEditingStyleNone : UITableViewCellEditingStyleDelete;
}

- (void)tableView:(UITableView *)tableView commitEditingStyle:(UITableViewCellEditingStyle)editingStyle forRowAtIndexPath:(NSIndexPath *)indexPath {
    if (editingStyle == UITableViewCellEditingStyleDelete) {
        [self.emailSchedules removeObjectAtIndex:indexPath.row];
        [tableView deleteRowsAtIndexPaths:@[indexPath] withRowAnimation:UITableViewRowAnimationAutomatic];
    }
}

- (void)addScheduleTapped:(id)sender {
    NSMutableDictionary *newSchedule = [@{
        @"days": @[@"Mon", @"Tue", @"Wed", @"Thu", @"Fri", @"Sat", @"Sun"],
        @"startHour": @0,
        @"startMinute": @0,
        @"endHour": @23,
        @"endMinute": @59,
        @"enabled": @YES
    } mutableCopy];
    [self.emailSchedules addObject:newSchedule];
    NSInteger newIndex = self.emailSchedules.count - 1;
    [self.tableView insertRowsAtIndexPaths:@[[NSIndexPath indexPathForRow:newIndex inSection:0]] withRowAnimation:UITableViewRowAnimationAutomatic];
    
    ScheduleEditorViewController *editorVC = [[ScheduleEditorViewController alloc] init];
    editorVC.schedule = newSchedule;
    editorVC.completionHandler = ^(NSDictionary *updatedSchedule) {
        self.emailSchedules[newIndex] = updatedSchedule;
        [self.tableView reloadRowsAtIndexPaths:@[[NSIndexPath indexPathForRow:newIndex inSection:0]] withRowAnimation:UITableViewRowAnimationNone];
    };
    [self.navigationController pushViewController:editorVC animated:YES];
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    if (self.completionHandler) {
        self.completionHandler(self.emailSchedules);
    }
}

@end

@implementation ScheduleEditorViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"Edit Schedule";
    [self.tableView registerClass:[UITableViewCell class] forCellReuseIdentifier:@"EditorCell"];
    
    UIBarButtonItem *doneButton = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemDone target:self action:@selector(doneTapped:)];
    self.navigationItem.rightBarButtonItem = doneButton;
}

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return 2; // Days and Time Window
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    if (section == 0) {
        return 7; // One row per day
    } else {
        return 2; // Start Time and End Time
    }
}

- (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    return section == 0 ? @"Repeat" : @"Time Window";
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"EditorCell" forIndexPath:indexPath];
    
    cell.textLabel.text = nil;
    cell.detailTextLabel.text = nil;
    cell.accessoryType = UITableViewCellAccessoryNone;
    cell.accessoryView = nil;
    
    if (indexPath.section == 0) {
        NSArray *allDays = @[@"Mon", @"Tue", @"Wed", @"Thu", @"Fri", @"Sat", @"Sun"];
        NSString *day = allDays[indexPath.row];
        cell.textLabel.text = day;
        cell.accessoryType = [self.schedule[@"days"] containsObject:day] ? UITableViewCellAccessoryCheckmark : UITableViewCellAccessoryNone;
    } else if (indexPath.section == 1) {
        if (indexPath.row == 0) {
            cell.textLabel.text = @"Start Time";
            NSInteger startHour = [self.schedule[@"startHour"] integerValue];
            NSInteger startMinute = [self.schedule[@"startMinute"] integerValue];
            cell.detailTextLabel.text = [NSString stringWithFormat:@"%02ld:%02ld", (long)startHour, (long)startMinute];
            cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        } else if (indexPath.row == 1) {
            cell.textLabel.text = @"End Time";
            NSInteger endHour = [self.schedule[@"endHour"] integerValue];
            NSInteger endMinute = [self.schedule[@"endMinute"] integerValue];
            cell.detailTextLabel.text = [NSString stringWithFormat:@"%02ld:%02ld", (long)endHour, (long)endMinute];
            cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        }
    }
    
    return cell;
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    if (indexPath.section == 0) {
        NSArray *allDays = @[@"Mon", @"Tue", @"Wed", @"Thu", @"Fri", @"Sat", @"Sun"];
        NSString *day = allDays[indexPath.row];
        NSMutableArray *days = [self.schedule[@"days"] mutableCopy];
        if ([days containsObject:day]) {
            [days removeObject:day];
        } else {
            [days addObject:day];
        }
        self.schedule[@"days"] = days;
        [tableView reloadRowsAtIndexPaths:@[indexPath] withRowAnimation:UITableViewRowAnimationNone];
    } else if (indexPath.section == 1) {
        [self showTimePickerForRow:indexPath.row];
    }
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

- (void)showTimePickerForRow:(NSInteger)row {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:nil
                                                                  message:nil
                                                           preferredStyle:UIAlertControllerStyleActionSheet];
    
    UIDatePicker *picker = [[UIDatePicker alloc] init];
    picker.datePickerMode = UIDatePickerModeTime;
    picker.translatesAutoresizingMaskIntoConstraints = NO;
    picker.tag = row; // Use tag to track which row (0 = start, 1 = end)
    
    if (row == 0) {
        NSDateComponents *startComponents = [[NSDateComponents alloc] init];
        startComponents.hour = [self.schedule[@"startHour"] integerValue];
        startComponents.minute = [self.schedule[@"startMinute"] integerValue];
        picker.date = [[NSCalendar currentCalendar] dateFromComponents:startComponents];
    } else {
        NSDateComponents *endComponents = [[NSDateComponents alloc] init];
        endComponents.hour = [self.schedule[@"endHour"] integerValue];
        endComponents.minute = [self.schedule[@"endMinute"] integerValue];
        picker.date = [[NSCalendar currentCalendar] dateFromComponents:endComponents];
    }
    
    [picker addTarget:self action:@selector(timePickerChanged:) forControlEvents:UIControlEventValueChanged];
    
    [alert.view addSubview:picker];
    [NSLayoutConstraint activateConstraints:@[
        [picker.topAnchor constraintEqualToAnchor:alert.view.topAnchor constant:10],
        [picker.leadingAnchor constraintEqualToAnchor:alert.view.leadingAnchor constant:10],
        [picker.trailingAnchor constraintEqualToAnchor:alert.view.trailingAnchor constant:-10],
        [picker.heightAnchor constraintEqualToConstant:162]
    ]];
    
    [alert.view.heightAnchor constraintGreaterThanOrEqualToConstant:200].active = YES;
    
    [alert addAction:[UIAlertAction actionWithTitle:@"Done" style:UIAlertActionStyleCancel handler:nil]];
    
    [self presentViewController:alert animated:YES completion:^{
        [self timePickerChanged:picker]; // Apply initial value
    }];
}

- (void)timePickerChanged:(UIDatePicker *)picker {
    NSDateComponents *components = [[NSCalendar currentCalendar] components:(NSCalendarUnitHour | NSCalendarUnitMinute) fromDate:picker.date];
    NSInteger row = picker.tag;
    if (row == 0) {
        self.schedule[@"startHour"] = @(components.hour);
        self.schedule[@"startMinute"] = @(components.minute);
    } else {
        self.schedule[@"endHour"] = @(components.hour);
        self.schedule[@"endMinute"] = @(components.minute);
    }
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:1] withRowAnimation:UITableViewRowAnimationNone];
}

- (void)doneTapped:(id)sender {
    NSInteger startHour = [self.schedule[@"startHour"] integerValue];
    NSInteger startMinute = [self.schedule[@"startMinute"] integerValue];
    NSInteger endHour = [self.schedule[@"endHour"] integerValue];
    NSInteger endMinute = [self.schedule[@"endMinute"] integerValue];
    
    if ([self.schedule[@"days"] count] == 0) {
        UIAlertController *errorAlert = [UIAlertController alertControllerWithTitle:@"Error"
                                                                           message:@"Please select at least one day."
                                                                    preferredStyle:UIAlertControllerStyleAlert];
        [errorAlert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
        [self presentViewController:errorAlert animated:YES completion:nil];
        return;
    }
    
    if (!((startHour < endHour) || (startHour == endHour && startMinute < endMinute))) {
        UIAlertController *errorAlert = [UIAlertController alertControllerWithTitle:@"Invalid Time"
                                                                           message:@"Start time must be before end time."
                                                                    preferredStyle:UIAlertControllerStyleAlert];
        [errorAlert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
        [self presentViewController:errorAlert animated:YES completion:nil];
        return;
    }
    
    if (self.completionHandler) {
        self.completionHandler(self.schedule);
    }
    [self.navigationController popViewControllerAnimated:YES];
}

@end
