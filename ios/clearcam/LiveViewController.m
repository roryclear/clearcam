#import "LiveViewController.h"
#import "StoreManager.h"
#import "DeviceStreamViewController.h" // Import DeviceStreamViewController

@interface LiveViewController () <UITableViewDelegate, UITableViewDataSource>
@property (nonatomic, strong) NSMutableArray<NSString *> *deviceNames;
@property (nonatomic, strong) NSMutableArray<NSNumber *> *alertsOnStates;
@property (nonatomic, strong) NSURLSession *session;
@property (nonatomic, strong) UIRefreshControl *refreshControl;
@end

@implementation LiveViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"Live Cameras";
    self.deviceNames = [NSMutableArray array];
    self.alertsOnStates = [NSMutableArray array];


    self.session = [NSURLSession sessionWithConfiguration:[NSURLSessionConfiguration defaultSessionConfiguration]];
    [self setupTableView];
    [[StoreManager sharedInstance] verifySubscriptionWithCompletionIfSubbed:^(BOOL isActive, NSDate *expiryDate) {
    [self fetchLiveDevices];
    }];
}

- (void)setupTableView {
    self.tableView = [[UITableView alloc] initWithFrame:CGRectZero style:UITableViewStylePlain];
    self.tableView.delegate = self;
    self.tableView.dataSource = self;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor];
    [self.tableView registerClass:[UITableViewCell class] forCellReuseIdentifier:@"DeviceCell"];
    
    [self.view addSubview:self.tableView];
    
    self.tableView.translatesAutoresizingMaskIntoConstraints = NO;
    [NSLayoutConstraint activateConstraints:@[
        [self.tableView.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor],
        [self.tableView.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor],
        [self.tableView.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor],
        [self.tableView.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor]
    ]];
    self.refreshControl = [[UIRefreshControl alloc] init];
    [self.refreshControl addTarget:self action:@selector(refreshDevices) forControlEvents:UIControlEventValueChanged];
    self.tableView.refreshControl = self.refreshControl;
}

- (void)updateTableViewBackground {
    if (self.deviceNames.count == 0) {
        UILabel *messageLabel = [[UILabel alloc] initWithFrame:self.tableView.bounds];
        messageLabel.numberOfLines = 0;
        messageLabel.lineBreakMode = NSLineBreakByWordWrapping;

        // Title centered
        NSMutableParagraphStyle *centeredStyle = [[NSMutableParagraphStyle alloc] init];
        centeredStyle.alignment = NSTextAlignmentCenter;
        centeredStyle.paragraphSpacing = 12.0;

        NSDictionary *titleAttrs = @{
            NSFontAttributeName: [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline],
            NSForegroundColorAttributeName: [UIColor labelColor],
            NSParagraphStyleAttributeName: centeredStyle
        };

        NSMutableAttributedString *message = [[NSMutableAttributedString alloc] init];
        [message appendAttributedString:[[NSAttributedString alloc] initWithString:NSLocalizedString(@"no_live_devices", "no live devices") attributes:titleAttrs]];

        // Steps with hanging indent, left-aligned
        NSMutableParagraphStyle *stepsStyle = [[NSMutableParagraphStyle alloc] init];
        stepsStyle.alignment = NSTextAlignmentLeft;
        stepsStyle.headIndent = 20.0;
        stepsStyle.tailIndent = -20.0;
        stepsStyle.firstLineHeadIndent = 20.0;
        stepsStyle.paragraphSpacing = 8.0;

        NSDictionary *stepsAttrs = @{
            NSFontAttributeName: [UIFont preferredFontForTextStyle:UIFontTextStyleBody],
            NSForegroundColorAttributeName: [UIColor secondaryLabelColor],
            NSParagraphStyleAttributeName: stepsStyle
        };

        NSString *stepsText = NSLocalizedString(@"steps_text_live","steps on how to make live work.");

        [message appendAttributedString:[[NSAttributedString alloc] initWithString:stepsText attributes:stepsAttrs]];

        messageLabel.attributedText = message;
        [messageLabel sizeToFit];

        self.tableView.backgroundView = messageLabel;
        self.tableView.separatorStyle = UITableViewCellSeparatorStyleNone;
    } else {
        self.tableView.backgroundView = nil;
        self.tableView.separatorStyle = UITableViewCellSeparatorStyleSingleLine;
    }
}



- (void)refreshDevices {
    [self fetchLiveDevices];
}

- (void)fetchLiveDevices {
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken) {
        NSLog(@"No session token available");
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.refreshControl endRefreshing];
        });
        return;
    }
    
    NSString *encodedSessionToken = [sessionToken stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];
    NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/get_live_devices"];
    components.queryItems = @[
        [NSURLQueryItem queryItemWithName:@"session_token" value:encodedSessionToken]
    ];
    
    NSURL *url = components.URL;
    NSURLSessionDataTask *task = [self.session dataTaskWithURL:url completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.refreshControl endRefreshing];
        });
        
        if (error) return;
        if (!data) return;
        
        NSError *jsonError;
        id jsonObj = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError) return;
        NSArray *devices = nil;
        if ([jsonObj isKindOfClass:[NSDictionary class]]) {
            NSDictionary *json = (NSDictionary *)jsonObj;
            if ([json[@"devices"] isKindOfClass:[NSArray class]]) {
                devices = json[@"devices"];
            } else if ([json[@"device_names"] isKindOfClass:[NSArray class]] &&
                       [json[@"alerts_on"] isKindOfClass:[NSArray class]]) {
                NSArray *names = json[@"device_names"];
                NSArray *alerts = json[@"alerts_on"];
                NSMutableArray *converted = [NSMutableArray array];
                for (NSInteger i = 0; i < names.count; i++) {
                    NSString *name = names[i];
                    BOOL alertsOn = (i < alerts.count) ? [alerts[i] boolValue] : NO;
                    [converted addObject:@{@"name": name, @"alerts_on": @(alertsOn)}];
                }
                devices = converted;
            }
        }
        
        if (![devices isKindOfClass:[NSArray class]]) return;
        
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.deviceNames removeAllObjects];
            [self.alertsOnStates removeAllObjects];
            
            for (NSDictionary *device in devices) {
                NSString *name = device[@"name"];
                NSNumber *alertsOnValue = device[@"alerts_on"];
                
                if ([name isKindOfClass:[NSString class]]) {
                    NSString *decodedName = [name stringByRemovingPercentEncoding];
                    [self.deviceNames addObject:decodedName ?: name];
                    
                    BOOL alertsOn = [alertsOnValue boolValue];
                    [self.alertsOnStates addObject:@(alertsOn)];
                }
            }
            
            [self.tableView reloadData];
            [self updateTableViewBackground];
        });
    }];
    
    [task resume];
}


#pragma mark - UITableViewDataSource

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return self.deviceNames.count;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"DeviceCell" forIndexPath:indexPath];
    
    // Clear previous content
    for (UIView *view in cell.contentView.subviews) {
        [view removeFromSuperview];
    }

    NSString *deviceName = self.deviceNames[indexPath.row];

    // Thumbnail container (4:3 ratio)
    UIView *thumbContainer = [[UIView alloc] initWithFrame:CGRectMake(15, 10, 120, 90)];
    thumbContainer.backgroundColor = [UIColor secondarySystemBackgroundColor];
    thumbContainer.layer.cornerRadius = 8;
    thumbContainer.clipsToBounds = YES;

    UIImageView *thumbnailView = [[UIImageView alloc] initWithFrame:thumbContainer.bounds];
    thumbnailView.contentMode = UIViewContentModeScaleAspectFill;
    thumbnailView.clipsToBounds = YES;

    NSString *filename = [NSString stringWithFormat:@"thumbnail_%@.jpg", deviceName];
    NSString *documentsPath = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    NSString *filePath = [documentsPath stringByAppendingPathComponent:filename];
    UIImage *thumbnailImage = [UIImage imageWithContentsOfFile:filePath];

    if (thumbnailImage) {
        thumbnailView.image = thumbnailImage;
    } else {
        thumbnailView.image = nil;
        // Add play icon in center
        UIImageView *playIcon = [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"play.circle.fill"]];
        playIcon.tintColor = [UIColor labelColor];
        playIcon.contentMode = UIViewContentModeScaleAspectFit;
        playIcon.frame = CGRectMake((thumbContainer.frame.size.width - 40)/2, (thumbContainer.frame.size.height - 40)/2, 40, 40);
        [thumbContainer addSubview:playIcon];
    }

    [thumbContainer addSubview:thumbnailView];
    [cell.contentView addSubview:thumbContainer];

    // Device Name Label
    UILabel *nameLabel = [[UILabel alloc] initWithFrame:CGRectMake(150, 25, tableView.frame.size.width - 160, 25)];
    nameLabel.text = deviceName;
    nameLabel.font = [UIFont systemFontOfSize:17 weight:UIFontWeightSemibold];
    nameLabel.textColor = [UIColor labelColor];

    // Online Indicator
    UIView *greenDot = [[UIView alloc] initWithFrame:CGRectMake(150, 60, 10, 10)];
    greenDot.backgroundColor = [UIColor systemGreenColor];
    greenDot.layer.cornerRadius = 5;

    UILabel *onlineLabel = [[UILabel alloc] initWithFrame:CGRectMake(165, 57, 100, 15)];
    onlineLabel.text = NSLocalizedString(@"online", "online");
    onlineLabel.font = [UIFont systemFontOfSize:13];
    onlineLabel.textColor = [UIColor secondaryLabelColor];

    [cell.contentView addSubview:nameLabel];
    [cell.contentView addSubview:greenDot];
    [cell.contentView addSubview:onlineLabel];

    // Alerts toggle section - switch centered with label beneath
    CGFloat toggleSectionWidth = 80;
    CGFloat toggleSectionX = tableView.frame.size.width - toggleSectionWidth - 15;
    
    UISwitch *alertSwitch = [[UISwitch alloc] initWithFrame:CGRectMake(toggleSectionX + (toggleSectionWidth - 51)/2, 35, 0, 0)];
    alertSwitch.tag = indexPath.row;
    [alertSwitch addTarget:self action:@selector(alertSwitchToggled:) forControlEvents:UIControlEventValueChanged];

    BOOL alertsOn = NO;
    if (indexPath.row < self.alertsOnStates.count) {
        alertsOn = [self.alertsOnStates[indexPath.row] boolValue];
    }
    [alertSwitch setOn:alertsOn animated:NO];

    UILabel *alertsLabel = [[UILabel alloc] initWithFrame:CGRectMake(toggleSectionX, 75, toggleSectionWidth, 15)];
    alertsLabel.text = NSLocalizedString(@"alerts", nil);
    alertsLabel.font = [UIFont systemFontOfSize:13];
    alertsLabel.textColor = [UIColor secondaryLabelColor];
    alertsLabel.textAlignment = NSTextAlignmentCenter;

    [cell.contentView addSubview:alertSwitch];
    [cell.contentView addSubview:alertsLabel];

    cell.backgroundColor = [UIColor systemBackgroundColor];

    return cell;
}

- (void)alertSwitchToggled:(UISwitch *)sender {
    NSInteger index = sender.tag;
    if (index >= self.deviceNames.count) return;

    NSString *deviceName = self.deviceNames[index];
    BOOL newState = sender.isOn;
    
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken) {
        NSLog(@"No session token available.");
        [sender setOn:!newState animated:YES];
        return;
    }

    NSDictionary *body = @{
        @"session_token": sessionToken,
        @"device_name": [deviceName stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]], // url encoding
        @"alerts_on": @(newState)
    };

    NSError *jsonError;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:body options:0 error:&jsonError];
    if (jsonError) {
        NSLog(@"JSON encode error: %@", jsonError);
        [sender setOn:!newState animated:YES];
        return;
    }

    NSURL *url = [NSURL URLWithString:@"https://rors.ai/toggle_alerts"];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    request.HTTPMethod = @"POST";
    [request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
    request.HTTPBody = jsonData;

    NSURLSessionDataTask *task = [self.session dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"Toggle request failed: %@", error);
            dispatch_async(dispatch_get_main_queue(), ^{
                [sender setOn:!newState animated:YES];
            });
            return;
        }

        NSError *parseError;
        NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&parseError];
        if (parseError || ![json isKindOfClass:[NSDictionary class]]) {
            NSLog(@"JSON parse error: %@", parseError);
            dispatch_async(dispatch_get_main_queue(), ^{
                [sender setOn:!newState animated:YES];
            });
            return;
        }

        BOOL success = [json[@"success"] boolValue];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (success) {
                NSLog(@"Toggle updated successfully for %@: %@", deviceName, newState ? @"ON" : @"OFF");
                self.alertsOnStates[index] = @(newState);
            } else {
                NSLog(@"Toggle update failed: %@", json[@"message"]);
                [sender setOn:!newState animated:YES];
            }
        });
    }];
    [task resume];
}


#pragma mark - UITableViewDelegate

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    NSString *selectedDevice = self.deviceNames[indexPath.row];
    NSLog(@"Selected device: %@", selectedDevice);
    
    DeviceStreamViewController *streamVC = [[DeviceStreamViewController alloc] init];
    streamVC.deviceName = selectedDevice; // Pass the device name
    [self.navigationController pushViewController:streamVC animated:YES];
    
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath {
    return 110;
}

@end
