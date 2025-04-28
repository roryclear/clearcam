#import "LiveViewController.h"
#import "StoreManager.h"
#import "DeviceStreamViewController.h" // Import DeviceStreamViewController

@interface LiveViewController () <UITableViewDelegate, UITableViewDataSource>
@property (nonatomic, strong) NSMutableArray<NSString *> *deviceNames;
@property (nonatomic, strong) NSURLSession *session;
@end

@implementation LiveViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"Live";
    self.deviceNames = [NSMutableArray array];


    self.session = [NSURLSession sessionWithConfiguration:[NSURLSessionConfiguration defaultSessionConfiguration]];
    [self setupTableView];
    [self fetchLiveDevices];
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
}

- (void)fetchLiveDevices {
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    if (!sessionToken) {
        NSLog(@"No session token available");
        return;
    }
    
    NSString *encodedSessionToken = [sessionToken stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];
    NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/get_live_devices"];
    components.queryItems = @[
        [NSURLQueryItem queryItemWithName:@"session_token" value:encodedSessionToken]
    ];
    
    NSURL *url = components.URL;
    NSLog(@"Final URL: %@", url);
    
    NSURLSessionDataTask *task = [self.session dataTaskWithURL:url completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"❌ Error: %@", error.localizedDescription);
            return;
        }
        
        NSError *jsonError;
        NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError) {
            NSLog(@"❌ JSON Error: %@", jsonError.localizedDescription);
            return;
        }
        
        NSArray<NSString *> *deviceNames = json[@"device_names"];
        if (deviceNames) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.deviceNames removeAllObjects];
                for (NSString *deviceName in deviceNames) {
                    NSString *decodedName = [deviceName stringByRemovingPercentEncoding];
                    if (decodedName) {
                        [self.deviceNames addObject:decodedName];
                    }
                }
                [self.tableView reloadData];
            });
        }
    }];
    [task resume];
}

#pragma mark - UITableViewDataSource

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return self.deviceNames.count;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"DeviceCell" forIndexPath:indexPath];
    cell.textLabel.text = self.deviceNames[indexPath.row];
    cell.textLabel.font = [UIFont systemFontOfSize:16 weight:UIFontWeightRegular];
    cell.backgroundColor = [UIColor systemBackgroundColor];
    return cell;
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
    return 44;
}

@end
