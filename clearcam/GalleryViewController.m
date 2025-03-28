// GalleryViewController.m
#import "GalleryViewController.h"
#import "StoreManager.h"
#import <AVKit/AVKit.h>
#import <AVFoundation/AVFoundation.h>

@interface GalleryViewController () <UITableViewDelegate, UITableViewDataSource>
@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSMutableArray<NSString *> *videoFiles;
@property (nonatomic, strong) NSString *downloadDirectory;
@property (nonatomic, strong) NSURLSession *downloadSession;
@end

@implementation GalleryViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Events";
    self.navigationController.navigationBarHidden = NO;
    
    // Initialize with empty array
    self.videoFiles = [NSMutableArray array];
    
    // Configure a session with higher timeout
    NSURLSessionConfiguration *config = [NSURLSessionConfiguration defaultSessionConfiguration];
    config.timeoutIntervalForRequest = 60.0;
    config.timeoutIntervalForResource = 600.0;
    self.downloadSession = [NSURLSession sessionWithConfiguration:config];
    
    [self setupDownloadDirectory];
    [self setupTableView];
    [self sendPostRequest];
}

- (void)setupDownloadDirectory {
    NSString *documentsDir = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    self.downloadDirectory = [documentsDir stringByAppendingPathComponent:@"downloaded-events"];
    
    // Clear existing directory
    NSError *error;
    if ([[NSFileManager defaultManager] fileExistsAtPath:self.downloadDirectory]) {
        [[NSFileManager defaultManager] removeItemAtPath:self.downloadDirectory error:&error];
        if (error) {
            NSLog(@"Failed to clear directory: %@", error.localizedDescription);
        }
    }
    
    // Recreate directory
    [[NSFileManager defaultManager] createDirectoryAtPath:self.downloadDirectory
                          withIntermediateDirectories:YES
                                           attributes:nil
                                                error:&error];
    if (error) {
        NSLog(@"Failed to create directory: %@", error.localizedDescription);
    }
}

- (void)setupTableView {
    self.tableView = [[UITableView alloc] initWithFrame:self.view.bounds style:UITableViewStylePlain];
    self.tableView.delegate = self;
    self.tableView.dataSource = self;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor];
    [self.tableView registerClass:[UITableViewCell class] forCellReuseIdentifier:@"VideoCell"];
    [self.view addSubview:self.tableView];
}

- (void)sendPostRequest {
    NSURL *url = [NSURL URLWithString:@"https://rors.ai/events"];
    NSString *boundary = [NSString stringWithFormat:@"Boundary-%@", [[NSUUID UUID] UUIDString]];
    
    // Build multipart form data
    NSMutableData *bodyData = [NSMutableData data];
    NSString *receipt = [[NSUserDefaults standardUserDefaults] stringForKey:@"subscriptionReceipt"];
    
    if (receipt) {
        [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"Content-Disposition: form-data; name=\"receipt\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[receipt dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    }
    
    [bodyData appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    [request setHTTPMethod:@"POST"];
    [request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
    [request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)bodyData.length] forHTTPHeaderField:@"Content-Length"];
    [request setHTTPBody:bodyData];
    
    [[self.downloadSession dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"Request failed: %@", error);
            return;
        }
        
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
        if (httpResponse.statusCode == 200) {
            NSError *jsonError;
            NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
            
            if (!jsonError && json[@"files"]) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self downloadFiles:json[@"files"]];
                });
            }
        }
    }] resume];
}

- (void)downloadFiles:(NSArray<NSString *> *)fileNames {
    NSString *receipt = [[NSUserDefaults standardUserDefaults] stringForKey:@"subscriptionReceipt"];
    if (!receipt) return;
    
    dispatch_group_t downloadGroup = dispatch_group_create();
    
    for (NSString *fileName in fileNames) {
        dispatch_group_enter(downloadGroup);
        
        NSURL *url = [NSURL URLWithString:@"https://rors.ai/video"];
        NSString *boundary = [NSString stringWithFormat:@"Boundary-%@", [[NSUUID UUID] UUIDString]];

        NSMutableData *bodyData = [NSMutableData data];

        // Retrieve session token from Keychain
        
        NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
        if (!sessionToken) {
            NSLog(@"No session token found in Keychain.");
            dispatch_group_leave(downloadGroup); // Ensure group is left on failure
            return;
        }

        // Add session_token field
        [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"Content-Disposition: form-data; name=\"session_token\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[sessionToken dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];

        // Add name field
        [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"Content-Disposition: form-data; name=\"name\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[fileName dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];

        [bodyData appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

        NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
        [request setHTTPMethod:@"POST"];
        [request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
        [request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)bodyData.length] forHTTPHeaderField:@"Content-Length"];
        [request setHTTPBody:bodyData];

        [[self.downloadSession downloadTaskWithRequest:request completionHandler:^(NSURL *location, NSURLResponse *response, NSError *error) {
            if (!error && [(NSHTTPURLResponse *)response statusCode] == 200) {
                NSString *destPath = [self.downloadDirectory stringByAppendingPathComponent:fileName];
                [[NSFileManager defaultManager] moveItemAtURL:location toURL:[NSURL fileURLWithPath:destPath] error:nil];
                NSLog(@"File downloaded to: %@", destPath);
            } else {
                NSLog(@"Download failed with error: %@, status code: %ld", error.localizedDescription, (long)[(NSHTTPURLResponse *)response statusCode]);
            }
            dispatch_group_leave(downloadGroup);
        }] resume];
    }
    
    dispatch_group_notify(downloadGroup, dispatch_get_main_queue(), ^{
        [self loadExistingVideos];
    });
}

- (void)loadExistingVideos {
    [self.videoFiles removeAllObjects];
    
    NSError *error;
    NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:self.downloadDirectory error:&error];
    
    for (NSString *file in contents) {
        if ([file.pathExtension isEqualToString:@"mp4"]) {
            [self.videoFiles addObject:[self.downloadDirectory stringByAppendingPathComponent:file]];
        }
    }
    
    [self.tableView reloadData];
}

// UITableView methods remain the same as in your original code
- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return self.videoFiles.count;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"VideoCell" forIndexPath:indexPath];
    cell.textLabel.text = [self.videoFiles[indexPath.row] lastPathComponent];
    return cell;
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    AVPlayerViewController *playerVC = [[AVPlayerViewController alloc] init];
    playerVC.player = [AVPlayer playerWithURL:[NSURL fileURLWithPath:self.videoFiles[indexPath.row]]];
    [self presentViewController:playerVC animated:YES completion:^{ [playerVC.player play]; }];
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

@end
