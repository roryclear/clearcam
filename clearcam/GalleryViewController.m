// GalleryViewController.m
#import "GalleryViewController.h"
#import <AVKit/AVKit.h>
#import <AVFoundation/AVFoundation.h>

@interface GalleryViewController () <UITableViewDelegate, UITableViewDataSource>
@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSMutableArray<NSString *> *videoFiles; // List of file paths
@property (nonatomic, strong) NSString *downloadDirectory; // Path to /downloaded-events
@end

@implementation GalleryViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Events";
    self.navigationController.navigationBarHidden = NO;
    
    // Initialize video files array
    self.videoFiles = [NSMutableArray array];
    
    // Set up the download directory
    [self setupDownloadDirectory];
    
    // Set up the table view
    [self setupTableView];
    
    // Send the POST request
    [self sendPostRequest];
}

- (void)setupDownloadDirectory {
    NSString *documentsDir = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    self.downloadDirectory = [documentsDir stringByAppendingPathComponent:@"downloaded-events"];
    
    NSError *error;
    if (![[NSFileManager defaultManager] fileExistsAtPath:self.downloadDirectory]) {
        [[NSFileManager defaultManager] createDirectoryAtPath:self.downloadDirectory
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:&error];
        if (error) {
            NSLog(@"Failed to create download directory: %@", error.localizedDescription);
        }
    }
    
    [self loadExistingVideos];
}

- (void)setupTableView {
    self.tableView = [[UITableView alloc] initWithFrame:self.view.bounds style:UITableViewStylePlain];
    self.tableView.delegate = self;
    self.tableView.dataSource = self;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor];
    [self.view addSubview:self.tableView];
    
    [self.tableView registerClass:[UITableViewCell class] forCellReuseIdentifier:@"VideoCell"];
}

- (void)loadExistingVideos {
    [self.videoFiles removeAllObjects];
    
    NSError *error;
    NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:self.downloadDirectory error:&error];
    if (error) {
        NSLog(@"Failed to load existing videos: %@", error.localizedDescription);
        return;
    }
    
    for (NSString *fileName in contents) {
        if ([[fileName pathExtension] isEqualToString:@"mp4"]) {
            NSString *filePath = [self.downloadDirectory stringByAppendingPathComponent:fileName];
            [self.videoFiles addObject:filePath];
        }
    }
    [self.tableView reloadData];
}

- (void)sendPostRequest {
    NSURL *url = [NSURL URLWithString:@"https://rors.ai/events"];
    NSString *boundary = [NSString stringWithFormat:@"Boundary-%@", [[NSUUID UUID] UUIDString]];
    NSMutableData *bodyData = [NSMutableData data];
    
    NSString *receipt = [[NSUserDefaults standardUserDefaults] stringForKey:@"subscriptionReceipt"];
    if (receipt) {
        [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"Content-Disposition: form-data; name=\"receipt\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[receipt dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    } else {
        NSLog(@"No subscription receipt found in NSUserDefaults");
    }
    
    [bodyData appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
    
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    [request setHTTPMethod:@"POST"];
    [request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
    [request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)bodyData.length] forHTTPHeaderField:@"Content-Length"];
    [request setHTTPBody:bodyData];
    
    NSURLSession *session = [NSURLSession sharedSession];
    NSURLSessionDataTask *task = [session dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (error) {
                NSLog(@"POST request failed: %@", error.localizedDescription);
                return;
            }
            
            if ([response isKindOfClass:[NSHTTPURLResponse class]]) {
                NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
                NSInteger statusCode = httpResponse.statusCode;
                
                // Log the raw response data for debugging
                NSString *rawResponse = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
                NSLog(@"Raw response from /events (status %ld): '%@'", (long)statusCode, rawResponse ?: @"[Non-UTF8 data]");
                
                if (statusCode == 200) {
                    NSError *jsonError;
                    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
                    if (jsonError) {
                        NSLog(@"Failed to parse JSON: %@", jsonError.localizedDescription);
                        return;
                    }
                    
                    NSArray *files = json[@"files"];
                    if (files) {
                        [self downloadFiles:files];
                    } else {
                        NSLog(@"No 'files' key in response: %@", json);
                    }
                } else {
                    NSLog(@"POST request returned status code: %ld", (long)statusCode);
                }
            }
        });
    }];
    [task resume];
}

- (void)downloadFiles:(NSArray<NSString *> *)fileNames {
    NSString *receipt = [[NSUserDefaults standardUserDefaults] stringForKey:@"subscriptionReceipt"];
    if (!receipt) {
        NSLog(@"Cannot download files: No receipt available");
        return;
    }
    
    for (NSString *fileName in fileNames) {
        NSURL *downloadURL = [NSURL URLWithString:@"https://rors.ai/video"];
        NSString *boundary = [NSString stringWithFormat:@"Boundary-%@", [[NSUUID UUID] UUIDString]];
        NSMutableData *bodyData = [NSMutableData data];
        
        // Add receipt field
        [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"Content-Disposition: form-data; name=\"receipt\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[receipt dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        
        // Add name field
        [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"Content-Disposition: form-data; name=\"name\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[fileName dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        
        // Close the boundary
        [bodyData appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        
        NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:downloadURL];
        [request setHTTPMethod:@"POST"];
        [request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
        [request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)bodyData.length] forHTTPHeaderField:@"Content-Length"];
        [request setHTTPBody:bodyData];
        
        NSURLSessionDownloadTask *downloadTask = [[NSURLSession sharedSession] downloadTaskWithRequest:request completionHandler:^(NSURL *location, NSURLResponse *response, NSError *error) {
            dispatch_async(dispatch_get_main_queue(), ^{
                if (error) {
                    NSLog(@"Failed to download %@: %@", fileName, error.localizedDescription);
                    return;
                }
                
                // Check if the download was successful (status code 200)
                if ([response isKindOfClass:[NSHTTPURLResponse class]] && [(NSHTTPURLResponse *)response statusCode] == 200) {
                    // Verify the temporary file exists
                    if (!location || ![[NSFileManager defaultManager] fileExistsAtPath:location.path]) {
                        NSLog(@"Temporary file doesn't exist at %@", location.path);
                        return;
                    }
                    
                    // Ensure the download directory exists
                    NSError *dirError;
                    if (![[NSFileManager defaultManager] fileExistsAtPath:self.downloadDirectory]) {
                        [[NSFileManager defaultManager] createDirectoryAtPath:self.downloadDirectory
                                                  withIntermediateDirectories:YES
                                                                   attributes:nil
                                                                        error:&dirError];
                        if (dirError) {
                            NSLog(@"Failed to create download directory: %@", dirError.localizedDescription);
                            return;
                        }
                    }
                    
                    NSString *destinationPath = [self.downloadDirectory stringByAppendingPathComponent:fileName];
                    NSError *moveError;
                    
                    // Remove existing file if it exists
                    if ([[NSFileManager defaultManager] fileExistsAtPath:destinationPath]) {
                        [[NSFileManager defaultManager] removeItemAtPath:destinationPath error:&moveError];
                        if (moveError) {
                            NSLog(@"Failed to remove existing file: %@", moveError.localizedDescription);
                        }
                    }
                    
                    // Perform the move operation
                    [[NSFileManager defaultManager] moveItemAtURL:location
                                                          toURL:[NSURL fileURLWithPath:destinationPath]
                                                          error:&moveError];
                    
                    if (moveError) {
                        NSLog(@"Failed to move file %@ from %@ to %@: %@",
                              fileName,
                              location.path,
                              destinationPath,
                              moveError.localizedDescription);
                    } else {
                        [self.videoFiles addObject:destinationPath];
                        [self.tableView reloadData];
                    }
                } else {
                    NSLog(@"Download failed for %@: status code %ld", fileName, (long)[(NSHTTPURLResponse *)response statusCode]);
                }
            });
        }];
        [downloadTask resume];
    }
}

#pragma mark - UITableViewDataSource

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return self.videoFiles.count;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"VideoCell" forIndexPath:indexPath];
    NSString *filePath = self.videoFiles[indexPath.row];
    cell.textLabel.text = [filePath lastPathComponent];
    cell.backgroundColor = [UIColor systemBackgroundColor];
    cell.textLabel.textColor = [UIColor labelColor];
    return cell;
}

#pragma mark - UITableViewDelegate

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    NSString *filePath = self.videoFiles[indexPath.row];
    NSURL *fileURL = [NSURL fileURLWithPath:filePath];
    
    AVPlayer *player = [AVPlayer playerWithURL:fileURL];
    AVPlayerViewController *playerVC = [[AVPlayerViewController alloc] init];
    playerVC.player = player;
    
    [self presentViewController:playerVC animated:YES completion:^{
        [player play];
    }];
    
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
}

@end
