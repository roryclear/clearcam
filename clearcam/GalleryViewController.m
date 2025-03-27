// GalleryViewController.m
#import "GalleryViewController.h"

@implementation GalleryViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Events";
    // Ensure the navigation bar is visible
    self.navigationController.navigationBarHidden = NO;
    [self sendPostRequest];
}


- (void)sendPostRequest {
    // Set up the URL
    NSURL *url = [NSURL URLWithString:@"https://rors.ai/events"]; // Replace with your actual URL
    
    // Create the boundary
    NSString *boundary = [NSString stringWithFormat:@"Boundary-%@", [[NSUUID UUID] UUIDString]];
    
    // Create the body
    NSMutableData *bodyData = [NSMutableData data];
    
    // Add receipt field
    NSString *receipt = [[NSUserDefaults standardUserDefaults] stringForKey:@"subscriptionReceipt"];
    if (receipt) {
        [bodyData appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"Content-Disposition: form-data; name=\"receipt\"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[receipt dataUsingEncoding:NSUTF8StringEncoding]];
        [bodyData appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
    } else {
        NSLog(@"No subscription receipt found in NSUserDefaults");
    }
    
    // Close the boundary
    [bodyData appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

    // Create the request
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    [request setHTTPMethod:@"POST"];
    [request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
    [request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)bodyData.length] forHTTPHeaderField:@"Content-Length"];
    [request setHTTPBody:bodyData];
    
    // Create the data task
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
                
                if (statusCode >= 200 && statusCode < 300) {
                    NSString *responseString = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
                    NSLog(@"POST response from website.com/send: %@", responseString ?: @"No response body");
                } else {
                    NSLog(@"POST request returned status code: %ld", (long)statusCode);
                }
            } else {
                NSLog(@"Unexpected response type: %@", response);
            }
        });
    }];
    
    // Start the task
    [task resume];
}

@end
