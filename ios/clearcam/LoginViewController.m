#import "LoginViewController.h"
#import "MainViewController.h"
#import "StoreManager.h"

@interface LoginViewController ()

@end

@implementation LoginViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    [[StoreManager sharedInstance] clearSessionTokenFromKeychain];
    
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.navigationItem.title = @"Log in";
    
    UIView *container = [[UIView alloc] initWithFrame:CGRectZero];
    container.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:container];
    
    self.usernameTextField = [[UITextField alloc] initWithFrame:CGRectZero];
    self.usernameTextField.placeholder = @"Clearcam User ID";
    self.usernameTextField.borderStyle = UITextBorderStyleRoundedRect;
    self.usernameTextField.backgroundColor = [UIColor secondarySystemBackgroundColor];
    self.usernameTextField.textColor = [UIColor labelColor];
    self.usernameTextField.translatesAutoresizingMaskIntoConstraints = NO;
    [container addSubview:self.usernameTextField];

    self.loginButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.loginButton setTitle:@"Log In" forState:UIControlStateNormal];
    self.loginButton.backgroundColor = [UIColor systemBlueColor];
    self.loginButton.tintColor = [UIColor whiteColor];
    self.loginButton.layer.cornerRadius = 8;
    self.loginButton.translatesAutoresizingMaskIntoConstraints = NO;
    [self.loginButton addTarget:self action:@selector(loginButtonTapped) forControlEvents:UIControlEventTouchUpInside];
    [container addSubview:self.loginButton];
    
    [NSLayoutConstraint activateConstraints:@[
        [container.centerYAnchor constraintEqualToAnchor:self.view.centerYAnchor],
        [container.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:32],
        [container.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-32],

        [self.usernameTextField.topAnchor constraintEqualToAnchor:container.topAnchor],
        [self.usernameTextField.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
        [self.usernameTextField.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
        [self.usernameTextField.heightAnchor constraintEqualToConstant:44],
        
        [self.loginButton.topAnchor constraintEqualToAnchor:self.usernameTextField.bottomAnchor constant:16],
        [self.loginButton.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
        [self.loginButton.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
        [self.loginButton.heightAnchor constraintEqualToConstant:44],
        [self.loginButton.bottomAnchor constraintEqualToAnchor:container.bottomAnchor]
    ]];
}

- (void)loginButtonTapped {
    NSString *enteredToken = [self.usernameTextField.text stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    
    if (enteredToken.length == 0) {
        [self showAlertWithTitle:@"Error" message:@"Please enter your Clearcam User ID."];
        return;
    }
    
    NSString *urlString = [NSString stringWithFormat:@"https://rors.ai/validate_user?session_token=%@", enteredToken];
    NSURL *url = [NSURL URLWithString:urlString];
    
    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithURL:url
                                                             completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
            
            if (!error && httpResponse.statusCode >= 200 && httpResponse.statusCode < 300) {
                [[StoreManager sharedInstance] storeSessionTokenInKeychain:enteredToken];
                MainViewController *mainVC = [[MainViewController alloc] init];
                [self.navigationController pushViewController:mainVC animated:YES];
            } else {
                [self showAlertWithTitle:@"Invalid ID" message:@"The Clearcam User ID you entered is not valid."];
            }
        });
    }];
    [task resume];
}

- (void)showAlertWithTitle:(NSString *)title message:(NSString *)message {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:title
                                                                   message:message
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
    [self presentViewController:alert animated:YES completion:nil];
}

@end
