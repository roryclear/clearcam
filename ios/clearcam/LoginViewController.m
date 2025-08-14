#import "LoginViewController.h"
#import "GalleryViewController.h"
#import "StoreManager.h"

@interface LoginViewController ()
@property (nonatomic, strong) UIImageView *appIconView;
@end

@implementation LoginViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    [[StoreManager sharedInstance] clearSessionTokenFromKeychain];
    
    self.navigationItem.title = @"";
    [self.navigationController.navigationBar setBackgroundImage:[UIImage new] forBarMetrics:UIBarMetricsDefault];
    self.navigationController.navigationBar.shadowImage = [UIImage new];
    self.navigationController.navigationBar.translucent = YES;
    
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    
    UIView *container = [[UIView alloc] initWithFrame:CGRectZero];
    container.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:container];
    
    NSArray *iconFiles = [[[NSBundle mainBundle] infoDictionary] valueForKeyPath:@"CFBundleIcons.CFBundlePrimaryIcon.CFBundleIconFiles"];
    NSString *iconName = [iconFiles lastObject];
    UIImage *appIcon = [UIImage imageNamed:iconName];
    self.appIconView = [[UIImageView alloc] initWithImage:appIcon];
    self.appIconView.contentMode = UIViewContentModeScaleAspectFit;
    self.appIconView.translatesAutoresizingMaskIntoConstraints = NO;
    self.appIconView.layer.cornerRadius = 12.0;
    self.appIconView.layer.masksToBounds = YES;
    self.appIconView.layer.borderWidth = 0.5;
    self.appIconView.layer.borderColor = [[UIColor systemGray4Color] CGColor];
    [container addSubview:self.appIconView];
    
    UIButton *signUpButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [signUpButton setTitle:@"Sign Up" forState:UIControlStateNormal];
    signUpButton.backgroundColor = [UIColor systemBlueColor];
    [signUpButton setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
    signUpButton.layer.cornerRadius = 8;
    signUpButton.translatesAutoresizingMaskIntoConstraints = NO;
    [container addSubview:signUpButton];
    
    UILabel *alreadyHaveLabel = [[UILabel alloc] initWithFrame:CGRectZero];
    alreadyHaveLabel.text = @"Already have an account?";
    alreadyHaveLabel.textAlignment = NSTextAlignmentCenter;
    alreadyHaveLabel.font = [UIFont boldSystemFontOfSize:15];
    alreadyHaveLabel.textColor = [UIColor labelColor];
    alreadyHaveLabel.translatesAutoresizingMaskIntoConstraints = NO;
    [container addSubview:alreadyHaveLabel];
    
    self.usernameTextField = [[UITextField alloc] initWithFrame:CGRectZero];
    self.usernameTextField.placeholder = @"Clearcam User ID";
    self.usernameTextField.borderStyle = UITextBorderStyleRoundedRect;
    self.usernameTextField.backgroundColor = [UIColor secondarySystemBackgroundColor];
    self.usernameTextField.textColor = [UIColor labelColor];
    self.usernameTextField.translatesAutoresizingMaskIntoConstraints = NO;
    [container addSubview:self.usernameTextField];
    
    self.loginButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.loginButton setTitle:@"Log In" forState:UIControlStateNormal];
    self.loginButton.backgroundColor = [UIColor systemGray5Color];
    [self.loginButton setTitleColor:[UIColor labelColor] forState:UIControlStateNormal];
    self.loginButton.layer.cornerRadius = 8;
    self.loginButton.translatesAutoresizingMaskIntoConstraints = NO;
    [self.loginButton addTarget:self action:@selector(loginButtonTapped) forControlEvents:UIControlEventTouchUpInside];
    [container addSubview:self.loginButton];

    UIButton *continueWithoutButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [continueWithoutButton setTitle:@"Continue without account" forState:UIControlStateNormal];
    continueWithoutButton.titleLabel.font = [UIFont systemFontOfSize:14 weight:UIFontWeightRegular];
    [continueWithoutButton setTitleColor:[UIColor systemBlueColor] forState:UIControlStateNormal];
    continueWithoutButton.backgroundColor = [UIColor clearColor];
    continueWithoutButton.translatesAutoresizingMaskIntoConstraints = NO;
    [container addSubview:continueWithoutButton];
    
    [NSLayoutConstraint activateConstraints:@[
        [container.centerYAnchor constraintEqualToAnchor:self.view.centerYAnchor constant:-40],
        [container.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:32],
        [container.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-32],

        [self.appIconView.topAnchor constraintEqualToAnchor:container.topAnchor],
        [self.appIconView.centerXAnchor constraintEqualToAnchor:container.centerXAnchor],
        [self.appIconView.widthAnchor constraintEqualToConstant:100],
        [self.appIconView.heightAnchor constraintEqualToConstant:100],

        [signUpButton.topAnchor constraintEqualToAnchor:self.appIconView.bottomAnchor constant:20],
        [signUpButton.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
        [signUpButton.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
        [signUpButton.heightAnchor constraintEqualToConstant:44],
        
        // Already have account label
        [alreadyHaveLabel.topAnchor constraintEqualToAnchor:signUpButton.bottomAnchor constant:20],
        [alreadyHaveLabel.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
        [alreadyHaveLabel.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
        
        // Username text field
        [self.usernameTextField.topAnchor constraintEqualToAnchor:alreadyHaveLabel.bottomAnchor constant:8],
        [self.usernameTextField.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
        [self.usernameTextField.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
        [self.usernameTextField.heightAnchor constraintEqualToConstant:44],
        
        // Login button
        [self.loginButton.topAnchor constraintEqualToAnchor:self.usernameTextField.bottomAnchor constant:16],
        [self.loginButton.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
        [self.loginButton.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
        [self.loginButton.heightAnchor constraintEqualToConstant:44],
        
        [continueWithoutButton.topAnchor constraintEqualToAnchor:self.loginButton.bottomAnchor constant:12],
        [continueWithoutButton.centerXAnchor constraintEqualToAnchor:container.centerXAnchor],
        [continueWithoutButton.bottomAnchor constraintEqualToAnchor:container.bottomAnchor]
    ]];
    [continueWithoutButton addTarget:self action:@selector(continueWithoutAccountTapped) forControlEvents:UIControlEventTouchUpInside];
}

- (void)continueWithoutAccountTapped {
    GalleryViewController *mainVC = [[GalleryViewController alloc] init];
    [self.navigationController pushViewController:mainVC animated:YES];
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
                GalleryViewController *mainVC = [[GalleryViewController alloc] init];
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
