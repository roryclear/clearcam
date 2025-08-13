#import "LoginViewController.h"
#import "MainViewController.h"

@interface LoginViewController ()

@end

@implementation LoginViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.navigationItem.title = @"Login";
    
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
        // Center container vertically
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
    MainViewController *mainVC = [[MainViewController alloc] init];
    [self.navigationController pushViewController:mainVC animated:YES];
}

@end
