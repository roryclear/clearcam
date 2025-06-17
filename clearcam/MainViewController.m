// MainViewController.m
#import "MainViewController.h"
#import "ViewController.h"
#import "GalleryViewController.h"
#import "SettingsViewController.h"
#import "FileServer.h"

@interface MainViewController ()
@property (nonatomic, strong) FileServer *fileServer;
@property (nonatomic, strong) UILabel *ipLabel;
@property (nonatomic, strong) UIStackView *mainStackView;
@property (nonatomic, strong) NSLayoutConstraint *mainStackViewTopToIPLabelConstraint;
@property (nonatomic, strong) NSLayoutConstraint *mainStackViewTopToSafeAreaConstraint;
@end

@implementation MainViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    self.fileServer = [FileServer sharedInstance];
    [self.fileServer start];

    self.view.backgroundColor = (self.traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark)
        ? [UIColor blackColor]
        : [UIColor whiteColor];

    self.title = @"Clearcam";
    self.navigationItem.backBarButtonItem = [[UIBarButtonItem alloc] initWithTitle:NSLocalizedString(@"home", nil)
                                                                            style:UIBarButtonItemStylePlain
                                                                           target:nil
                                                                           action:nil];

    // Setup IP label
    self.ipLabel = [[UILabel alloc] init];
    self.ipLabel.translatesAutoresizingMaskIntoConstraints = NO;
    self.ipLabel.font = [UIFont systemFontOfSize:14 weight:UIFontWeightMedium];
    self.ipLabel.textColor = [UIColor secondaryLabelColor];
    self.ipLabel.textAlignment = NSTextAlignmentCenter;
    self.ipLabel.numberOfLines = 1;
    [self.view addSubview:self.ipLabel];

    [NSLayoutConstraint activateConstraints:@[
        [self.ipLabel.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor constant:10],
        [self.ipLabel.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor constant:20],
        [self.ipLabel.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor constant:-20]
    ]];

    // Stack View setup
    self.mainStackView = [[UIStackView alloc] init];
    self.mainStackView.axis = UILayoutConstraintAxisVertical;
    self.mainStackView.distribution = UIStackViewDistributionFillEqually;
    self.mainStackView.alignment = UIStackViewAlignmentFill;
    self.mainStackView.spacing = 20;
    self.mainStackView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.mainStackView];

    // Create the two possible top constraints for the stack view
    // The constant for mainStackViewTopToIPLabelConstraint should ensure there's enough space below the label
    self.mainStackViewTopToIPLabelConstraint = [self.mainStackView.topAnchor constraintEqualToAnchor:self.ipLabel.bottomAnchor constant:20]; // Keep original constant
    self.mainStackViewTopToSafeAreaConstraint = [self.mainStackView.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor constant:20]; // Keep original constant

    [NSLayoutConstraint activateConstraints:@[
        [self.mainStackView.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor constant:20],
        [self.mainStackView.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor constant:-20],
        [self.mainStackView.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor constant:-20]
    ]];

    [self addItemToStackView:self.mainStackView
                     title:NSLocalizedString(@"camera_desc", nil)
                     image:[UIImage systemImageNamed:@"camera.fill"]
                    action:@selector(cameraTapped)];

    [self addItemToStackView:self.mainStackView
                     title:NSLocalizedString(@"gallery_desc", nil)
                     image:[UIImage systemImageNamed:@"photo.on.rectangle"]
                    action:@selector(galleryTapped)];

    [self addItemToStackView:self.mainStackView
                     title:NSLocalizedString(@"settings_desc", nil)
                     image:[UIImage systemImageNamed:@"gear"]
                    action:@selector(settingsTapped)];

    // Observe IP address changes
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(updateIPAddressLabel)
                                                 name:@"DeviceIPAddressDidChangeNotification"
                                               object:nil];
}

- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    [self updateIPAddressLabel]; // Ensure layout is correct when view appears/reappears
}

- (void)updateIPAddressLabel {
    BOOL streamViaWifiEnabled = [[NSUserDefaults standardUserDefaults] boolForKey:@"stream_via_wifi_enabled"];
    NSString *ipAddress = [[NSUserDefaults standardUserDefaults] stringForKey:@"DeviceIPAddress"];

    if (streamViaWifiEnabled) {
        // If streaming is enabled, IP label should be visible
        self.ipLabel.hidden = NO;
        self.ipLabel.text = (ipAddress.length > 0) ? [NSString stringWithFormat:@"Streaming over local Wi-Fi at: %@", ipAddress] : @"Waiting for Wi-Fi...";

        // Activate constraint to IP label, deactivate constraint to safe area
        self.mainStackViewTopToSafeAreaConstraint.active = NO;
        self.mainStackViewTopToIPLabelConstraint.active = YES;
    } else {
        // If streaming is disabled, IP label should be hidden
        self.ipLabel.hidden = YES;
        self.ipLabel.text = nil; // Clear text when hidden

        // Activate constraint to safe area, deactivate constraint to IP label
        self.mainStackViewTopToIPLabelConstraint.active = NO;
        self.mainStackViewTopToSafeAreaConstraint.active = YES;
    }

    // Force layout update after constraint changes
    [self.view layoutIfNeeded];
}

- (void)addItemToStackView:(UIStackView *)stackView title:(NSString *)title image:(UIImage *)image action:(SEL)action {
    // Create container view
    UIView *container = [[UIView alloc] init];
    container.backgroundColor = [UIColor systemGray6Color];
    container.layer.cornerRadius = 10;
    container.translatesAutoresizingMaskIntoConstraints = NO;

    // Create image view
    UIImageView *imageView = [[UIImageView alloc] initWithImage:image];
    imageView.contentMode = UIViewContentModeScaleAspectFit;
    imageView.tintColor = [UIColor systemBlueColor];
    imageView.translatesAutoresizingMaskIntoConstraints = NO;

    // Create label
    UILabel *label = [[UILabel alloc] init];
    label.text = title;
    label.numberOfLines = 0;
    label.font = [UIFont systemFontOfSize:18];
    label.translatesAutoresizingMaskIntoConstraints = NO;

    // Create horizontal stack view for image and label
    UIStackView *itemStackView = [[UIStackView alloc] init];
    itemStackView.axis = UILayoutConstraintAxisHorizontal;
    itemStackView.alignment = UIStackViewAlignmentCenter;
    itemStackView.spacing = 15;
    itemStackView.translatesAutoresizingMaskIntoConstraints = NO;
    [itemStackView addArrangedSubview:imageView];
    [itemStackView addArrangedSubview:label];

    // Add stack view to container
    [container addSubview:itemStackView];
    [NSLayoutConstraint activateConstraints:@[
        [itemStackView.leadingAnchor constraintEqualToAnchor:container.leadingAnchor constant:15],
        [itemStackView.trailingAnchor constraintEqualToAnchor:container.trailingAnchor constant:-15],
        [itemStackView.topAnchor constraintEqualToAnchor:container.topAnchor constant:10],
        [itemStackView.bottomAnchor constraintEqualToAnchor:container.bottomAnchor constant:-10],
        [imageView.widthAnchor constraintEqualToConstant:50],
        [imageView.heightAnchor constraintEqualToConstant:50]
    ]];

    // Add tap gesture
    container.userInteractionEnabled = YES;
    UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:action];
    [container addGestureRecognizer:tap];

    // Add to main stack view
    [stackView addArrangedSubview:container];
}

#pragma mark - Actions

- (void)cameraTapped {
    ViewController *cameraVC = [[ViewController alloc] init];
    [self.navigationController pushViewController:cameraVC animated:YES];
}

- (void)galleryTapped {
    GalleryViewController *galleryVC = [[GalleryViewController alloc] init];
    [self.navigationController pushViewController:galleryVC animated:YES];
}

- (void)settingsTapped {
    SettingsViewController *settingsVC = [[SettingsViewController alloc] init];
    [self.navigationController pushViewController:settingsVC animated:YES];
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self name:@"DeviceIPAddressDidChangeNotification" object:nil];
}

@end
