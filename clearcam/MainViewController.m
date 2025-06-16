// MainViewController.m
#import "MainViewController.h"
#import "ViewController.h"
#import "GalleryViewController.h"
#import "SettingsViewController.h"
#import "FileServer.h"

@interface MainViewController ()
@property (nonatomic, strong) FileServer *fileServer;
@property (nonatomic, strong) UILabel *ipLabel;
@end

@implementation MainViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.fileServer = [FileServer sharedInstance];
    [self.fileServer start];
    if (self.traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark) {
        self.view.backgroundColor = [UIColor blackColor];
    } else {
        self.view.backgroundColor = [UIColor whiteColor];
    }
    
    self.title = @"Clearcam";
    self.navigationItem.backBarButtonItem = [[UIBarButtonItem alloc] initWithTitle:NSLocalizedString(@"home", nil)
                                                                                 style:UIBarButtonItemStylePlain
                                                                                target:nil
                                                                                action:nil];
    
    NSString *ipAddress = [[NSUserDefaults standardUserDefaults] stringForKey:@"DeviceIPAddress"];
    NSLayoutYAxisAnchor *stackTopAnchor;

    if (ipAddress && ipAddress.length > 0) {
        self.ipLabel = [[UILabel alloc] init];
        self.ipLabel.translatesAutoresizingMaskIntoConstraints = NO;
        self.ipLabel.font = [UIFont systemFontOfSize:14 weight:UIFontWeightMedium];
        self.ipLabel.textColor = [UIColor secondaryLabelColor];
        self.ipLabel.textAlignment = NSTextAlignmentCenter;
        self.ipLabel.numberOfLines = 1;
        self.ipLabel.text = [NSString stringWithFormat:@"Streaming over local Wi-Fi at: http://%@", ipAddress];
        [self.view addSubview:self.ipLabel];

        [NSLayoutConstraint activateConstraints:@[
            [self.ipLabel.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor constant:10],
            [self.ipLabel.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor constant:20],
            [self.ipLabel.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor constant:-20]
        ]];

        stackTopAnchor = self.ipLabel.bottomAnchor;
    } else {
        stackTopAnchor = self.view.safeAreaLayoutGuide.topAnchor;
    }


    UIStackView *stackView = [[UIStackView alloc] init];
    stackView.axis = UILayoutConstraintAxisVertical;
    stackView.distribution = UIStackViewDistributionFillEqually;
    stackView.alignment = UIStackViewAlignmentFill;
    stackView.spacing = 20;
    stackView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:stackView];

    // Add constraints for stack view
    [NSLayoutConstraint activateConstraints:@[
        [stackView.topAnchor constraintEqualToAnchor:stackTopAnchor constant:20],
        [stackView.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor constant:20],
        [stackView.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor constant:-20],
        [stackView.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor constant:-20]
    ]];

    // Create the three items
    [self addItemToStackView:stackView
                       title:NSLocalizedString(@"camera_desc", nil)
                       image:[UIImage systemImageNamed:@"camera.fill"]
                      action:@selector(cameraTapped)];
    [self addItemToStackView:stackView
                       title:NSLocalizedString(@"gallery_desc", nil)
                       image:[UIImage systemImageNamed:@"photo.on.rectangle"]
                      action:@selector(galleryTapped)];
    [self addItemToStackView:stackView
                       title:NSLocalizedString(@"settings_desc", nil)
                       image:[UIImage systemImageNamed:@"gear"]
                      action:@selector(settingsTapped)];
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

@end

