#import <UIKit/UIKit.h>

@interface GalleryViewController : UIViewController
@property (nonatomic, strong) UILabel *ipLabel;
@property (nonatomic, strong) NSLayoutConstraint *tableViewTopConstraint;
@property (nonatomic, strong) NSLayoutConstraint *mainStackViewTopToIPLabelConstraint;
@property (nonatomic, strong) NSLayoutConstraint *mainStackViewTopToSafeAreaConstraint;
@property (nonatomic, strong) UILabel *titleLabel;
@property (nonatomic, strong) UIButton *cameraButton;
@property (nonatomic, strong) UIButton *settingsButton;
@property (nonatomic, strong) UIStackView *headerStackView;
@property (nonatomic, strong) UIView *statusDot;
@property (nonatomic, strong) UILabel *statusLabel;
@property (nonatomic, strong) NSTimer *statusTimer;
- (void)getEvents;

@end
