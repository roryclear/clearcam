// GalleryViewController.m
#import "GalleryViewController.h"

@implementation GalleryViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];
    self.title = @"Events";
    // Ensure the navigation bar is visible
    self.navigationController.navigationBarHidden = NO;
}

@end
