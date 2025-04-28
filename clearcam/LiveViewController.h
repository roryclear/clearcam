#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface LiveViewController : UIViewController <UITableViewDelegate, UITableViewDataSource>
@property (nonatomic, strong) UITableView *tableView;
@end

NS_ASSUME_NONNULL_END
