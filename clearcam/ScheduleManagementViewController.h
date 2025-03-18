#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface ScheduleManagementViewController : UITableViewController

@property (nonatomic, strong) NSMutableArray<NSDictionary *> *emailSchedules;
@property (nonatomic, copy) void (^completionHandler)(NSArray<NSDictionary *> *schedules);

@end

NS_ASSUME_NONNULL_END
