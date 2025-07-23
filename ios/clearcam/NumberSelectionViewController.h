#import <UIKit/UIKit.h>

@interface NumberSelectionViewController : UIViewController <UITableViewDelegate, UITableViewDataSource>

@property (nonatomic, strong) NSString *presetKey; // The preset being edited
@property (nonatomic, strong) NSMutableArray<NSNumber *> *selectedIndexes; // Selected numbers (0-79)
@property (nonatomic, copy) void (^completionHandler)(NSArray<NSNumber *> *selectedIndexes); // Callback to save the selected numbers

@end
