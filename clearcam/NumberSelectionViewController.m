#import <UIKit/UIKit.h>

@interface NumberSelectionViewController : UIViewController <UITableViewDelegate, UITableViewDataSource>

@property (nonatomic, strong) NSString *presetKey; // The preset being edited
@property (nonatomic, strong) NSMutableArray<NSNumber *> *selectedIndexes; // Selected numbers (0-79)
@property (nonatomic, copy) void (^completionHandler)(NSArray<NSNumber *> *selectedIndexes); // Callback to save the selected numbers
@property (nonatomic, strong) NSArray<NSString *> *classes; // Selected numbers (0-79)

@end

@implementation NumberSelectionViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    self.classes = @[
        @"person", @"bicycle", @"car", @"motorcycle", @"airplane",
        @"bus", @"train", @"truck", @"boat", @"traffic light",
        @"fire hydrant", @"stop sign", @"parking meter", @"bench",
        @"bird", @"cat", @"dog", @"horse", @"sheep", @"cow",
        @"elephant", @"bear", @"zebra", @"giraffe", @"backpack",
        @"umbrella", @"handbag", @"tie", @"suitcase", @"frisbee",
        @"skis", @"snowboard", @"sports ball", @"kite", @"baseball bat",
        @"baseball glove", @"skateboard", @"surfboard", @"tennis racket",
        @"bottle", @"wine glass", @"cup", @"fork", @"knife",
        @"spoon", @"bowl", @"banana", @"apple", @"sandwich",
        @"orange", @"broccoli", @"carrot", @"hot dog", @"pizza",
        @"donut", @"cake", @"chair", @"couch", @"potted plant",
        @"bed", @"dining table", @"toilet", @"tv", @"laptop",
        @"mouse", @"remote", @"keyboard", @"cell phone", @"microwave",
        @"oven", @"toaster", @"sink", @"refrigerator", @"book",
        @"clock", @"vase", @"scissors", @"teddy bear", @"hair drier",
        @"toothbrush"
    ];
    
    self.title = self.presetKey ?: @"New Preset";
    self.view.backgroundColor = [UIColor systemBackgroundColor];

    // Create table view
    UITableView *tableView = [[UITableView alloc] initWithFrame:self.view.bounds style:UITableViewStyleGrouped];
    tableView.delegate = self;
    tableView.dataSource = self;
    [self.view addSubview:tableView];

    // Add a "Save" button
    UIBarButtonItem *saveButton = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemSave target:self action:@selector(savePreset)];
    self.navigationItem.rightBarButtonItem = saveButton;
}

#pragma mark - UITableView DataSource

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return 80; // Numbers 0-79
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"NumberCell"];
    if (!cell) {
        cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleDefault reuseIdentifier:@"NumberCell"];
    }
    cell.textLabel.text = [NSString stringWithFormat:@"%@", self.classes[indexPath.row]];
    cell.accessoryType = [self.selectedIndexes containsObject:@(indexPath.row)] ? UITableViewCellAccessoryCheckmark : UITableViewCellAccessoryNone;
    return cell;
}

#pragma mark - UITableView Delegate

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    NSNumber *number = @(indexPath.row);
    if ([self.selectedIndexes containsObject:number]) {
        [self.selectedIndexes removeObject:number];
    } else {
        [self.selectedIndexes addObject:number];
    }
    [tableView reloadRowsAtIndexPaths:@[indexPath] withRowAnimation:UITableViewRowAnimationAutomatic];
}

#pragma mark - Save Preset

- (void)savePreset {
    if (self.completionHandler) {
        self.completionHandler(self.selectedIndexes);
    }
    [self.navigationController popViewControllerAnimated:YES];
}

@end
