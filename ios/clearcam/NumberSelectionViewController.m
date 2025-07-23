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
        NSLocalizedString(@"person", nil),
        NSLocalizedString(@"bicycle", nil),
        NSLocalizedString(@"car", nil),
        NSLocalizedString(@"motorcycle", nil),
        NSLocalizedString(@"airplane", nil),
        NSLocalizedString(@"bus", nil),
        NSLocalizedString(@"train", nil),
        NSLocalizedString(@"truck", nil),
        NSLocalizedString(@"boat", nil),
        NSLocalizedString(@"traffic light", nil),
        NSLocalizedString(@"fire hydrant", nil),
        NSLocalizedString(@"stop sign", nil),
        NSLocalizedString(@"parking meter", nil),
        NSLocalizedString(@"bench", nil),
        NSLocalizedString(@"bird", nil),
        NSLocalizedString(@"cat", nil),
        NSLocalizedString(@"dog", nil),
        NSLocalizedString(@"horse", nil),
        NSLocalizedString(@"sheep", nil),
        NSLocalizedString(@"cow", nil),
        NSLocalizedString(@"elephant", nil),
        NSLocalizedString(@"bear", nil),
        NSLocalizedString(@"zebra", nil),
        NSLocalizedString(@"giraffe", nil),
        NSLocalizedString(@"backpack", nil),
        NSLocalizedString(@"umbrella", nil),
        NSLocalizedString(@"handbag", nil),
        NSLocalizedString(@"tie", nil),
        NSLocalizedString(@"suitcase", nil),
        NSLocalizedString(@"frisbee", nil),
        NSLocalizedString(@"skis", nil),
        NSLocalizedString(@"snowboard", nil),
        NSLocalizedString(@"sports ball", nil),
        NSLocalizedString(@"kite", nil),
        NSLocalizedString(@"baseball bat", nil),
        NSLocalizedString(@"baseball glove", nil),
        NSLocalizedString(@"skateboard", nil),
        NSLocalizedString(@"surfboard", nil),
        NSLocalizedString(@"tennis racket", nil),
        NSLocalizedString(@"bottle", nil),
        NSLocalizedString(@"wine glass", nil),
        NSLocalizedString(@"cup", nil),
        NSLocalizedString(@"fork", nil),
        NSLocalizedString(@"knife", nil),
        NSLocalizedString(@"spoon", nil),
        NSLocalizedString(@"bowl", nil),
        NSLocalizedString(@"banana", nil),
        NSLocalizedString(@"apple", nil),
        NSLocalizedString(@"sandwich", nil),
        NSLocalizedString(@"orange", nil),
        NSLocalizedString(@"broccoli", nil),
        NSLocalizedString(@"carrot", nil),
        NSLocalizedString(@"hot dog", nil),
        NSLocalizedString(@"pizza", nil),
        NSLocalizedString(@"donut", nil),
        NSLocalizedString(@"cake", nil),
        NSLocalizedString(@"chair", nil),
        NSLocalizedString(@"couch", nil),
        NSLocalizedString(@"potted plant", nil),
        NSLocalizedString(@"bed", nil),
        NSLocalizedString(@"dining table", nil),
        NSLocalizedString(@"toilet", nil),
        NSLocalizedString(@"tv", nil),
        NSLocalizedString(@"laptop", nil),
        NSLocalizedString(@"mouse", nil),
        NSLocalizedString(@"remote", nil),
        NSLocalizedString(@"keyboard", nil),
        NSLocalizedString(@"cell phone", nil),
        NSLocalizedString(@"microwave", nil),
        NSLocalizedString(@"oven", nil),
        NSLocalizedString(@"toaster", nil),
        NSLocalizedString(@"sink", nil),
        NSLocalizedString(@"refrigerator", nil),
        NSLocalizedString(@"book", nil),
        NSLocalizedString(@"clock", nil),
        NSLocalizedString(@"vase", nil),
        NSLocalizedString(@"scissors", nil),
        NSLocalizedString(@"teddy bear", nil),
        NSLocalizedString(@"hair drier", nil),
        NSLocalizedString(@"toothbrush", nil)
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
