#import <Foundation/Foundation.h>

@interface SettingsManager : NSObject

@property (nonatomic, strong) NSArray<NSNumber *> *yolo_indexes;
@property (nonatomic, strong) NSArray<NSNumber *> *events;


+ (instancetype)sharedManager;
- (void)saveYoloIndexes;
- (void)loadYoloIndexes;
- (void)updateYoloIndexes:(NSArray<NSNumber *> *)newIndexes;

@end
