#import <Foundation/Foundation.h>

@interface SettingsManager : NSObject

@property (nonatomic, strong) NSArray<NSNumber *> *yolo_indexes;
@property (nonatomic, strong) NSArray<NSNumber *> *events;
@property (nonatomic, strong) NSDictionary *alerts;


+ (instancetype)sharedManager;
- (void)saveYoloIndexes;
- (void)loadYoloIndexes;
- (void)updateYoloIndexes:(NSArray<NSNumber *> *)newIndexes;

@end
