#import <Foundation/Foundation.h>

@interface SettingsManager : NSObject

@property (nonatomic, strong) NSArray<NSNumber *> *events;
@property (nonatomic, strong) NSDictionary *alerts;
@property (nonatomic, assign) BOOL *delete_on_launch;

@property (nonatomic, strong) NSString *width;
@property (nonatomic, strong) NSString *height;
@property (nonatomic, strong) NSString *text_size;
@property (nonatomic, strong) NSString *preset; //todo, resolution
@property (nonatomic, strong) NSString *yolo_preset;
@property (nonatomic, strong) NSMutableDictionary *yolo_presets;


+ (instancetype)sharedManager;
- (void)saveYoloIndexes;
- (void)loadYoloIndexes;
- (void)updateYoloIndexesKey:(NSString *)key;
- (void)loadResolutionSettings;
- (void)saveResolutionSettings;
- (void)updateResolutionWithWidth:(NSString *)width height:(NSString *)height textSize:(NSString *)textSize preset:(NSString *)preset;

@end
