#import <Foundation/Foundation.h>

@interface SceneState : NSObject

@property (nonatomic, strong) NSMutableArray<NSDictionary *> *lastN;
@property (nonatomic, strong) NSMutableDictionary *lastN_total;
@property (nonatomic, strong) NSArray<NSNumber *> *events;
@property (nonatomic, strong) NSMutableArray<NSDate *> *event_times;
@property (nonatomic, strong) NSDictionary *alerts;
- (void)processOutput:(NSArray *)array;
- (void)writeToFileWithString:(NSString *)customString;

@end
