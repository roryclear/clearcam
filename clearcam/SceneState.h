#import <Foundation/Foundation.h>

@interface SceneState : NSObject

@property (nonatomic, strong) NSMutableArray<NSDictionary *> *lastN;
@property (nonatomic, strong) NSMutableDictionary *lastN_total;
@property (nonatomic, strong) NSMutableDictionary *lastN_state;
@property (nonatomic, strong) NSArray<NSNumber *> *events;
- (void)processOutput:(NSArray *)array;
- (void)writeToFileWithString:(NSString *)customString;

@end
