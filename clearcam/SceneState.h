#import <Foundation/Foundation.h>
#import <CoreData/CoreData.h>

@interface SceneState : NSObject

@property (nonatomic, strong) NSMutableArray<NSDictionary *> *lastN;
@property (nonatomic, strong) NSMutableDictionary *lastN_total;
@property (nonatomic, strong) NSArray<NSNumber *> *events;
@property (nonatomic, strong) NSDictionary *alerts;
@property (strong, nonatomic) NSManagedObjectContext *backgroundContext;
- (void)processOutput:(NSArray *)array;
- (void)writeToFileWithString:(NSString *)customString;

@end
