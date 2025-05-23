#import <Foundation/Foundation.h>
#import <CoreData/CoreData.h>
#import <AVFoundation/AVFoundation.h>

@interface SceneState : NSObject

@property (nonatomic, strong) NSMutableArray<NSDictionary *> *lastN;
@property (nonatomic, strong) NSMutableDictionary *lastN_total;
@property (nonatomic, strong) NSArray<NSNumber *> *events;
@property (nonatomic, strong) NSDictionary *alerts;
@property (strong, nonatomic) NSManagedObjectContext *backgroundContext;
@property (strong, nonatomic) NSDate *last_notif_time;
@property (strong, nonatomic) NSDate *left_live_time;
- (void)processOutput:(NSArray *)array withImage:(CIImage *)image orientation:(AVCaptureVideoOrientation)orientation;
- (void)sendnotification; //todo, remove

@end
