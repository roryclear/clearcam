#import <Foundation/Foundation.h>
#import <CoreData/CoreData.h>
#import <AVFoundation/AVFoundation.h>
#import "pgp.h"

@interface SceneState : NSObject

@property (nonatomic, strong) NSMutableArray<NSDictionary *> *lastN;
@property (nonatomic, strong) NSMutableDictionary *lastN_total;
@property (nonatomic, strong) NSArray<NSNumber *> *events;
@property (nonatomic, strong) NSDictionary *alerts;
@property (strong, nonatomic) NSManagedObjectContext *backgroundContext;
@property (strong, nonatomic) NSDate *last_email_time;
@property (strong, nonatomic) PGP *pgp;
- (void)sendEmailWithImageAtPath:(NSString *)imagePath;
- (void)processOutput:(NSArray *)array withImage:(CIImage *)image;
- (void)writeToFileWithString:(NSString *)customString;

@end
