#import <Foundation/Foundation.h>
#import "PortScanner.h"
#import <CoreData/CoreData.h>

@interface FileServer : NSObject

- (void)start;
- (void)fetchAndProcessSegmentsFromCoreDataForDateParam:(NSString *)dateParam context:(NSManagedObjectContext *)context;
@property (nonatomic, strong) NSMutableDictionary *segmentsDict;
@property (nonatomic, assign) NSInteger segment_length;
@property (nonatomic, strong) NSDate *last_req_time;
@property (nonatomic, strong) PortScanner *scanner;
@property (nonatomic, strong) NSManagedObjectContext *context;

@end
