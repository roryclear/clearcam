#import <Foundation/Foundation.h>
#import "PortScanner.h"
#import <CoreData/CoreData.h>

@interface FileServer : NSObject

+ (instancetype)sharedInstance;

- (void)start;
- (NSArray *)fetchEventDataFromCoreData:(NSManagedObjectContext *)context;
- (NSString *)processVideoDownloadWithLowRes:(BOOL)low_res
                                  startTime:(NSTimeInterval)startTimeStamp
                                    endTime:(NSTimeInterval)endTimeStamp
                                     context:(NSManagedObjectContext *)context;
+ (void)performPostRequestWithURL:(NSString *)urlString
                           method:(NSString *)method
                      contentType:(NSString *)contentType
                             body:(id)body
                completionHandler:(void (^)(NSData *data, NSHTTPURLResponse *response, NSError *error))completion;
@property (nonatomic, assign) NSInteger segment_length;
@property (nonatomic, strong) NSDate *last_req_time;
@property (nonatomic, strong) PortScanner *scanner;
@property (nonatomic, strong) NSManagedObjectContext *context;

@end
