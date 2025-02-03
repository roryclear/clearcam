#import <Foundation/Foundation.h>
#import "PortScanner.h"

@interface FileServer : NSObject

- (void)start;
@property (nonatomic, strong) NSMutableDictionary *segmentsDict;
@property (nonatomic, assign) NSInteger segment_length;
@property (nonatomic, strong) NSDate *last_req_time;
@property (nonatomic, strong) PortScanner *scanner;

@end
