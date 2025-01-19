#import <Foundation/Foundation.h>

@interface FileServer : NSObject

- (void)start;
@property (nonatomic, strong) NSMutableDictionary *segmentsDict;

@end
