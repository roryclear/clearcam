#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Yolo : NSObject

@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) NSMutableDictionary<NSString *, id> *pipeline_states;
@property (nonatomic, strong) NSMutableDictionary<NSString *, id> *buffers;
@property (nonatomic, strong) id<MTLCommandQueue> mtl_queue;
@property (nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *mtl_buffers_in_flight;
@property (nonatomic, assign) int yolo_res;
@property (nonatomic, strong) NSArray *yolo_classes;
@property (nonatomic, assign) CFDataRef data;
@property (nonatomic, strong) NSMutableDictionary *_h;
@property (nonatomic, strong) NSMutableArray *_q;
@property (nonatomic, strong) NSString *input_buffer;
@property (nonatomic, strong) NSString *output_buffer;
@property (nonatomic, assign) UInt8 *rgbData;
- (NSArray *)yolo_infer:(CGImageRef)cgImage withOrientation:(AVCaptureVideoOrientation)orientation;

// Initializer
- (instancetype)init;

- (CGFloat)intersectionBetweenBox:(NSArray *)box1 andBox:(NSArray *)box2;
- (CGFloat)unionBetweenBox:(NSArray *)box1 andBox:(NSArray *)box2;
- (CGFloat)iouBetweenBox:(NSArray *)box1 andBox:(NSArray *)box2;
- (NSMutableDictionary<NSString *, id> *)extractValues:(NSString *)x;
- (NSArray *)processOutput:(const float *)output;

@end

NS_ASSUME_NONNULL_END

