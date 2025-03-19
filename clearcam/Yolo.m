#import "Yolo.h"
#import <Metal/Metal.h>
#import <UIKit/UIKit.h>
#import <sys/utsname.h>
#import "SettingsManager.h"

@implementation Yolo

id<MTLDevice> device;
NSMutableDictionary<NSString *, id> *pipeline_states;
NSMutableDictionary<NSString *, id> *buffers;
id<MTLCommandQueue> mtl_queue;
NSMutableArray<id<MTLCommandBuffer>> *mtl_buffers_in_flight;
int yolo_res;
NSArray *yolo_classes;
CFDataRef data;
NSMutableDictionary *_h;
NSMutableArray *_q;
NSString *input_buffer;
NSString *output_buffer;
UInt8 *rgbData;

- (instancetype)init {
    self = [super init];
    self.device = MTLCreateSystemDefaultDevice();
    self.pipeline_states = [[NSMutableDictionary alloc] init];
    self.buffers = [[NSMutableDictionary alloc] init];
    self.mtl_queue = [self.device newCommandQueueWithMaxCommandBufferCount:1024];
    self.mtl_buffers_in_flight = [[NSMutableArray alloc] init];
    self.yolo_res = 640;
    self.rgbData = (UInt8 *)malloc(self.yolo_res * self.yolo_res * 3);
    
    self.yolo_classes = @[
        @[@"person", [UIColor redColor]],@[@"bicycle", [UIColor greenColor]],@[@"car", [UIColor blueColor]],
        @[@"motorcycle", [UIColor cyanColor]],@[@"airplane", [UIColor magentaColor]],@[@"bus", [UIColor yellowColor]],
        @[@"train", [UIColor orangeColor]],@[@"truck", [UIColor purpleColor]],@[@"boat", [UIColor brownColor]],
        @[@"traffic light", [UIColor colorWithRed:0.5 green:0.7 blue:0.2 alpha:1.0]],@[@"fire hydrant", [UIColor colorWithRed:0.8 green:0.1 blue:0.1 alpha:1.0]],
        @[@"stop sign", [UIColor colorWithRed:0.3 green:0.3 blue:0.8 alpha:1.0]],@[@"parking meter", [UIColor colorWithRed:0.7 green:0.5 blue:0.3 alpha:1.0]],
        @[@"bench", [UIColor colorWithRed:0.4 green:0.4 blue:0.2 alpha:1.0]],@[@"bird", [UIColor colorWithRed:0.1 green:0.5 blue:0.9 alpha:1.0]],
        @[@"cat", [UIColor colorWithRed:0.8 green:0.2 blue:0.6 alpha:1.0]],@[@"dog", [UIColor colorWithRed:0.9 green:0.3 blue:0.3 alpha:1.0]],
        @[@"horse", [UIColor colorWithRed:0.2 green:0.6 blue:0.7 alpha:1.0]],@[@"sheep", [UIColor colorWithRed:0.7 green:0.3 blue:0.5 alpha:1.0]],
        @[@"cow", [UIColor colorWithRed:0.4 green:0.8 blue:0.4 alpha:1.0]],@[@"elephant", [UIColor colorWithRed:0.3 green:0.4 blue:0.9 alpha:1.0]],
        @[@"bear", [UIColor colorWithRed:0.6 green:0.2 blue:0.8 alpha:1.0]],@[@"zebra", [UIColor colorWithRed:0.8 green:0.5 blue:0.2 alpha:1.0]],
        @[@"giraffe", [UIColor colorWithRed:0.5 green:0.9 blue:0.1 alpha:1.0]],@[@"backpack", [UIColor colorWithRed:0.3 green:0.7 blue:0.4 alpha:1.0]],
        @[@"umbrella", [UIColor colorWithRed:0.4 green:0.6 blue:0.9 alpha:1.0]],@[@"handbag", [UIColor colorWithRed:0.9 green:0.2 blue:0.5 alpha:1.0]],
        @[@"tie", [UIColor colorWithRed:0.5 green:0.3 blue:0.7 alpha:1.0]],@[@"suitcase", [UIColor colorWithRed:0.6 green:0.7 blue:0.2 alpha:1.0]],
        @[@"frisbee", [UIColor colorWithRed:0.7 green:0.2 blue:0.4 alpha:1.0]],@[@"skis", [UIColor colorWithRed:0.3 green:0.9 blue:0.3 alpha:1.0]],
        @[@"snowboard", [UIColor colorWithRed:0.8 green:0.1 blue:0.6 alpha:1.0]],@[@"sports ball", [UIColor colorWithRed:0.4 green:0.3 blue:0.8 alpha:1.0]],
        @[@"kite", [UIColor colorWithRed:0.2 green:0.5 blue:0.7 alpha:1.0]],@[@"baseball bat", [UIColor colorWithRed:0.6 green:0.4 blue:0.2 alpha:1.0]],
        @[@"baseball glove", [UIColor colorWithRed:0.7 green:0.1 blue:0.4 alpha:1.0]],@[@"skateboard", [UIColor colorWithRed:0.5 green:0.8 blue:0.5 alpha:1.0]],
        @[@"surfboard", [UIColor colorWithRed:0.8 green:0.3 blue:0.6 alpha:1.0]],@[@"tennis racket", [UIColor colorWithRed:0.2 green:0.7 blue:0.9 alpha:1.0]],
        @[@"bottle", [UIColor colorWithRed:0.9 green:0.2 blue:0.3 alpha:1.0]],@[@"wine glass", [UIColor colorWithRed:0.6 green:0.6 blue:0.3 alpha:1.0]],
        @[@"cup", [UIColor colorWithRed:0.3 green:0.4 blue:0.9 alpha:1.0]],@[@"fork", [UIColor colorWithRed:0.4 green:0.7 blue:0.2 alpha:1.0]],
        @[@"knife", [UIColor colorWithRed:0.8 green:0.2 blue:0.5 alpha:1.0]],@[@"spoon", [UIColor colorWithRed:0.6 green:0.3 blue:0.7 alpha:1.0]],
        @[@"bowl", [UIColor colorWithRed:0.2 green:0.8 blue:0.4 alpha:1.0]],@[@"banana", [UIColor colorWithRed:0.7 green:0.7 blue:0.1 alpha:1.0]],
        @[@"apple", [UIColor colorWithRed:0.9 green:0.1 blue:0.4 alpha:1.0]],@[@"sandwich", [UIColor colorWithRed:0.4 green:0.5 blue:0.8 alpha:1.0]],
        @[@"orange", [UIColor colorWithRed:0.8 green:0.6 blue:0.2 alpha:1.0]],@[@"broccoli", [UIColor colorWithRed:0.3 green:0.8 blue:0.3 alpha:1.0]],
        @[@"carrot", [UIColor colorWithRed:0.7 green:0.2 blue:0.6 alpha:1.0]],@[@"hot dog", [UIColor colorWithRed:0.9 green:0.3 blue:0.5 alpha:1.0]],
        @[@"pizza", [UIColor colorWithRed:0.5 green:0.3 blue:0.8 alpha:1.0]],@[@"donut", [UIColor colorWithRed:0.8 green:0.1 blue:0.4 alpha:1.0]],
        @[@"cake", [UIColor colorWithRed:0.7 green:0.5 blue:0.1 alpha:1.0]],@[@"chair", [UIColor colorWithRed:0.6 green:0.2 blue:0.4 alpha:1.0]],
        @[@"couch", [UIColor colorWithRed:0.4 green:0.6 blue:0.2 alpha:1.0]],@[@"potted plant", [UIColor colorWithRed:0.8 green:0.4 blue:0.5 alpha:1.0]],
        @[@"bed", [UIColor colorWithRed:0.3 green:0.7 blue:0.7 alpha:1.0]],@[@"dining table", [UIColor colorWithRed:0.5 green:0.8 blue:0.3 alpha:1.0]],
        @[@"toilet", [UIColor colorWithRed:0.7 green:0.4 blue:0.6 alpha:1.0]],@[@"tv", [UIColor colorWithRed:0.9 green:0.5 blue:0.2 alpha:1.0]],
        @[@"laptop", [UIColor colorWithRed:0.6 green:0.3 blue:0.7 alpha:1.0]],@[@"mouse", [UIColor colorWithRed:0.2 green:0.9 blue:0.5 alpha:1.0]],
        @[@"remote", [UIColor colorWithRed:0.8 green:0.4 blue:0.3 alpha:1.0]],@[@"keyboard", [UIColor colorWithRed:0.3 green:0.6 blue:0.8 alpha:1.0]],
        @[@"cell phone", [UIColor colorWithRed:0.7 green:0.3 blue:0.9 alpha:1.0]],@[@"microwave", [UIColor colorWithRed:0.4 green:0.9 blue:0.4 alpha:1.0]],
        @[@"oven", [UIColor colorWithRed:0.5 green:0.7 blue:0.2 alpha:1.0]],@[@"toaster", [UIColor colorWithRed:0.9 green:0.2 blue:0.3 alpha:1.0]],
        @[@"sink", [UIColor colorWithRed:0.6 green:0.8 blue:0.3 alpha:1.0]],@[@"refrigerator", [UIColor colorWithRed:0.8 green:0.4 blue:0.7 alpha:1.0]],
        @[@"book", [UIColor colorWithRed:0.3 green:0.5 blue:0.9 alpha:1.0]],@[@"clock", [UIColor colorWithRed:0.7 green:0.7 blue:0.2 alpha:1.0]],
        @[@"vase", [UIColor colorWithRed:0.9 green:0.4 blue:0.5 alpha:1.0]],@[@"scissors", [UIColor colorWithRed:0.2 green:0.7 blue:0.8 alpha:1.0]],
        @[@"teddy bear", [UIColor colorWithRed:0.6 green:0.3 blue:0.9 alpha:1.0]],@[@"hair drier", [UIColor colorWithRed:0.8 green:0.2 blue:0.3 alpha:1.0]],
        @[@"toothbrush", [UIColor colorWithRed:0.4 green:0.7 blue:0.6 alpha:1.0]]
    ];
    
    NSString *documentsPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
    NSString *path = @"batch_req_%dx%d";
    
    struct utsname systemInfo;
    uname(&systemInfo);
    NSString *deviceModel = [NSString stringWithCString:systemInfo.machine encoding:NSUTF8StringEncoding];

    NSString *urlPath = @"https://raw.githubusercontent.com/roryclear/yolov8-tinygrad-ios/main/batch_req_%dx%d";
    if ([deviceModel isEqualToString:@"iPhone8,4"] || [deviceModel isEqualToString:@"iPhone12,8"]) { //IPHONE SE1 and SE2, TODO finish
        path = @"batch_req_se1_%dx%d";
        urlPath = @"https://raw.githubusercontent.com/roryclear/yolov8-tinygrad-ios/main/batch_req_se1_%dx%d";
    }
    NSString *filePath = [documentsPath stringByAppendingPathComponent:[NSString stringWithFormat:path, self.yolo_res, self.yolo_res]];
    if (![[NSFileManager defaultManager] fileExistsAtPath:filePath]) {
        NSURL *url = [NSURL URLWithString:[NSString stringWithFormat:urlPath, self.yolo_res, self.yolo_res]];
        NSData *data = [NSData dataWithContentsOfURL:url];
        [data writeToFile:filePath atomically:YES];
    }

    NSData *ns_data = [NSData dataWithContentsOfFile:filePath];
    self.data = CFDataCreate(NULL, [ns_data bytes], [ns_data length]);
    
    const UInt8 *bytes = CFDataGetBytePtr(self.data);
    NSInteger length = CFDataGetLength(self.data);
    NSData *range_data;
    self._h = [[NSMutableDictionary alloc] init];
    NSInteger ptr = 0;
    NSString *string_data;
    NSMutableString *datahash = [NSMutableString stringWithCapacity:0x40];
    while (ptr < length) {
        NSData *slicedData = [NSData dataWithBytes:bytes + ptr + 0x20 length:0x28 - 0x20];
        uint64_t datalen = 0;
        [slicedData getBytes:&datalen length:sizeof(datalen)];
        datalen = CFSwapInt64LittleToHost(datalen);
        const UInt8 *datahash_bytes = bytes + ptr;
        datahash = [NSMutableString stringWithCapacity:0x40];
        for (int i = 0; i < 0x20; i++) {
            [datahash appendFormat:@"%02x", datahash_bytes[i]];
        }
        range_data = [NSData dataWithBytes:bytes + (ptr + 0x28) length:datalen];
        self._h[datahash] = range_data;
        ptr += 0x28 + datalen;
    }
    CFRelease(self.data);
    string_data = [[NSString alloc] initWithData:range_data encoding:NSUTF8StringEncoding];
    self._q = [NSMutableArray array];
    NSMutableArray *_q_exec = [NSMutableArray array];
    NSArray *ops = @[@"BufferAlloc", @"CopyIn", @"ProgramAlloc",@"ProgramExec",@"CopyOut"];
    NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:[NSString stringWithFormat:@"(%@)\\(", [ops componentsJoinedByString:@"|"]] options:0 error:nil];
    __block NSInteger lastIndex = 0;
    [regex enumerateMatchesInString:string_data options:0 range:NSMakeRange(0, string_data.length) usingBlock:^(NSTextCheckingResult *match, NSMatchingFlags flags, BOOL *stop) {
        [self._q addObject:[self extractValues:([[string_data substringWithRange:NSMakeRange(lastIndex, match.range.location - lastIndex)] stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@", "]])]];
        lastIndex = match.range.location;
    }];
    [self._q addObject:[self extractValues:([[string_data substringFromIndex:lastIndex] stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@", "]])]];
    for (NSMutableDictionary *values in self._q) {
        if ([values[@"op"] isEqualToString:@"BufferAlloc"]) {
            [self.buffers setObject:[self.device newBufferWithLength:[values[@"size"][0] intValue] options:MTLResourceStorageModeShared] forKey:values[@"buffer_num"][0]];
        } else if ([values[@"op"] isEqualToString:@"CopyIn"]) {
            id<MTLBuffer> buffer = self.buffers[values[@"buffer_num"][0]];
            NSData *data = self._h[values[@"datahash"][0]];
            memcpy(buffer.contents, data.bytes, data.length);
            self.input_buffer = values[@"buffer_num"][0];
        } else if ([values[@"op"] isEqualToString:@"ProgramAlloc"]) {
            NSString *key = [NSString stringWithFormat:@"%@_%@", values[@"name"][0], values[@"datahash"][0]];
            if ([self.pipeline_states objectForKey:key]) continue;
            NSString *prg = [[NSString alloc] initWithData:self._h[values[@"datahash"][0]] encoding:NSUTF8StringEncoding];
            NSError *error = nil;
            id<MTLLibrary> library = [self.device newLibraryWithSource:prg
                                                          options:nil
                                                            error:&error];
            MTLComputePipelineDescriptor *descriptor = [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = [library newFunctionWithName:values[@"name"][0]];;
            descriptor.supportIndirectCommandBuffers = YES;
            MTLComputePipelineReflection *reflection = nil;
            id<MTLComputePipelineState> pipeline_state = [self.device newComputePipelineStateWithDescriptor:descriptor
                                                                                               options:MTLPipelineOptionNone
                                                                                            reflection:&reflection
                                                                                                 error:&error];
            [self.pipeline_states setObject:pipeline_state forKey:@[values[@"name"][0],values[@"datahash"][0]]];
        } else if ([values[@"op"] isEqualToString:@"ProgramExec"]) {
            [_q_exec addObject:values];
        } else if ([values[@"op"] isEqualToString:@"CopyOut"]) {
            self.output_buffer = values[@"buffer_num"][0];
        }
    }
    self._q = [_q_exec mutableCopy];
    [self._h removeAllObjects];
    
    return self;
}

- (NSArray *)processOutput:(const float *)output {
    NSArray<NSNumber *> *yoloIndexes = [[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_presets"][[[NSUserDefaults standardUserDefaults] objectForKey:@"yolo_preset_idx"]];
    NSMutableArray *boxes = [NSMutableArray array];
    int numPredictions = pow(self.yolo_res / 32, 2) * 21;

    for (int index = 0; index < numPredictions; index++) {
        int classId = 0;
        float prob = 0.0;
        
        for (int i = 0; i < yoloIndexes.count; i++) {
            float confidence = output[numPredictions * ([yoloIndexes[i] intValue] + 4) + index];
            if (confidence > prob) {
                prob = confidence;
                classId = [yoloIndexes[i] intValue];
            }
        }

        if (prob < 0.25) continue;

        float xc = output[index];
        float yc = output[numPredictions + index];
        float w = output[2 * numPredictions + index];
        float h = output[3 * numPredictions + index];

        float x1 = (xc - w / 2);
        float y1 = (yc - h / 2);
        float x2 = (xc + w / 2);
        float y2 = (yc + h / 2);

        [boxes addObject:@[@(x1), @(y1), @(x2), @(y2), @(classId), @(prob)]];
    }

    [boxes sortUsingComparator:^NSComparisonResult(NSArray *box1, NSArray *box2) {
        NSNumber *prob1 = box1[5];
        NSNumber *prob2 = box2[5];
        return [prob2 compare:prob1];
    }];

    NSMutableArray *result = [NSMutableArray array];
    while ([boxes count] > 0) {
        NSArray *bestBox = boxes[0];
        [result addObject:bestBox];
        [boxes removeObjectAtIndex:0];

        NSMutableArray *filteredBoxes = [NSMutableArray array];
        for (NSArray *box in boxes) {
            if ([self iouBetweenBox:bestBox andBox:box] < 0.7) {
                [filteredBoxes addObject:box];
            }
        }
        boxes = filteredBoxes;
    }
    return result;
}

// Calculate intersection between two boxes
- (CGFloat)intersectionBetweenBox:(NSArray *)box1 andBox:(NSArray *)box2 {
    CGFloat x1 = MAX([box1[0] floatValue], [box2[0] floatValue]);
    CGFloat y1 = MAX([box1[1] floatValue], [box2[1] floatValue]);
    CGFloat x2 = MIN([box1[2] floatValue], [box2[2] floatValue]);
    CGFloat y2 = MIN([box1[3] floatValue], [box2[3] floatValue]);
    return MAX(0, x2 - x1) * MAX(0, y2 - y1);
}

// Calculate union between two boxes
- (CGFloat)unionBetweenBox:(NSArray *)box1 andBox:(NSArray *)box2 {
    CGFloat box1Area = ([box1[2] floatValue] - [box1[0] floatValue]) * ([box1[3] floatValue] - [box1[1] floatValue]);
    CGFloat box2Area = ([box2[2] floatValue] - [box2[0] floatValue]) * ([box2[3] floatValue] - [box2[1] floatValue]);
    return box1Area + box2Area - [self intersectionBetweenBox:box1 andBox:box2];
}

// Calculate Intersection over Union (IoU) between two boxes
- (CGFloat)iouBetweenBox:(NSArray *)box1 andBox:(NSArray *)box2 {
    return [self intersectionBetweenBox:box1 andBox:box2] / [self unionBetweenBox:box1 andBox:box2];
}

// Extract values from a string
- (NSMutableDictionary<NSString *, id> *)extractValues:(NSString *)x {
    NSMutableDictionary<NSString *, id> *values = [@{@"op": [x componentsSeparatedByString:@"("][0]} mutableCopy];
    NSDictionary<NSString *, NSString *> *patterns = @{@"name": @"name='([^']+)'", @"datahash": @"datahash='([^']+)'", @"global_sizes": @"global_size=\\(([^)]+)\\)",
                                                       @"local_sizes": @"local_size=\\(([^)]+)\\)", @"bufs": @"bufs=\\(([^)]+)",
                                                       @"vals": @"vals=\\(([^)]+)", @"buffer_num": @"buffer_num=(\\d+)", @"size": @"size=(\\d+)"};
    [patterns enumerateKeysAndObjectsUsingBlock:^(NSString *key, NSString *pattern, BOOL *stop) {
        NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:nil];
        NSTextCheckingResult *match = [regex firstMatchInString:x options:0 range:NSMakeRange(0, x.length)];
        if (match) {
            NSString *contents = [x substringWithRange:[match rangeAtIndex:1]];
            NSMutableArray<NSString *> *extracted_values = [NSMutableArray array];
            for (NSString *value in [contents componentsSeparatedByString:@","]) {
                NSString *trimmed_value = [value stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
                if (trimmed_value.length > 0) {
                    [extracted_values addObject:trimmed_value];
                }
            }
            values[key] = [extracted_values copy];
        }
    }];
    return values;
}

- (NSArray *)yolo_infer:(CGImageRef)cgImage withOrientation:(AVCaptureVideoOrientation)orientation {
    CFDataRef rawData = CGDataProviderCopyData(CGImageGetDataProvider(cgImage));
    if (!rawData) {
        NSLog(@"Failed to get raw image data");
        return nil;
    }

    const UInt8 *rawBytes = CFDataGetBytePtr(rawData);
    size_t length = CFDataGetLength(rawData);
    size_t width = CGImageGetWidth(cgImage);
    size_t height = CGImageGetHeight(cgImage);
    size_t rgbLength = (length / 4) * 3;
    
    if (orientation == AVCaptureVideoOrientationLandscapeRight) {
        for (size_t i = 0, j = 0; i < length; i += 4, j += 3) {
            self.rgbData[j] = rawBytes[i];
            self.rgbData[j + 1] = rawBytes[i + 1];
            self.rgbData[j + 2] = rawBytes[i + 2];
        }
    } else if (orientation == AVCaptureVideoOrientationLandscapeLeft) {
        for (size_t i = 0, j = 0; i < length; i += 4, j += 3) {
            self.rgbData[rgbLength - 1 - j - 2] = rawBytes[i];
            self.rgbData[rgbLength - 1 - j - 1] = rawBytes[i + 1];
            self.rgbData[rgbLength - 1 - j] = rawBytes[i + 2];
        }
    } else if (orientation == AVCaptureVideoOrientationPortrait) {
        for (size_t i = 0; i < length; i += 4) {
            int row = (int)(i / (width * 4));
            int col = (int)((i % (width * 4)) / 4);
            self.rgbData[col * (width * 3) + ((height - 1 - row) * 3)] = rawBytes[i];
            self.rgbData[col * (width * 3) + ((height - 1 - row) * 3) + 1] = rawBytes[i + 1];
            self.rgbData[col * (width * 3) + ((height - 1 - row) * 3) + 2] = rawBytes[i + 2];
        }
    }

    CFRelease(rawData);

    id<MTLBuffer> buffer = self.buffers[self.input_buffer];
    if (!buffer || !buffer.contents) {
        NSLog(@"Metal buffer is not initialized or invalid");
        return nil;
    }

    memset(buffer.contents, 0, buffer.length);
    memcpy(buffer.contents, self.rgbData, MIN(buffer.length, width * width * 3));

    for (NSMutableDictionary *values in self._q) {
        id<MTLCommandBuffer> commandBuffer = [self.mtl_queue commandBuffer];
        if (!commandBuffer) {
            NSLog(@"Failed to create Metal command buffer");
            continue;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            NSLog(@"Failed to create compute command encoder");
            continue;
        }

        [encoder setComputePipelineState:self.pipeline_states[@[values[@"name"][0], values[@"datahash"][0]]]];
        for (int i = 0; i < [(NSArray *)values[@"bufs"] count]; i++) {
            [encoder setBuffer:self.buffers[values[@"bufs"][i]] offset:0 atIndex:i];
        }
        for (int i = 0; i < [(NSArray *)values[@"vals"] count]; i++) {
            NSInteger value = [values[@"vals"][i] integerValue];
            [encoder setBytes:&value length:sizeof(NSInteger) atIndex:i + [(NSArray *)values[@"bufs"] count]];
        }

        MTLSize globalSize = MTLSizeMake([values[@"global_sizes"][0] intValue], [values[@"global_sizes"][1] intValue], [values[@"global_sizes"][2] intValue]);
        MTLSize localSize = MTLSizeMake([values[@"local_sizes"][0] intValue], [values[@"local_sizes"][1] intValue], [values[@"local_sizes"][2] intValue]);
        [encoder dispatchThreadgroups:globalSize threadsPerThreadgroup:localSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [self.mtl_buffers_in_flight addObject:commandBuffer];
    }

    for (int i = 0; i < self.mtl_buffers_in_flight.count; i++) {
        [self.mtl_buffers_in_flight[i] waitUntilCompleted];
    }
    [self.mtl_buffers_in_flight removeAllObjects];

    buffer = self.buffers[self.output_buffer];
    if (!buffer || !buffer.contents) {
        NSLog(@"Output buffer is invalid");
        return nil;
    }

    const void *bufferPointer = buffer.contents;
    float *floatArray = malloc(buffer.length);
    if (!floatArray) {
        NSLog(@"Memory allocation failed for output array");
        return nil;
    }

    memcpy(floatArray, bufferPointer, buffer.length);
    NSArray *output = [self processOutput:floatArray];
    free(floatArray);
    
    NSMutableString *classNamesString = [NSMutableString string];
    for (int i = 0; i < output.count; i++) {
        [classNamesString appendString:self.yolo_classes[[output[i][4] intValue]][0]];
        if (i < output.count - 1) [classNamesString appendString:@", "];
    }
    /* todo, move or remove?
    if(output.count > 0){
        NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
        [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
        NSString *currentTimestamp = [dateFormatter stringFromDate:[NSDate date]];
        NSString *logEntry = [NSString stringWithFormat:@"%@ - Class Names: %@", currentTimestamp, classNamesString];
        NSString *filePath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0] stringByAppendingPathComponent:@"logs.txt"];
        NSFileHandle *fileHandle = [NSFileHandle fileHandleForWritingAtPath:filePath] ?: [NSFileHandle fileHandleForWritingAtPath:([[NSFileManager defaultManager] createFileAtPath:filePath contents:nil attributes:nil], filePath)];
        [fileHandle seekToEndOfFile];
        [fileHandle writeData:[[logEntry stringByAppendingString:@"\n"] dataUsingEncoding:NSUTF8StringEncoding]];
        [fileHandle closeFile];
    }
    */
    return output;
}

@end
