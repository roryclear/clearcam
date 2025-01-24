#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <Metal/Metal.h>
#import "Yolo.h"
#import "FileServer.h"

@interface ViewController ()

@property (nonatomic, strong) AVCaptureSession *captureSession;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *previewLayer;
@property (nonatomic, strong) UILabel *fpsLabel;
@property (nonatomic, assign) CFTimeInterval lastFrameTime;
@property (nonatomic, assign) NSUInteger frameCount;
@property (nonatomic, strong) Yolo *yolo;
@property (nonatomic, strong) CIContext *ciContext;

@property (atomic, assign) BOOL isProcessing;
@property (nonatomic, assign) BOOL isRecording;
@property (nonatomic, assign) CMTime startTime;
@property (nonatomic, assign) CMTime currentTime;
@property (nonatomic, strong) AVAssetWriter *assetWriter;
@property (nonatomic, strong) AVAssetWriterInput *videoWriterInput;
@property (nonatomic, strong) AVAssetWriterInputPixelBufferAdaptor *adaptor;
@property (nonatomic, strong) FileServer *fileServer;
@property (nonatomic, assign) int seg_number;
@property (nonatomic, strong) NSURL *last_file_url;
@property (nonatomic, assign) CMTime last_file_duration;
@property (nonatomic, assign) NSDate *last_file_timestamp;
@property (nonatomic, assign) NSDate *current_file_timestamp;
@property (nonatomic, strong) NSMutableDictionary *digits;
@property (nonatomic, strong) NSMutableArray *last_segment_squares;
@property (nonatomic, strong) NSMutableArray *current_segment_squares;

@end

@implementation ViewController

NSMutableDictionary *classColorMap;

- (void)viewDidLoad {
    [super viewDidLoad];
    self.last_segment_squares = [[NSMutableArray alloc] init];
    self.current_segment_squares = [[NSMutableArray alloc] init];
    self.digits = [NSMutableDictionary dictionary];
    self.digits[@"0"] = @[@[ @0, @0, @3, @1], @[ @0, @1, @1 , @3], @[ @2, @1, @1 , @3], @[ @0, @4, @3 , @1]];
    self.digits[@"1"] = @[@[ @2, @0, @1, @5 ]];
    self.digits[@"2"] = @[@[ @0, @0, @3, @1 ], @[ @2, @1, @1, @1 ], @[ @0, @2, @3, @1 ], @[ @0, @3, @1, @1 ], @[ @0, @4, @3, @1 ]];
    self.digits[@"3"] = @[@[ @0, @0, @3, @1 ], @[ @2, @1, @1, @3 ], @[ @0, @2, @3, @1 ], @[ @0, @4, @3, @1 ]];
    self.digits[@"4"] = @[@[ @2, @0, @1, @5 ], @[ @0, @0, @1, @2 ], @[ @0, @2, @3, @1 ]];
    self.digits[@"5"] = @[@[ @0, @0, @3, @1 ], @[ @0, @1, @1, @1 ], @[ @0, @2, @3, @1 ], @[ @2, @3, @1, @1 ], @[ @0, @4, @3, @1 ]];
    self.digits[@"6"] = @[@[ @0, @0, @1, @5 ], @[ @1, @2, @2, @1 ], @[ @1, @4, @2, @1 ], @[ @2, @3, @1, @1 ]];
    self.digits[@"7"] = @[@[ @0, @0, @3, @1 ], @[ @2, @1, @1, @4 ]];
    self.digits[@"8"] = @[@[ @0, @0, @1, @5 ], @[ @2, @0, @1, @5 ], @[ @1, @0, @1, @1 ], @[ @1, @2, @1, @1 ], @[ @1, @4, @1, @1 ]];
    self.digits[@"9"] = @[@[ @2, @0, @1, @5 ], @[ @1, @0, @1, @1 ], @[ @0, @0, @1, @2 ], @[ @0, @2, @3, @1 ]];
    self.digits[@"-"] = @[@[ @0, @2, @3, @1 ]];
    self.digits[@":"] = @[@[ @1, @1, @1, @1 ], @[ @1, @3, @1, @1 ]];
    [[NSFileManager defaultManager] removeItemAtPath:[NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:@"segments.txt"] error:nil]; //TODO remove
    
    //TODO REMOVE THIS
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSURL *documentsURL = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    NSString *documentsPath = [documentsURL path];

    NSError *error;
    NSArray *contents = [fileManager contentsOfDirectoryAtPath:documentsPath error:&error];

    if (error) {
        NSLog(@"Failed to get contents of Documents directory: %@", error.localizedDescription);
        return;
    }

    for (NSString *file in contents) {
        if ([file hasPrefix:@"batch_req"]) continue;

        NSString *filePath = [documentsPath stringByAppendingPathComponent:file];
        BOOL success = [fileManager removeItemAtPath:filePath error:&error];

        if (!success) {
            NSLog(@"Failed to delete %@: %@", file, error.localizedDescription);
        } else {
            NSLog(@"Deleted: %@", file);
        }
    }

    ///TODO
    
    self.ciContext = [CIContext context];
    self.yolo = [[Yolo alloc] init];
    self.seg_number = 0;
    self.fileServer = [[FileServer alloc] init];
    [self.fileServer start];
    
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(handleDeviceOrientationChange)
                                                 name:UIDeviceOrientationDidChangeNotification
                                               object:nil];
    
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
    [self handleDeviceOrientationChange];
    [self setupCamera];
    [self setupFPSLabel];
}

- (void)dealloc {
    [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
}

- (void)handleDeviceOrientationChange {
    UIDeviceOrientation deviceOrientation = [[UIDevice currentDevice] orientation];
    AVCaptureVideoOrientation videoOrientation;
    
    switch (deviceOrientation) {
        case UIDeviceOrientationLandscapeLeft:
            videoOrientation = AVCaptureVideoOrientationLandscapeRight;
            break;
        case UIDeviceOrientationLandscapeRight:
            videoOrientation = AVCaptureVideoOrientationLandscapeLeft;
            break;
        case UIDeviceOrientationPortrait:
            videoOrientation = AVCaptureVideoOrientationPortrait;
            break;
        case UIDeviceOrientationPortraitUpsideDown:
            videoOrientation = AVCaptureVideoOrientationPortraitUpsideDown;
            break;
        default:
            videoOrientation = AVCaptureVideoOrientationLandscapeRight;
            break;
    }
    
    if (self.previewLayer.connection.isVideoOrientationSupported) {
        self.previewLayer.connection.videoOrientation = videoOrientation;
    }
}

- (void)setupCamera {
    self.captureSession = [[AVCaptureSession alloc] init];
    self.captureSession.sessionPreset = AVCaptureSessionPresetPhoto;
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    NSError *error = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (!input) {
        NSLog(@"Error setting up camera input: %@", error.localizedDescription);
        return;
    }
    [self.captureSession addInput:input];
    
    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    output.videoSettings = @{(NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)};
    output.alwaysDiscardsLateVideoFrames = YES;
    [output setSampleBufferDelegate:self queue:dispatch_get_main_queue()];
    [self.captureSession addOutput:output];
    self.previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.captureSession];
    self.previewLayer.videoGravity = AVLayerVideoGravityResizeAspect;
    [self.view.layer addSublayer:self.previewLayer];
    [self.captureSession startRunning];
    [self startNewRecording];
}

- (void)startNewRecording {
    NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
    [formatter setDateFormat:@"yyyy-MM-dd"];
    NSString *hourFolderName = [formatter stringFromDate:[NSDate date]];
    
    [formatter setDateFormat:@"yyyy-MM-dd_HH:mm:ss:SSS"];
    self.current_file_timestamp = [NSDate date];
    NSString *timestamp = [formatter stringFromDate:self.current_file_timestamp];
    NSString *segNumberString = [NSString stringWithFormat:@"_%05ld_", (long)self.seg_number];
    self.seg_number += 1;
    NSString *finalTimestamp = [NSString stringWithFormat:@"%@%@%@",
                                [[timestamp componentsSeparatedByString:@"_"] firstObject],
                                segNumberString,
                                [[timestamp componentsSeparatedByString:@"_"] lastObject]];
    
    // Create a folder for the hour within the documents directory
    NSURL *documentsDirectory = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    NSURL *hourFolderURL = [documentsDirectory URLByAppendingPathComponent:hourFolderName];
    
    // Ensure the directory exists
    NSError *error = nil;
    if (![[NSFileManager defaultManager] fileExistsAtPath:hourFolderURL.path]) {
        [[NSFileManager defaultManager] createDirectoryAtURL:hourFolderURL withIntermediateDirectories:YES attributes:nil error:&error];
        if (error) {
            NSLog(@"Error creating hour folder: %@", error.localizedDescription);
            return;
        }
    }
    
    NSURL *outputURL = [hourFolderURL URLByAppendingPathComponent:[NSString stringWithFormat:@"output_%@.mp4", finalTimestamp]];
    self.assetWriter = [AVAssetWriter assetWriterWithURL:outputURL fileType:AVFileTypeMPEG4 error:&error];
    if (error) {
        NSLog(@"Error creating asset writer: %@", error.localizedDescription);
        return;
    }
    
    NSDictionary *videoSettings = @{
        AVVideoCodecKey: AVVideoCodecTypeH264,
        AVVideoWidthKey: @640,
        AVVideoHeightKey: @480,
        AVVideoScalingModeKey: AVVideoScalingModeResizeAspectFill
    };
    self.videoWriterInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:videoSettings];
    self.videoWriterInput.expectsMediaDataInRealTime = YES;
    
    NSDictionary *sourcePixelBufferAttributes = @{
        (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (NSString *)kCVPixelBufferWidthKey: @640,
        (NSString *)kCVPixelBufferHeightKey: @480
    };
    self.adaptor = [AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:self.videoWriterInput sourcePixelBufferAttributes:sourcePixelBufferAttributes];
    
    if ([self.assetWriter canAddInput:self.videoWriterInput]) {
        [self.assetWriter addInput:self.videoWriterInput];
    } else {
        NSLog(@"Cannot add video writer input");
        return;
    }
    
    self.startTime = kCMTimeInvalid;
    self.currentTime = kCMTimeZero;
    self.isRecording = YES;
    
    [self.assetWriter startWriting];
    [self.assetWriter startSessionAtSourceTime:kCMTimeZero];
    
    NSLog(@"Started new recording");
}

- (void)drawSquareWithTopLeftX:(CGFloat)xOrigin topLeftY:(CGFloat)yOrigin bottomRightX:(CGFloat)bottomRightX bottomRightY:(CGFloat)bottomRightY classIndex:(int)classIndex aspectRatio:(float)aspectRatio {
    CGFloat leftEdgeX = (self.view.bounds.size.width - (self.view.bounds.size.height * aspectRatio)) / 2;
    CGFloat scaledXOrigin, scaledYOrigin, scaledWidth, scaledHeight;

    if (self.view.bounds.size.width < self.view.bounds.size.height) { //portrait
        scaledYOrigin = (self.view.bounds.size.height / 2) - (self.view.bounds.size.width * aspectRatio / 2);
        scaledYOrigin += (yOrigin / self.yolo.yolo_res) * (self.view.bounds.size.width * aspectRatio);
        scaledHeight = ((bottomRightY - yOrigin) / self.yolo.yolo_res) * (self.view.bounds.size.width * aspectRatio);
        scaledXOrigin = (xOrigin / (self.yolo.yolo_res / aspectRatio)) * self.view.bounds.size.width;
        scaledWidth = self.view.bounds.size.width * (bottomRightX - xOrigin) / (self.yolo.yolo_res / aspectRatio);
    } else {
        scaledXOrigin = leftEdgeX + (xOrigin * aspectRatio / self.yolo.yolo_res) * self.view.bounds.size.height;
        scaledYOrigin = (aspectRatio * yOrigin / self.yolo.yolo_res) * self.view.bounds.size.height;
        scaledWidth = (bottomRightX - xOrigin) * (aspectRatio * self.view.bounds.size.height / self.yolo.yolo_res);
        scaledHeight = (bottomRightY - yOrigin) * (aspectRatio * self.view.bounds.size.height / self.yolo.yolo_res);
    }

    UIColor *color = self.yolo.yolo_classes[classIndex][1];
    CAShapeLayer *squareLayer = [CAShapeLayer layer];
    squareLayer.name = @"rectangleLayer";
    squareLayer.strokeColor = color.CGColor;
    squareLayer.lineWidth = 2.0;
    squareLayer.fillColor = [UIColor clearColor].CGColor;
    squareLayer.path = [UIBezierPath bezierPathWithRect:CGRectMake(scaledXOrigin, scaledYOrigin, scaledWidth, scaledHeight)].CGPath;
    
    [self.view.layer addSublayer:squareLayer];
    
    NSString *className = self.yolo.yolo_classes[classIndex][0];
    NSDictionary *textAttributes = @{NSFontAttributeName: [UIFont systemFontOfSize:12],
                                     NSForegroundColorAttributeName: [UIColor whiteColor]};
    NSString *labelText = [className lowercaseString];
    CGSize textSize = [labelText sizeWithAttributes:textAttributes];
    
    CGFloat labelX = scaledXOrigin - 2;
    CGFloat labelY = scaledYOrigin - textSize.height - 2;
    
    UILabel *label = [[UILabel alloc] initWithFrame:CGRectMake(labelX, labelY, textSize.width + 4, textSize.height + 2)];
    label.backgroundColor = color;
    label.textColor = [UIColor whiteColor];
    label.font = [UIFont systemFontOfSize:12];
    label.text = labelText;
    
    [self.view addSubview:label];
}

- (void)resetSquares {
    NSMutableArray *layersToRemove = [NSMutableArray array];
    NSMutableArray *labelsToRemove = [NSMutableArray array];

    for (CALayer *layer in self.view.layer.sublayers) {
        if ([layer.name isEqualToString:@"rectangleLayer"]) {
            [layersToRemove addObject:layer];
        }
    }
    
    for (UIView *subview in self.view.subviews) {
        if ([subview isKindOfClass:[UILabel class]] && subview != self.fpsLabel) {
            [labelsToRemove addObject:subview];
        }
    }
    
    for (CALayer *layer in layersToRemove) {
        [layer removeFromSuperlayer];
    }
    
    for (UIView *label in labelsToRemove) {
        [label removeFromSuperview];
    }
}

- (void)setupFPSLabel {
    self.fpsLabel = [[UILabel alloc] initWithFrame:CGRectMake(10, 30, 150, 30)];
    self.fpsLabel.backgroundColor = [UIColor colorWithWhite:0 alpha:0.5];
    self.fpsLabel.textColor = [UIColor whiteColor];
    self.fpsLabel.font = [UIFont boldSystemFontOfSize:18];
    self.fpsLabel.text = @"FPS: 0";
    [self.view addSubview:self.fpsLabel];
}

- (void)viewWillLayoutSubviews {
    [super viewWillLayoutSubviews];
    self.previewLayer.frame = [self frameForCurrentOrientation];
}

- (CGRect)frameForCurrentOrientation {
    CGFloat width = self.view.bounds.size.width;
    CGFloat height = self.view.bounds.size.height;
    return CGRectMake(0, 0, width, height);
}

- (CGImageRef)cropAndResizeCGImage:(CGImageRef)image toSize:(CGSize)size {
    CGFloat originalWidth = CGImageGetWidth(image);
    CGFloat originalHeight = CGImageGetHeight(image);
    CGFloat sideLength = MIN(originalWidth, originalHeight);
    CGRect cropRect = CGRectMake((originalWidth - sideLength) / 2.0,
                                 (originalHeight - sideLength) / 2.0,
                                 sideLength,
                                 sideLength);

    CGImageRef croppedImageRef = CGImageCreateWithImageInRect(image, cropRect);
    UIGraphicsBeginImageContextWithOptions(size, NO, 0.0);
    [[UIImage imageWithCGImage:croppedImageRef] drawInRect:CGRectMake(0, 0, size.width, size.height)];
    UIImage *resizedImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    CGImageRelease(croppedImageRef);

    return resizedImage.CGImage;
}

- (void)finishRecording {
    if (self.isRecording && self.assetWriter.status == AVAssetWriterStatusWriting) {
        self.isRecording = NO;
        [self.videoWriterInput markAsFinished];
        [self.assetWriter finishWritingWithCompletionHandler:^{
            NSLog(@"Finished recording and saving file");

            NSString *fileName = [self.assetWriter.outputURL lastPathComponent];
            NSLog(@"File saved is %@", fileName);

            AVAsset *asset = [AVAsset assetWithURL:self.assetWriter.outputURL];
            CMTime time = asset.duration;

            // Use self.current_file_timestamp to get the date and hour without minutes and seconds
            NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
            [dateFormatter setDateFormat:@"yyyy-MM-dd"];
            NSString *dateFolderName = [dateFormatter stringFromDate:self.last_file_timestamp];

            // Get the documents directory
            NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
            NSString *segmentsDirectory = [documentsDirectory stringByAppendingPathComponent:dateFolderName];

            // Create the folder if it doesn't exist
            NSError *error = nil;
            if (![[NSFileManager defaultManager] fileExistsAtPath:segmentsDirectory]) {
                [[NSFileManager defaultManager] createDirectoryAtPath:segmentsDirectory withIntermediateDirectories:YES attributes:nil error:&error];
                if (error) {
                    NSLog(@"Error creating directory: %@", error);
                }
            }

            // Path to segments.txt in the date folder
            NSString *segmentsFilePath = [segmentsDirectory stringByAppendingPathComponent:@"segments.txt"];

            NSCalendar *calendar = [NSCalendar currentCalendar];
            NSDateComponents *components = [calendar components:NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay | NSCalendarUnitHour fromDate:self.last_file_timestamp];
            NSString *dateKey = [NSString stringWithFormat:@"%04ld-%02ld-%02ld", (long)components.year, (long)components.month, (long)components.day];

            if (!self.fileServer.segmentsDict[dateKey]) {
                self.fileServer.segmentsDict[dateKey] = [NSMutableArray array];
            }

            if (CMTIME_IS_VALID(self.last_file_duration)) {
                CMTime timeDifference = CMTimeMakeWithSeconds([self.current_file_timestamp timeIntervalSinceDate:self.last_file_timestamp], NSEC_PER_SEC);

                NSMutableDictionary *segmentEntry = [NSMutableDictionary dictionaryWithDictionary:@{ //does this fix last segment in hour/day?
                    @"url": [NSString stringWithFormat:@"%@/%@", [[self.last_file_url URLByDeletingLastPathComponent] lastPathComponent], self.last_file_url.lastPathComponent],
                    @"duration": @(CMTimeGetSeconds(self.last_file_duration)),
                    @"timeDifference": @(CMTimeGetSeconds(timeDifference)),
                    @"frames":self.last_segment_squares
                }];

                NSMutableArray *segmentsForDate = self.fileServer.segmentsDict[dateKey];
                if (segmentsForDate.count == 0) {
                    NSCalendar *calendar = [NSCalendar currentCalendar];
                    NSDateComponents *components = [calendar components:NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay fromDate:self.last_file_timestamp];
                    NSDate *midnight = [calendar dateFromComponents:components];
                    NSTimeInterval timeStamp = [self.last_file_timestamp timeIntervalSinceDate:midnight];
                    segmentEntry[@"timeStamp"] = @(timeStamp);
                }
                
                [self saveSegmentEntry:segmentEntry toFile:segmentsFilePath]; // Not used yet
                NSLog(@"adding %@ to date %@ in dict",segmentEntry,dateKey);
                [self.fileServer.segmentsDict[dateKey] addObject:segmentEntry];
            }
            self.last_file_duration = time;
            self.last_file_url = [self.assetWriter.outputURL copy];
            self.last_file_timestamp = self.current_file_timestamp;
            self.last_segment_squares = [self.current_segment_squares copy];
            [self.current_segment_squares removeAllObjects];
            NSLog(@"Duration is %f seconds", CMTimeGetSeconds(time));
            [self startNewRecording];
        }];
    } else {
        NSLog(@"Cannot finish writing. Asset writer status: %ld", (long)self.assetWriter.status);
    }
}

- (void)saveSegmentEntry:(NSDictionary *)segmentEntry toFile:(NSString *)filePath {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    if (![fileManager fileExistsAtPath:filePath]) {
        NSError *error;
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:@[segmentEntry] options:NSJSONWritingPrettyPrinted error:&error];
        if (!error) {
            [jsonData writeToFile:filePath atomically:YES];
        } else {
            NSLog(@"Error writing JSON to file: %@", error.localizedDescription);
        }
    } else {
        NSFileHandle *fileHandle = [NSFileHandle fileHandleForWritingAtPath:filePath];
        if (fileHandle) {
            [fileHandle seekToEndOfFile];
            [fileHandle truncateFileAtOffset:[fileHandle offsetInFile] - 1];
            NSString *entryString = [NSString stringWithFormat:@", %@]", [self jsonStringFromDictionary:segmentEntry]];
            [fileHandle writeData:[entryString dataUsingEncoding:NSUTF8StringEncoding]];
            [fileHandle closeFile];
        } else {
            NSLog(@"Error opening file for writing.");
        }
    }
}

- (NSString *)jsonStringFromDictionary:(NSDictionary *)dictionary {
    NSError *error;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:dictionary options:0 error:&error];
    if (!error) {
        return [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
    } else {
        NSLog(@"Error converting dictionary to JSON string: %@", error.localizedDescription);
        return nil;
    }
}


- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    @try {
        if (self.isRecording && self.assetWriter.status == AVAssetWriterStatusWriting) {
            CMTime timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
            if (CMTIME_IS_INVALID(self.startTime)) {
                self.startTime = timestamp;
                self.currentTime = kCMTimeZero;
            } else {
                self.currentTime = CMTimeSubtract(timestamp, self.startTime);
            }

            if (self.videoWriterInput.readyForMoreMediaData) {
                CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
                pixelBuffer = [self addTimeStampToPixelBuffer:pixelBuffer];
                
                BOOL success = NO;
                int retryCount = 3;

                while (!success && retryCount > 0) {
                    success = [self.adaptor appendPixelBuffer:pixelBuffer withPresentationTime:self.currentTime];
                    if (!success) {
                        retryCount--;
                        [NSThread sleepForTimeInterval:0.01];
                    }
                }
                if (!success) [self finishRecording];
            }
            NSTimeInterval elapsedTime = CMTimeGetSeconds(self.currentTime);
            if (elapsedTime >= 10.0) { // todo, takes a little time to change
                [self finishRecording];
            }
        }

        AVCaptureVideoOrientation videoOrientation = self.previewLayer.connection.videoOrientation;
        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        CIImage *ciImage = [CIImage imageWithCVPixelBuffer:imageBuffer];

        if (self.isProcessing) {
            return;
        }

        self.isProcessing = YES;

        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
            size_t width = CVPixelBufferGetWidth(imageBuffer);
            size_t height = CVPixelBufferGetHeight(imageBuffer);
            CGFloat aspect_ratio = (CGFloat)width / (CGFloat)height;
            
            NSMutableDictionary *frame = [[NSMutableDictionary alloc] init];//todo, init elsewhere
            NSMutableDictionary *frameSquare = [[NSMutableDictionary alloc] init];;
            NSMutableArray *frameSquares = [[NSMutableArray alloc] init];;
            

            NSCalendar *calendar = [NSCalendar currentCalendar];
            NSDate *now = [NSDate date];
            NSDateComponents *components = [calendar components:NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay fromDate:now];
            NSDate *midnight = [calendar dateFromComponents:components];
            NSTimeInterval timeStamp = [now timeIntervalSinceDate:midnight];
            
            frame[@"timeStamp"] = @(timeStamp);
            frame[@"res"] = @(self.yolo.yolo_res);
            frame[@"aspect_ratio"] = @(aspect_ratio);
            
            CGFloat targetWidth = self.yolo.yolo_res;
            CGSize targetSize = CGSizeMake(targetWidth, targetWidth / aspect_ratio);

            CGFloat scaleX = targetSize.width / width;
            CGFloat scaleY = targetSize.height / height;
            CIImage *resizedImage = [ciImage imageByApplyingTransform:CGAffineTransformMakeScale(scaleX, scaleY)];
            
            CGRect cropRect = CGRectMake(0, 0, targetSize.width, targetSize.height);
            CIImage *croppedImage = [resizedImage imageByCroppingToRect:cropRect];
            
            CGImageRef cgImage = [self.ciContext createCGImage:croppedImage fromRect:cropRect];
            
            NSArray *output = [self.yolo yolo_infer:cgImage withOrientation:videoOrientation];
            CGImageRelease(cgImage);
            
            __weak typeof(self) weak_self = self;
            dispatch_async(dispatch_get_main_queue(), ^{
                [weak_self resetSquares];
                for (int i = 0; i < output.count; i++) {
                    [weak_self drawSquareWithTopLeftX:[output[i][0] floatValue]
                                              topLeftY:[output[i][1] floatValue]
                                          bottomRightX:[output[i][2] floatValue]
                                          bottomRightY:[output[i][3] floatValue]
                                            classIndex:[output[i][4] intValue]
                                           aspectRatio:aspect_ratio];
                    frameSquare[@"originX"] = output[i][0];
                    frameSquare[@"originY"] = output[i][1];
                    frameSquare[@"bottomRightX"] = output[i][2];
                    frameSquare[@"bottomRightY"] = output[i][3];
                    frameSquare[@"classIndex"] = output[i][4];
                    [frameSquares addObject:[frameSquare copy]];
                }
                frame[@"squares"] = frameSquares;
                [self.current_segment_squares addObject:frame];
                [weak_self updateFPS];
                weak_self.isProcessing = NO;
            });
        });
    } @catch (NSException *exception) {
            NSLog(@"Exception occurred: %@, %@", exception, [exception callStackSymbols]);
    }
}

- (CVPixelBufferRef)addColoredRectangleToPixelBuffer:(CVPixelBufferRef)pixelBuffer withColor:(UIColor *)color originX:(CGFloat)originX originY:(CGFloat)originY width:(CGFloat)width height:(CGFloat)height opacity:(CGFloat)opacity {
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);

    CGContextRef context = [self createContextForPixelBuffer:pixelBuffer];

    size_t pixelBufferHeight = CVPixelBufferGetHeight(pixelBuffer);
    // Adjust originY to be relative to the top-left corner
    CGRect rectangleRect = CGRectMake(originX, pixelBufferHeight - originY - height, width, height);
    
    // Set the fill color with the adjusted opacity
    UIColor *colorWithOpacity = [color colorWithAlphaComponent:opacity];
    CGContextSetFillColorWithColor(context, colorWithOpacity.CGColor);
    CGContextFillRect(context, rectangleRect);

    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    CGContextRelease(context);

    return pixelBuffer;
}


- (CVPixelBufferRef)addTimeStampToPixelBuffer:(CVPixelBufferRef)pixelBuffer{
    NSInteger pixelSize = 4;
    NSInteger spaceSize = 2;
    NSInteger digitOriginX = spaceSize;
    NSInteger digitOriginY = spaceSize;
    NSInteger height = spaceSize*2 + pixelSize*5;

    NSDate *currentDate = [NSDate date];
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
    NSString *timestamp = [dateFormatter stringFromDate:currentDate];
    pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer withColor:[UIColor blackColor] originX:0 originY:0 width:(pixelSize*3+spaceSize)*timestamp.length height:height opacity:0.4];
        
    for (NSUInteger k = 0; k < [timestamp length]; k++) {
        unichar character = [timestamp characterAtIndex:k];
        if(character == ' '){
            digitOriginX += pixelSize*3;
            continue;
        }
        NSString *key = [NSString stringWithFormat:@"%C", character];
        for (int i = 0; i < [self.digits[key] count]; i++) {
            pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer
                                                        withColor:[UIColor whiteColor]
                                                         originX:digitOriginX + [self.digits[key][i][0] doubleValue] * pixelSize
                                                         originY:digitOriginY + [self.digits[key][i][1] doubleValue] * pixelSize
                                                           width:[self.digits[key][i][2] doubleValue] * pixelSize
                                                           height:[self.digits[key][i][3] doubleValue] * pixelSize opacity:1];
        }
        digitOriginX += pixelSize*3 + spaceSize;
    }
    return pixelBuffer;
}


// Helper method to create a CGContext for a CVPixelBufferRef
- (CGContextRef)createContextForPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);

    void *baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8, bytesPerRow, colorSpace, kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);
    CGColorSpaceRelease(colorSpace);

    return context;
}


- (void)updateFPS {
    CFTimeInterval currentTime = CACurrentMediaTime();
    if (self.lastFrameTime > 0) {
        CFTimeInterval deltaTime = currentTime - self.lastFrameTime;
        self.frameCount++;
        if (deltaTime >= 1.0) {
            CGFloat fps = self.frameCount / deltaTime;
            self.fpsLabel.text = [NSString stringWithFormat:@"FPS: %.1f", fps];
            self.frameCount = 0;
            self.lastFrameTime = currentTime;
        }
    } else {
        self.lastFrameTime = currentTime;
    }
}
@end



