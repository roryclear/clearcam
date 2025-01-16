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

@end

@implementation ViewController

NSMutableDictionary *classColorMap;

- (void)viewDidLoad {
    [super viewDidLoad];
    self.ciContext = [CIContext context];
    self.yolo = [[Yolo alloc] init];
    self.seg_number = 0;
    [self setupCamera];
    [self setupFPSLabel];
    self.fileServer = [[FileServer alloc] init];
    [self.fileServer start];
    
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(handleDeviceOrientationChange)
                                                 name:UIDeviceOrientationDidChangeNotification
                                               object:nil];
    
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
    [self handleDeviceOrientationChange];
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
    [formatter setDateFormat:@"yyyy-MM-dd_HH:mm:ss:SSS"];
    NSString *timestamp = [formatter stringFromDate:[NSDate date]];
    NSString *segNumberString = [NSString stringWithFormat:@"_%05ld_", (long)self.seg_number];
    self.seg_number += 1;
    NSString *finalTimestamp = [NSString stringWithFormat:@"%@%@%@",
                                [[timestamp componentsSeparatedByString:@"_"] firstObject],
                                segNumberString,
                                [[timestamp componentsSeparatedByString:@"_"] lastObject]];
    NSURL *documentsDirectory = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    NSURL *outputURL = [documentsDirectory URLByAppendingPathComponent:[NSString stringWithFormat:@"output_%@.mp4", finalTimestamp]];
    NSError *error = nil;
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
            [self startNewRecording];
        }];
    } else {
        NSLog(@"Cannot finish writing. Asset writer status: %ld", (long)self.assetWriter.status);
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
                }
                [weak_self updateFPS];
                weak_self.isProcessing = NO;
            });
        });
    } @catch (NSException *exception) {
            NSLog(@"Exception occurred: %@, %@", exception, [exception callStackSymbols]);
    }
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




