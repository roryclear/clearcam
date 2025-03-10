#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <Metal/Metal.h>
#import "Yolo.h"
#import "FileServer.h"
#import "SettingsManager.h"
#import "SceneState.h"
#import "SettingsViewController.h"
//#import "pgp.h"

@interface ViewController ()

@property (nonatomic, strong) AVCaptureSession *captureSession;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *previewLayer;
@property (nonatomic, strong) UILabel *fpsLabel;
@property (nonatomic, strong) UIButton *recordButton;
@property (nonatomic, strong) UIButton *settingsButton;
@property (nonatomic, assign) CFTimeInterval lastFrameTime;
@property (nonatomic, assign) NSUInteger frameCount;
@property (nonatomic, strong) Yolo *yolo;
@property (nonatomic, strong) CIContext *ciContext;

@property (atomic, assign) BOOL isProcessing;
@property (nonatomic, assign) BOOL recordPressed;
@property (nonatomic, assign) BOOL isRecording;
@property (nonatomic, assign) CMTime startTime;
@property (nonatomic, assign) CMTime currentTime;
@property (nonatomic, strong) AVAssetWriter *assetWriter;
@property (nonatomic, strong) AVAssetWriterInput *videoWriterInput;
@property (nonatomic, strong) AVAssetWriterInputPixelBufferAdaptor *adaptor;
@property (nonatomic, strong) FileServer *fileServer;
@property (nonatomic, assign) int seg_number;
@property (nonatomic, assign) NSDate *current_file_timestamp;
@property (nonatomic, strong) NSMutableDictionary *digits;
@property (nonatomic, strong) NSMutableArray *current_segment_squares;
@property (nonatomic, strong) NSLock *segmentLock;
@property (nonatomic, strong) NSManagedObjectContext *backgroundContext;
@property (nonatomic, strong) NSString *dayFolderName;
@property (nonatomic, strong) dispatch_queue_t segmentQueue;
@property (nonatomic, strong) SceneState *scene;

#define MIN_FREE_SPACE_MB 500  //threshold to start deleting

@end

@implementation ViewController

NSMutableDictionary *classColorMap;

- (void)viewDidLoad {
    [super viewDidLoad];
    self.recordPressed = NO;
    //PGP *pgp = [[PGP alloc] init];
    self.scene = [[SceneState alloc] init];
    self.segmentQueue = dispatch_queue_create("com.example.segmentQueue", DISPATCH_QUEUE_SERIAL);
    self.current_segment_squares = [[NSMutableArray alloc] init];
    self.digits = [NSMutableDictionary dictionary];
    self.digits[@"0"] = @[@[ @0, @0, @3, @1], @[ @0, @1, @1 , @3], @[ @2, @1, @1 , @3], @[ @0, @4, @3 , @1]];
    self.digits[@"1"] = @[@[ @2, @0, @1, @5 ]];
    self.digits[@"2"] = @[@[ @0, @0, @3, @1 ], @[ @2, @1, @1, @1 ], @[ @0, @2, @3, @1 ], @[ @0, @3, @1, @1 ], @[ @0, @4, @3, @1 ]];
    self.digits[@"3"] = @[@[ @0, @0, @3, @1 ], @[ @2, @1, @1, @3 ], @[ @0, @2, @3, @1 ], @[ @0, @4, @3, @1 ]];
    self.digits[@"4"] = @[@[ @2, @0, @1, @5 ], @[ @0, @0, @1, @2 ], @[ @0, @2, @3, @1 ]];
    self.digits[@"5"] = @[@[ @0, @0, @3, @1 ], @[ @0, @1, @1, @1 ], @[ @0, @2, @3, @1 ], @[ @2, @3, @1, @1 ], @[ @0, @4, @3, @1 ]];
    self.digits[@"6"] = @[@[ @0, @0, @3, @1 ],@[ @0, @0, @1, @5 ], @[ @1, @2, @2, @1 ], @[ @1, @4, @2, @1 ], @[ @2, @3, @1, @1 ]];
    self.digits[@"7"] = @[@[ @0, @0, @3, @1 ], @[ @2, @1, @1, @4 ]];
    self.digits[@"8"] = @[@[ @0, @0, @1, @5 ], @[ @2, @0, @1, @5 ], @[ @1, @0, @1, @1 ], @[ @1, @2, @1, @1 ], @[ @1, @4, @1, @1 ]];
    self.digits[@"9"] = @[@[ @2, @0, @1, @5 ], @[ @1, @0, @1, @1 ], @[ @0, @0, @1, @2 ], @[ @0, @2, @3, @1 ],@[ @0, @4, @3, @1 ]];
    self.digits[@"-"] = @[@[ @0, @2, @3, @1 ]];
    self.digits[@":"] = @[@[ @1, @1, @1, @1 ], @[ @1, @3, @1, @1 ]];
    self.segmentLock = [[NSLock alloc] init]; //dont allow current_segment_squares to be accessed twice at once!
        
    self.ciContext = [CIContext context];
    self.yolo = [[Yolo alloc] init];
    self.seg_number = 0;
    self.fileServer = [[FileServer alloc] init];
    [self.fileServer start];
    self.backgroundContext = [[NSManagedObjectContext alloc] initWithConcurrencyType:NSPrivateQueueConcurrencyType];
    self.backgroundContext.mergePolicy = NSMergeByPropertyStoreTrumpMergePolicy; //prevents crash??
    self.backgroundContext.parentContext = self.fileServer.context;
    
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(handleDeviceOrientationChange)
                                                 name:UIDeviceOrientationDidChangeNotification
                                               object:nil];
    
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
    [self handleDeviceOrientationChange];
    SettingsManager *settings = [SettingsManager sharedManager];
    
    // Add KVO observers for resolution properties
    [settings addObserver:self forKeyPath:@"width" options:NSKeyValueObservingOptionNew context:nil];
    [settings addObserver:self forKeyPath:@"height" options:NSKeyValueObservingOptionNew context:nil];
    [settings addObserver:self forKeyPath:@"text_size" options:NSKeyValueObservingOptionNew context:nil];
    [settings addObserver:self forKeyPath:@"preset" options:NSKeyValueObservingOptionNew context:nil];
    
    [self setupCameraWithWidth:settings.width height:settings.height];
    [self setupUI];
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary<NSKeyValueChangeKey,id> *)change context:(void *)context {
    if ([keyPath isEqualToString:@"preset"]) {
        [self finishRecording];
        [self resetUI];
        SettingsManager *settings = [SettingsManager sharedManager];
        self.captureSession = [[AVCaptureSession alloc] init];
        NSString *presetString = [NSString stringWithFormat:@"AVCaptureSessionPreset%@x%@", settings.width, settings.height];

        if ([self.captureSession canSetSessionPreset:presetString]) {
            self.captureSession.sessionPreset = presetString;
        } else {
            NSLog(@"Unsupported preset: %@", presetString);
            return;
        }

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
        
        if (self.previewLayer) {
            [self.previewLayer removeFromSuperlayer]; // Remove from the view's layer hierarchy
            self.previewLayer = nil; // Optionally, set it to nil if you're done with it
        }
        self.previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.captureSession];
        self.previewLayer.videoGravity = AVLayerVideoGravityResizeAspect;
        [self.view.layer addSublayer:self.previewLayer];
        
        [self.captureSession startRunning];
        if(self.isRecording) [self startNewRecording];
        [self setupUI];
    }
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

- (void)setupCameraWithWidth:(NSString *)width height:(NSString *)height {
    self.captureSession = [[AVCaptureSession alloc] init];
    NSString *presetString = [NSString stringWithFormat:@"AVCaptureSessionPreset%@x%@", width, height];

    if ([self.captureSession canSetSessionPreset:presetString]) {
        self.captureSession.sessionPreset = presetString;
    } else {
        NSLog(@"Unsupported preset: %@", presetString);
        return;
    }

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

- (NSString*)getDateString {
    NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
    [formatter setDateFormat:@"yyyy-MM-dd"];
    return [formatter stringFromDate:[NSDate date]];
}

- (NSString*)fake_getDateString { //todo remove
    NSDate *now = [NSDate date];
    NSCalendar *calendar = [NSCalendar currentCalendar];
    NSDateComponents *nowComponents = [calendar components:(NSCalendarUnitHour | NSCalendarUnitMinute) fromDate:now];
    
    if (nowComponents.hour < 12 || (nowComponents.hour == 12 && nowComponents.minute < 20)) {
        return @"2025-02-14";
    } else {
        return @"2025-02-15";
    }
}

- (void)startNewRecording {
    self.dayFolderName = [self getDateString];
    NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
    [formatter setDateFormat:@"yyyy-MM-dd_HH:mm:ss:SSS"];
    self.current_file_timestamp = [NSDate date];
    NSString *timestamp = [formatter stringFromDate:self.current_file_timestamp];
    NSString *segNumberString = [NSString stringWithFormat:@"_%05ld_", (long)self.seg_number];
    self.seg_number += 1;
    NSString *finalTimestamp = [NSString stringWithFormat:@"%@%@%@",
                                [[timestamp componentsSeparatedByString:@"_"] firstObject],
                                segNumberString,
                                [[timestamp componentsSeparatedByString:@"_"] lastObject]];
    
    // Get resolution settings from SettingsManager
    SettingsManager *settings = [SettingsManager sharedManager];
    int videoWidth = [settings.width intValue];
    int videoHeight = [settings.height intValue];
    
    // Create a folder for the day within the documents directory
    NSURL *documentsDirectory = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    NSURL *dayFolderURL = [documentsDirectory URLByAppendingPathComponent:self.dayFolderName];
    
    // Ensure the directory exists
    NSError *error = nil;
    if (![[NSFileManager defaultManager] fileExistsAtPath:dayFolderURL.path]) {
        [[NSFileManager defaultManager] createDirectoryAtURL:dayFolderURL withIntermediateDirectories:YES attributes:nil error:&error];
        if (error) {
            NSLog(@"Error creating day folder: %@", error.localizedDescription);
            return;
        }
    }
    
    NSURL *outputURL = [dayFolderURL URLByAppendingPathComponent:[NSString stringWithFormat:@"output_%@.mp4", finalTimestamp]];
    self.assetWriter = [AVAssetWriter assetWriterWithURL:outputURL fileType:AVFileTypeMPEG4 error:&error];
    if (error) {
        NSLog(@"Error creating asset writer: %@", error.localizedDescription);
        return;
    }
    
    // Check if the device supports HEVC encoding
    NSDictionary *videoSettings;
    videoSettings = @{
        AVVideoCodecKey: AVVideoCodecTypeH264, // Fallback to H.264
        AVVideoWidthKey: @(videoWidth),
        AVVideoHeightKey: @(videoHeight),
        AVVideoScalingModeKey: AVVideoScalingModeResizeAspectFill
    };
    
    NSArray<NSString *> *exportPresets = [AVAssetExportSession allExportPresets];
    if ([exportPresets containsObject:AVAssetExportPresetHEVCHighestQuality]) {
        // HEVC is supported
        videoSettings = @{
            AVVideoCodecKey: AVVideoCodecTypeHEVC, // Use HEVC (H.265)
            AVVideoWidthKey: @(videoWidth),
            AVVideoHeightKey: @(videoHeight),
            AVVideoScalingModeKey: AVVideoScalingModeResizeAspectFill
        };
    } else {
        NSLog(@"Device doesn't support H265");
    }
        
    self.videoWriterInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:videoSettings];
    self.videoWriterInput.expectsMediaDataInRealTime = YES;
    
    if (self.previewLayer.connection.videoOrientation == AVCaptureVideoOrientationPortrait) {
        self.videoWriterInput.transform = CGAffineTransformMakeRotation(M_PI_2); // Rotate for portrait
    } else if (self.previewLayer.connection.videoOrientation == AVCaptureVideoOrientationLandscapeLeft) {
        self.videoWriterInput.transform = CGAffineTransformMakeRotation(M_PI); // Rotate for landscape left
    }
    
    NSDictionary *sourcePixelBufferAttributes = @{
        (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (NSString *)kCVPixelBufferWidthKey: @(videoWidth),
        (NSString *)kCVPixelBufferHeightKey: @(videoHeight)
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
    if(![self ensureFreeDiskSpace]) {
        [self stopRecording];
        NSLog(@"no space, recording stopped. Delete some stuff");
    }
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

- (void)resetUI {
    // Remove all subviews
    for (UIView *subview in self.view.subviews) {
        [subview removeFromSuperview];
    }

    // Remove all sublayers (using a copy of the sublayers array)
    NSArray<CALayer *> *sublayers = [self.view.layer.sublayers copy];
    for (CALayer *sublayer in sublayers) {
        [sublayer removeFromSuperlayer];
    }

    // Reset UI-related properties
    self.previewLayer = nil;
}

- (void)setupUI {
    self.fpsLabel = [[UILabel alloc] initWithFrame:CGRectMake(10, 30, 150, 30)];
    self.fpsLabel.backgroundColor = [UIColor colorWithWhite:0 alpha:0.5];
    self.fpsLabel.textColor = [UIColor whiteColor];
    self.fpsLabel.font = [UIFont boldSystemFontOfSize:18];
    self.fpsLabel.text = @"FPS: 0";
    [self.view addSubview:self.fpsLabel];

    // Create the record button
    self.recordButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.recordButton setTitle:@"Record" forState:UIControlStateNormal];
    self.recordButton.backgroundColor = [UIColor redColor];
    self.recordButton.tintColor = [UIColor whiteColor];
    self.recordButton.titleLabel.font = [UIFont boldSystemFontOfSize:16];

    CGFloat buttonSize = 60;
    self.recordButton.frame = CGRectMake(0, 0, buttonSize, buttonSize);
    self.recordButton.layer.cornerRadius = buttonSize / 2;
    self.recordButton.clipsToBounds = YES;

    [self.recordButton addTarget:self action:@selector(toggleRecording) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.recordButton];

    // Create the settings button
    self.settingsButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.settingsButton setTitle:@"⚙️" forState:UIControlStateNormal]; // Gear icon
    self.settingsButton.backgroundColor = [UIColor grayColor];
    self.settingsButton.tintColor = [UIColor whiteColor];
    self.settingsButton.titleLabel.font = [UIFont boldSystemFontOfSize:20];

    self.settingsButton.frame = CGRectMake(0, 0, buttonSize, buttonSize);
    self.settingsButton.layer.cornerRadius = buttonSize / 2;
    self.settingsButton.clipsToBounds = YES;

    [self.settingsButton addTarget:self action:@selector(openSettings) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.settingsButton];

    [self updateButtonFrames]; // Set initial positions
}

// Position the buttons correctly based on orientation
- (void)updateButtonFrames {
    CGFloat screenWidth = self.view.bounds.size.width;
    CGFloat screenHeight = self.view.bounds.size.height;
    CGFloat buttonSize = 60;
    CGFloat margin = 20;
    CGFloat spacing = 10; // Space between buttons

    self.recordButton.layer.cornerRadius = buttonSize / 2;
    self.settingsButton.layer.cornerRadius = buttonSize / 2;

    if (screenWidth > screenHeight) { // Landscape: Right side
        self.recordButton.frame = CGRectMake(screenWidth - buttonSize - margin, screenHeight / 2 - buttonSize / 2, buttonSize, buttonSize);
        self.settingsButton.frame = CGRectMake(self.recordButton.frame.origin.x, self.recordButton.frame.origin.y - buttonSize - spacing, buttonSize, buttonSize);
    } else { // Portrait: Bottom center, settings to the right
        CGFloat recordX = (screenWidth - buttonSize) / 2;
        self.recordButton.frame = CGRectMake(recordX, screenHeight - buttonSize - margin, buttonSize, buttonSize);
        self.settingsButton.frame = CGRectMake(recordX + buttonSize + spacing, self.recordButton.frame.origin.y, buttonSize, buttonSize);
    }
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    [self updateButtonFrames];
}

- (void)openSettings {
    NSLog(@"Settings button tapped");
    SettingsViewController *settingsVC = [[SettingsViewController alloc] init];
    [self.navigationController pushViewController:settingsVC animated:YES];
}

// Toggle recording state
- (void)toggleRecording {
    if ([[self.recordButton titleForState:UIControlStateNormal] isEqualToString:@"Record"]) {
        self.recordPressed = YES;
        [self.recordButton setTitle:@"Stop" forState:UIControlStateNormal];
        self.recordButton.backgroundColor = [UIColor darkGrayColor]; // Change color to indicate recording
    } else {
        self.recordPressed = NO;
        [self.recordButton setTitle:@"Record" forState:UIControlStateNormal];
        self.recordButton.backgroundColor = [UIColor redColor]; // Reset to red when stopped
    }
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

- (void)stopRecording {
    if (!(self.isRecording && self.assetWriter.status == AVAssetWriterStatusWriting)) {
        NSLog(@"Cannot finish writing. Asset writer status: %ld", (long)self.assetWriter.status);
        return;
    }
    
    self.isRecording = NO;
    [self.videoWriterInput markAsFinished];
    
    [self.assetWriter finishWritingWithCompletionHandler:^{
        NSLog(@"Recording stopped.");
    }];
}


- (void)finishRecording {
    if (!(self.isRecording && self.assetWriter.status == AVAssetWriterStatusWriting)) {
        NSLog(@"Cannot finish writing. Asset writer status: %ld", (long)self.assetWriter.status);
        return;
    }

    self.isRecording = NO;
    [self.videoWriterInput markAsFinished];

    [self.assetWriter finishWritingWithCompletionHandler:^{
        if (!self.assetWriter.outputURL) return;

        AVAsset *asset = [AVAsset assetWithURL:self.assetWriter.outputURL];
        CMTime time = asset.duration;

        NSString *segmentsDirectory = [[NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject] stringByAppendingPathComponent:self.dayFolderName];

        [[NSFileManager defaultManager] createDirectoryAtPath:segmentsDirectory withIntermediateDirectories:YES attributes:nil error:nil];

        NSString *segmentURL = [NSString stringWithFormat:@"%@/%@", [[self.assetWriter.outputURL URLByDeletingLastPathComponent] lastPathComponent], self.assetWriter.outputURL.lastPathComponent];

        NSCalendar *calendar = [NSCalendar currentCalendar];
        NSDateComponents *components = [calendar components:NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay fromDate:self.current_file_timestamp];
        NSTimeInterval timeStamp = [self.current_file_timestamp timeIntervalSinceDate:[calendar dateFromComponents:components]];

        // Create a local copy of current_segment_squares to avoid mutation issues
        __block NSArray *segmentSquaresCopy;
        dispatch_sync(self.segmentQueue, ^{
            segmentSquaresCopy = [self.current_segment_squares copy];
        });

        [self.backgroundContext performBlockAndWait:^{
            NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"DayEntity"];
            fetchRequest.predicate = [NSPredicate predicateWithFormat:@"date == %@", self.dayFolderName];

            NSError *error = nil;
            NSArray *fetchedDays = [self.backgroundContext executeFetchRequest:fetchRequest error:&error];
            NSManagedObject *dayEntity = fetchedDays.firstObject ?: [NSEntityDescription insertNewObjectForEntityForName:@"DayEntity" inManagedObjectContext:self.backgroundContext];

            if (!fetchedDays.count) {
                [dayEntity setValue:self.dayFolderName forKey:@"date"];
            }

            NSManagedObject *newSegment = [NSEntityDescription insertNewObjectForEntityForName:@"SegmentEntity" inManagedObjectContext:self.backgroundContext];
            [newSegment setValuesForKeysWithDictionary:@{
                @"url": segmentURL,
                @"timeStamp": @(timeStamp),
                @"duration": @(CMTimeGetSeconds(time))
            }];

            NSMutableArray<NSManagedObject *> *segmentFrames = [NSMutableArray arrayWithCapacity:segmentSquaresCopy.count];

            for (NSDictionary *frameData in segmentSquaresCopy) {
                NSManagedObject *newFrame = [NSEntityDescription insertNewObjectForEntityForName:@"FrameEntity" inManagedObjectContext:self.backgroundContext];
                [newFrame setValuesForKeysWithDictionary:@{
                    @"frame_timeStamp": frameData[@"frame_timeStamp"],
                    @"aspect_ratio": frameData[@"aspect_ratio"],
                    @"res": frameData[@"res"]
                }];

                NSMutableArray<NSManagedObject *> *frameSquares = [NSMutableArray arrayWithCapacity:[frameData[@"squares"] count]];

                for (NSDictionary *squareData in frameData[@"squares"]) {
                    NSManagedObject *newSquare = [NSEntityDescription insertNewObjectForEntityForName:@"SquareEntity" inManagedObjectContext:self.backgroundContext];
                    [newSquare setValuesForKeysWithDictionary:squareData];
                    [frameSquares addObject:newSquare];
                }

                [newFrame setValue:[NSOrderedSet orderedSetWithArray:frameSquares] forKey:@"squares"];
                [segmentFrames addObject:newFrame];
            }

            [newSegment setValue:[NSOrderedSet orderedSetWithArray:segmentFrames] forKey:@"frames"];
            [[dayEntity mutableOrderedSetValueForKey:@"segments"] addObject:newSegment];

            if ([self.backgroundContext save:&error]) {
                NSLog(@"Segment saved successfully under DayEntity with date %@", self.dayFolderName);
                [self.fileServer.context performBlockAndWait:^{
                    [self.fileServer.context save:nil];
                }];
            } else {
                NSLog(@"Failed to save segment: %@", error.localizedDescription);
            }

            // Clean up memory after saving, still within the same thread
            dispatch_async(self.segmentQueue, ^{
                [self.current_segment_squares removeAllObjects];
            });
        }];

        // Start new recording after the block completes
        dispatch_async(dispatch_get_main_queue(), ^{
            [self startNewRecording];
        });
    }];
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

- (uint64_t)getFreeDiskSpace {
    NSError *error = nil;
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfFileSystemForPath:NSHomeDirectory() error:&error];

    if (error) {
        NSLog(@"Error retrieving file system info: %@", error.localizedDescription);
        return 0;
    }

    uint64_t freeSpace = [[attributes objectForKey:NSFileSystemFreeSize] unsignedLongLongValue];
    return freeSpace;
}

- (BOOL)ensureFreeDiskSpace {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];

    NSInteger lastDeletedDayIndex = [defaults integerForKey:@"LastDeletedDayIndex"];
    NSInteger lastDeletedSegmentIndex = [defaults integerForKey:@"LastDeletedSegmentIndex"];

    @try {
        if ((double)[[[[NSFileManager defaultManager] attributesOfFileSystemForPath:NSHomeDirectory() error:nil] objectForKey:NSFileSystemFreeSize] unsignedLongLongValue] / (1024.0 * 1024.0) < MIN_FREE_SPACE_MB) {
            NSLog(@"deleting stuff");
            NSLog(@"NOT ENOUGH SPACE!");
            
            // Fetch all DayEntities, sorted by date
            NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"DayEntity"];
            fetchRequest.sortDescriptors = @[[NSSortDescriptor sortDescriptorWithKey:@"date" ascending:YES]];
            
            NSError *fetchError = nil;
            NSArray *dayEntities = [self.fileServer.context executeFetchRequest:fetchRequest error:&fetchError];
            
            if (fetchError) {
                NSLog(@"Failed to fetch DayEntity objects: %@", fetchError.localizedDescription);
                return YES;
            }
            
            if (dayEntities.count == 0 || lastDeletedDayIndex >= dayEntities.count) {
                NSLog(@"No more DayEntities available. Resetting deletion indexes.");
                [defaults removeObjectForKey:@"LastDeletedDayIndex"];
                [defaults removeObjectForKey:@"LastDeletedSegmentIndex"];
                return YES;
            }
            
            // Fetch event timestamps from Core Data
            NSArray *eventDataArray = [self.fileServer fetchEventDataFromCoreData:self.fileServer.context];

            if (!eventDataArray || eventDataArray.count == 0) {
                NSLog(@"No event timestamps found.");
            }

            NSMutableArray<NSNumber *> *eventTimestamps = [NSMutableArray array];
            NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
            [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];

            NSCalendar *calendar = [NSCalendar currentCalendar];

            for (NSDictionary *eventData in eventDataArray) {
                NSString *timestampString = eventData[@"timeStamp"];
                NSDate *eventDate = [dateFormatter dateFromString:timestampString];

                if (eventDate) {
                    NSDateComponents *components = [calendar components:(NSCalendarUnitHour | NSCalendarUnitMinute | NSCalendarUnitSecond) fromDate:eventDate];

                    // Convert to seconds since midnight
                    double eventSecondsSinceMidnight = (components.hour * 3600) + (components.minute * 60) + components.second;
                    [eventTimestamps addObject:@(eventSecondsSinceMidnight)];
                }
            }



            BOOL deletedFile = NO;
            
            for (NSInteger dayIndex = lastDeletedDayIndex; dayIndex < dayEntities.count; dayIndex++) {
                NSManagedObject *dayEntity = dayEntities[dayIndex];
                NSLog(@"Checking DayEntity: %@", [dayEntity valueForKey:@"date"]);
                
                NSArray *segments = [[dayEntity valueForKey:@"segments"] allObjects];
                
                if (lastDeletedSegmentIndex >= segments.count) {
                    lastDeletedSegmentIndex = 0; // Reset segment index if we move to the next day
                }
                
                for (NSInteger segmentIndex = lastDeletedSegmentIndex; segmentIndex < segments.count; segmentIndex++) {
                    NSManagedObject *segmentEntity = segments[segmentIndex];
                    NSString *segmentURL = [segmentEntity valueForKey:@"url"];

                    if (!segmentURL || [segmentURL isEqualToString:@""]) {
                        continue;
                    }

                    NSNumber *segmentTimeStampNumber = [segmentEntity valueForKey:@"timeStamp"];
                    if (!segmentTimeStampNumber) {
                        continue;
                    }

                    double segmentSecondsSinceMidnight = segmentTimeStampNumber.doubleValue;
                    NSLog(@"Checking segment: %@ (Seconds since midnight: %.2f)", segmentURL, segmentSecondsSinceMidnight);

                    // Check if the segment is within ±60 seconds of any event timestamp
                    BOOL shouldSkipDeletion = NO;
                    for (NSNumber *eventTimestamp in eventTimestamps) {
                        double eventSecondsSinceMidnight = eventTimestamp.doubleValue;
                        double timeDifference = fabs(segmentSecondsSinceMidnight - eventSecondsSinceMidnight);

                        NSLog(@"Comparing with event time: %.2f sec (Difference: %.2f sec)", eventSecondsSinceMidnight, timeDifference);

                        if (timeDifference <= 60) { // Within one minute
                            shouldSkipDeletion = YES;
                            NSLog(@"Skipping deletion for segment at %.2f sec because it is close to an event time!", segmentSecondsSinceMidnight);
                            break;
                        }
                    }

                    if (shouldSkipDeletion) {
                        continue;
                    }

                    NSLog(@"\tChecking Segment: %@", segmentURL);

                    @try {
                        NSString *filePath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:segmentURL];

                        NSError *deleteError = nil;
                        if ([[NSFileManager defaultManager] fileExistsAtPath:filePath]) {
                            if ([[NSFileManager defaultManager] removeItemAtPath:filePath error:&deleteError]) {
                                NSLog(@"\tDeleted %@", segmentURL);
                                // Keep the segmentEntity but clear all but URL (blank)
                                [segmentEntity setValue:@"" forKey:@"url"];
                                [segmentEntity setValue:nil forKey:@"frames"];
                                [segmentEntity setValue:nil forKey:@"duration"];

                                // Save the changes to Core Data
                                NSManagedObjectContext *context = self.fileServer.context;
                                NSError *saveError = nil;
                                if (![context save:&saveError]) {
                                    NSLog(@"Failed to update segment entity: %@", saveError.localizedDescription);
                                } else {
                                    NSLog(@"Successfully cleared segment entity data.");
                                }

                                deletedFile = YES;
                                if((double)[[[[NSFileManager defaultManager] attributesOfFileSystemForPath:NSHomeDirectory() error:nil] objectForKey:NSFileSystemFreeSize] unsignedLongLongValue] / (1024.0 * 1024.0) >= MIN_FREE_SPACE_MB + 500) return YES;

                                // Save indexes
                                [defaults setInteger:dayIndex forKey:@"LastDeletedDayIndex"];
                                [defaults setInteger:segmentIndex forKey:@"LastDeletedSegmentIndex"];
                                [defaults synchronize];
                            } else {
                                NSLog(@"Failed to delete file %@: %@", segmentURL, deleteError.localizedDescription);
                            }
                        }

                    } @catch (NSException *exception) {
                        NSLog(@"Exception while deleting file: %@, reason: %@", segmentURL, exception.reason);
                    }
                }
                
                // If we finish all segments in a day, move to the next one
                lastDeletedSegmentIndex = 0;
            }
            
            // If nothing was deleted, reset the indexes to avoid getting stuck
            if (!deletedFile) {
                NSLog(@"No more files to delete but still low on space! Resetting deletion indexes.");
                [defaults removeObjectForKey:@"LastDeletedDayIndex"];
                [defaults removeObjectForKey:@"LastDeletedSegmentIndex"];
                [defaults synchronize];
                return NO;
            }
        }
    } @catch (NSException *exception) {
        NSLog(@"Exception in ensureFreeDiskSpace: %@", exception.reason);
        return YES;
    }

    NSLog(@"Free space = %f MB", (double)[[[[NSFileManager defaultManager] attributesOfFileSystemForPath:NSHomeDirectory() error:nil] objectForKey:NSFileSystemFreeSize] unsignedLongLongValue] / (1024.0 * 1024.0));
    return YES;
}

- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    @try {
        if (self.recordPressed && self.isRecording && self.assetWriter.status == AVAssetWriterStatusWriting) {
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
            if (self.fileServer.segment_length == 1 && [[NSDate now] timeIntervalSinceDate:self.fileServer.last_req_time] > 60) {
                self.fileServer.segment_length = 60;
            }
            if (elapsedTime >= self.fileServer.segment_length) {
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

            NSMutableArray *frameSquares = [[NSMutableArray alloc] init];

            NSCalendar *calendar = [NSCalendar currentCalendar];
            NSDate *now = [NSDate date];
            NSDateComponents *components = [calendar components:NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay fromDate:now];
            NSDate *midnight = [calendar dateFromComponents:components];
            NSTimeInterval timeStamp = [now timeIntervalSinceDate:midnight];

            NSMutableDictionary *frame = [[NSMutableDictionary alloc] init];
            frame[@"frame_timeStamp"] = @(timeStamp);
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
            if(self.recordPressed) [self.scene processOutput:output withImage:ciImage]; //todo, should event detection without recording be a thing?
            CGImageRelease(cgImage);

            __weak typeof(self) weak_self = self;
            dispatch_async(dispatch_get_main_queue(), ^{
                @try {
                    __strong typeof(weak_self) strongSelf = weak_self;
                    if (!strongSelf) {
                        return;
                    }

                    [strongSelf resetSquares];

                    for (NSArray *detection in output) {
                        NSMutableDictionary *frameSquare = [[NSMutableDictionary alloc] init];
                        frameSquare[@"originX"] = detection[0];
                        frameSquare[@"originY"] = detection[1];
                        frameSquare[@"bottomRightX"] = detection[2];
                        frameSquare[@"bottomRightY"] = detection[3];
                        frameSquare[@"classIndex"] = detection[4];
                        [frameSquares addObject:frameSquare];

                        [strongSelf drawSquareWithTopLeftX:[detection[0] floatValue]
                                                   topLeftY:[detection[1] floatValue]
                                               bottomRightX:[detection[2] floatValue]
                                               bottomRightY:[detection[3] floatValue]
                                                 classIndex:[detection[4] intValue]
                                                aspectRatio:aspect_ratio];
                    }

                    frame[@"squares"] = frameSquares;

                    dispatch_async(self.segmentQueue, ^{
                        if (!self.current_segment_squares) {
                            self.current_segment_squares = [[NSMutableArray alloc] init];
                        }
                        [self.current_segment_squares addObject:frame];
                    });

                    [strongSelf updateFPS];
                    strongSelf.isProcessing = NO;
                } @catch (NSException *exception) {
                    NSLog(@"Exception in main queue block: %@", exception);
                }
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

- (CVPixelBufferRef)addTimeStampToPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    // Get text_size from SettingsManager
    SettingsManager *settings = [SettingsManager sharedManager];
    NSInteger textSize = [settings.text_size intValue];
    
    NSInteger pixelSize = textSize * 2;
    NSInteger spaceSize = textSize;
    NSInteger digitOriginX = spaceSize;
    NSInteger digitOriginY = spaceSize;
    NSInteger height = spaceSize * 2 + pixelSize * 5;

    NSDate *currentDate = [NSDate date];
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
    NSString *timestamp = [dateFormatter stringFromDate:currentDate];

    // Check rotation
    CGAffineTransform transform = self.videoWriterInput.transform;
    CGFloat angle = atan2(transform.b, transform.a);

    // Convert radians to degrees
    CGFloat degrees = angle * (180.0 / M_PI);

    size_t width_res = CVPixelBufferGetWidth(pixelBuffer);
    size_t height_res = CVPixelBufferGetHeight(pixelBuffer);

    if (degrees == 90 || degrees == -90) {
        pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer
                                                   withColor:[UIColor blackColor]
                                                    originX:0
                                                    originY:height_res - (pixelSize * 3 + spaceSize) * timestamp.length
                                                     width:height
                                                    height:(pixelSize * 3 + spaceSize) * timestamp.length
                                                   opacity:0.4];
    } else if (degrees == 180 || degrees == -180) {
        pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer
                                                   withColor:[UIColor blackColor]
                                                    originX:width_res - (pixelSize * 3 + spaceSize) * timestamp.length
                                                    originY:height_res - height
                                                     width:(pixelSize * 3 + spaceSize) * timestamp.length
                                                    height:height
                                                   opacity:0.4];
    } else {
        // No rotation
        pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer
                                                   withColor:[UIColor blackColor]
                                                    originX:0
                                                    originY:0
                                                     width:(pixelSize * 3 + spaceSize) * timestamp.length
                                                    height:height
                                                   opacity:0.4];
    }

    for (NSUInteger k = 0; k < [timestamp length]; k++) {
        unichar character = [timestamp characterAtIndex:k];
        if (character == ' ') {
            digitOriginX += pixelSize * 3;
            continue;
        }
        NSString *key = [NSString stringWithFormat:@"%C", character];
        for (int i = 0; i < [self.digits[key] count]; i++) {
            if (degrees == 90 || degrees == -90) {
                pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer
                                                           withColor:[UIColor whiteColor]
                                                            originX:digitOriginY + [self.digits[key][i][1] doubleValue] * pixelSize
                                                            originY:height_res - (digitOriginX + [self.digits[key][i][0] doubleValue] * pixelSize) - ([self.digits[key][i][2] doubleValue] * pixelSize)
                                                             width:[self.digits[key][i][3] doubleValue] * pixelSize
                                                            height:[self.digits[key][i][2] doubleValue] * pixelSize
                                                           opacity:1];
            } else if (degrees == 180 || degrees == -180) {
                pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer
                                                           withColor:[UIColor whiteColor]
                                                            originX:width_res - (digitOriginX + [self.digits[key][i][0] doubleValue] * pixelSize) - ([self.digits[key][i][2] doubleValue] * pixelSize)
                                                            originY:height_res - (digitOriginY + [self.digits[key][i][1] doubleValue] * pixelSize) - ([self.digits[key][i][3] doubleValue] * pixelSize)
                                                             width:[self.digits[key][i][2] doubleValue] * pixelSize
                                                            height:[self.digits[key][i][3] doubleValue] * pixelSize
                                                           opacity:1];
            } else {
                pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer
                                                           withColor:[UIColor whiteColor]
                                                            originX:digitOriginX + [self.digits[key][i][0] doubleValue] * pixelSize
                                                            originY:digitOriginY + [self.digits[key][i][1] doubleValue] * pixelSize
                                                             width:[self.digits[key][i][2] doubleValue] * pixelSize
                                                            height:[self.digits[key][i][3] doubleValue] * pixelSize
                                                           opacity:1];
            }
        }
        digitOriginX += pixelSize * 3 + spaceSize;
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



