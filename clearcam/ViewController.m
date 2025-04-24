#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <Metal/Metal.h>
#import "Yolo.h"
#import "FileServer.h"
#import "SettingsManager.h"
#import "StoreManager.h"
#import "SceneState.h"
#import "SettingsViewController.h"
#import "GalleryViewController.h"
#import <Security/Security.h>
#import <CommonCrypto/CommonCryptor.h>

@interface ViewController ()

@property (nonatomic, strong) AVCaptureSession *captureSession;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *previewLayer;
@property (nonatomic, strong) UILabel *fpsLabel;
@property (nonatomic, strong) UIButton *recordButton;
@property (nonatomic, strong) UIButton *settingsButton;
@property (nonatomic, strong) UIButton *galleryButton;
@property (nonatomic, assign) CFTimeInterval lastFrameTime;
@property (nonatomic, assign) NSUInteger frameCount;
@property (nonatomic, strong) Yolo *yolo;
@property (nonatomic, strong) CIContext *ciContext;

@property (atomic, assign) BOOL isProcessing;
@property (nonatomic, assign) BOOL isProcessingCoreData;
@property (nonatomic, assign) BOOL recordPressed;
@property (nonatomic, assign) BOOL isRecording;
@property (nonatomic, assign) BOOL isStreaming; //over the internet
@property (nonatomic, assign) NSTimeInterval last_check_time;
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
@property (nonatomic, strong) NSString *streamLink;
@property (nonatomic, strong) dispatch_queue_t segmentQueue;
@property (nonatomic, strong) dispatch_queue_t finishRecordingQueue;
@property (nonatomic, strong) SceneState *scene;

#define MIN_FREE_SPACE_MB 500  //threshold to start deleting

@end

@implementation ViewController

NSMutableDictionary *classColorMap;

- (void)viewDidLoad {
    NSLog(@"NSCameraUsageDescription: %@", [[NSBundle mainBundle] objectForInfoDictionaryKey:@"NSCameraUsageDescription"]);
    [super viewDidLoad];
    if(![[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"] || ![[NSDate date] compare:[[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"]] || [[NSDate date] compare:[[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"]] == NSOrderedDescending){
        [[StoreManager sharedInstance] verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
            // do nothing
        }];
    }
    self.streamLink = @"";
    self.recordPressed = NO;
    self.last_check_time = [[NSDate date] timeIntervalSince1970];
    self.scene = [[SceneState alloc] init];
    self.segmentQueue = dispatch_queue_create("com.example.segmentQueue", DISPATCH_QUEUE_SERIAL);
    self.finishRecordingQueue = dispatch_queue_create("com.example.finishRecordingQueue", DISPATCH_QUEUE_SERIAL);
    
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
    self.segmentLock = [[NSLock alloc] init];
    self.isStreaming = NO;
        
    self.ciContext = [CIContext context];
    self.yolo = [[Yolo alloc] init];
    self.seg_number = 0;
    self.fileServer = [FileServer sharedInstance];
    [self.fileServer start];
    self.backgroundContext = [[NSManagedObjectContext alloc] initWithConcurrencyType:NSPrivateQueueConcurrencyType];
    self.backgroundContext.mergePolicy = NSMergeByPropertyStoreTrumpMergePolicy;
    self.backgroundContext.parentContext = self.fileServer.context;
    
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(handleDeviceOrientationChange)
                                                 name:UIDeviceOrientationDidChangeNotification
                                               object:nil];
    
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
    
    SettingsManager *settings = [SettingsManager sharedManager];
    [settings addObserver:self forKeyPath:@"width" options:NSKeyValueObservingOptionNew context:nil];
    [settings addObserver:self forKeyPath:@"height" options:NSKeyValueObservingOptionNew context:nil];
    [settings addObserver:self forKeyPath:@"text_size" options:NSKeyValueObservingOptionNew context:nil];
    [settings addObserver:self forKeyPath:@"preset" options:NSKeyValueObservingOptionNew context:nil];
    
    [self setupCameraWithWidth:settings.width height:settings.height];
    [self.captureSession startRunning];
    [self startNewRecording];
    [self setupUI];
    UIDeviceOrientation initialOrientation = [self getCurrentOrientation];
}

- (void)refreshView {
    dispatch_async(dispatch_get_main_queue(), ^{
        // Preserve recording state
        BOOL wasRecording = self.recordPressed;
        
        // Stop and clean up the current capture session
        if (self.captureSession.isRunning) {
            [self.captureSession stopRunning];
        }
        [self.captureSession.inputs enumerateObjectsUsingBlock:^(AVCaptureInput *input, NSUInteger idx, BOOL *stop) {
            [self.captureSession removeInput:input];
        }];
        [self.captureSession.outputs enumerateObjectsUsingBlock:^(AVCaptureOutput *output, NSUInteger idx, BOOL *stop) {
            [self.captureSession removeOutput:output];
        }];
        self.captureSession = nil;
        
        // Finish any ongoing recording
        if (self.isRecording) {
            [self finishRecording];
        }
        
        // Remove notification observers
        [[NSNotificationCenter defaultCenter] removeObserver:self name:UIDeviceOrientationDidChangeNotification object:nil];
        [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
        
        // Remove KVO observers
        SettingsManager *settings = [SettingsManager sharedManager];
        [settings removeObserver:self forKeyPath:@"width"];
        [settings removeObserver:self forKeyPath:@"height"];
        [settings removeObserver:self forKeyPath:@"text_size"];
        [settings removeObserver:self forKeyPath:@"preset"];
        
        // Reset UI and remove all sublayers
        [self resetUI];
        [self.previewLayer removeFromSuperlayer];
        self.previewLayer = nil;
        
        // Reset all properties to initial state
        self.fpsLabel = nil;
        self.recordButton = nil;
        self.settingsButton = nil;
        self.galleryButton = nil;
        self.lastFrameTime = 0;
        self.frameCount = 0;
        self.yolo = nil;
        self.ciContext = nil;
        self.isProcessing = NO;
        self.isProcessingCoreData = NO;
        self.recordPressed = wasRecording; // Restore recording state
        self.isRecording = NO;
        self.last_check_time = [[NSDate date] timeIntervalSince1970];
        self.startTime = kCMTimeInvalid;
        self.currentTime = kCMTimeZero;
        self.assetWriter = nil;
        self.videoWriterInput = nil;
        self.adaptor = nil;
        self.fileServer = nil;
        self.seg_number = 0;
        self.current_file_timestamp = nil;
        self.digits = nil;
        self.current_segment_squares = nil;
        self.segmentLock = nil;
        self.backgroundContext = nil;
        self.dayFolderName = nil;
        self.segmentQueue = nil;
        self.finishRecordingQueue = nil;
        self.scene = nil;
        
        // Reinitialize all components as in viewDidLoad
        self.scene = [[SceneState alloc] init];
        self.segmentQueue = dispatch_queue_create("com.example.segmentQueue", DISPATCH_QUEUE_SERIAL);
        self.finishRecordingQueue = dispatch_queue_create("com.example.finishRecordingQueue", DISPATCH_QUEUE_SERIAL);
        
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
        self.segmentLock = [[NSLock alloc] init];
        
        self.ciContext = [CIContext context];
        self.yolo = [[Yolo alloc] init];
        self.seg_number = 0;
        self.fileServer = [FileServer sharedInstance];
        [self.fileServer start];
        self.backgroundContext = [[NSManagedObjectContext alloc] initWithConcurrencyType:NSPrivateQueueConcurrencyType];
        self.backgroundContext.mergePolicy = NSMergeByPropertyStoreTrumpMergePolicy;
        self.backgroundContext.parentContext = self.fileServer.context;
        
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(handleDeviceOrientationChange)
                                                     name:UIDeviceOrientationDidChangeNotification
                                                   object:nil];
        
        [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
        
        [settings addObserver:self forKeyPath:@"width" options:NSKeyValueObservingOptionNew context:nil];
        [settings addObserver:self forKeyPath:@"height" options:NSKeyValueObservingOptionNew context:nil];
        [settings addObserver:self forKeyPath:@"text_size" options:NSKeyValueObservingOptionNew context:nil];
        [settings addObserver:self forKeyPath:@"preset" options:NSKeyValueObservingOptionNew context:nil];
        
        // Re-setup camera with current settings
        [self setupCameraWithWidth:settings.width height:settings.height];
        
        // Re-setup UI
        [self setupUI];
        
        // Update record button state if was recording
        if (self.recordPressed) {
            [self.recordButton setTitle:@"Stop" forState:UIControlStateNormal];
            for (CALayer *layer in self.recordButton.layer.sublayers) {
                if ([layer.name isEqualToString:@"redShape"]) {
                    CAShapeLayer *redShape = (CAShapeLayer *)layer;
                    [CATransaction begin];
                    [CATransaction setAnimationDuration:0.2];
                    redShape.path = [UIBezierPath bezierPathWithRect:CGRectMake(25, 25, 30, 30)].CGPath;
                    [CATransaction commit];
                    break;
                }
            }
        }
        
        // Set initial orientation
        [self setInitialOrientation];
        
        // Start the capture session
        [self.captureSession startRunning];
        
        // Start a new recording if was recording
        if (self.recordPressed) {
            [self startNewRecording];
        }
        
        // Mimic viewWillAppear setup
        [UIApplication sharedApplication].idleTimerDisabled = YES;
    });
}


- (void)setInitialOrientation {
    UIDeviceOrientation deviceOrientation = [[UIDevice currentDevice] orientation];
    AVCaptureVideoOrientation videoOrientation;
    if (deviceOrientation == UIDeviceOrientationUnknown || deviceOrientation == UIDeviceOrientationFaceUp || deviceOrientation == UIDeviceOrientationFaceDown) {
        UIInterfaceOrientation statusBarOrientation = UIApplication.sharedApplication.statusBarOrientation;
        if (statusBarOrientation == UIInterfaceOrientationLandscapeLeft) {
            deviceOrientation = UIDeviceOrientationLandscapeRight;
        } else if (statusBarOrientation == UIInterfaceOrientationLandscapeRight) {
            deviceOrientation = UIDeviceOrientationLandscapeLeft;
        } else {
            deviceOrientation = UIDeviceOrientationPortrait;
        }
    }

    if (deviceOrientation == UIDeviceOrientationPortrait) {
        videoOrientation = AVCaptureVideoOrientationPortrait;
    } else if (deviceOrientation == UIDeviceOrientationLandscapeLeft) {
        videoOrientation = AVCaptureVideoOrientationLandscapeRight;
    } else if (deviceOrientation == UIDeviceOrientationLandscapeRight) {
        videoOrientation = AVCaptureVideoOrientationLandscapeLeft;
    } else {
        videoOrientation = AVCaptureVideoOrientationPortrait; // Default
    }

    if (self.previewLayer.connection.isVideoOrientationSupported) {
        self.previewLayer.connection.videoOrientation = videoOrientation;
    }

    // Set preview frame to full view bounds
    self.previewLayer.frame = self.view.bounds;
    [self updateButtonFrames];
}


- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    [UIApplication sharedApplication].idleTimerDisabled = YES;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self.captureSession startRunning];
    });
    
    // Force button layout after view is about to appear
    UIDeviceOrientation initialOrientation = [self getCurrentOrientation];
    [self setVideoOrientationForOrientation:initialOrientation];
    self.previewLayer.frame = self.view.bounds;
    [self updateButtonFrames];
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary<NSKeyValueChangeKey,id> *)change context:(void *)context {
    if ([keyPath isEqualToString:@"preset"]) {
        [self finishRecording];
        [self resetUI];
        SettingsManager *settings = [SettingsManager sharedManager];
        [self setupCameraWithWidth:settings.width height:settings.height];
        if(self.isRecording) [self startNewRecording];
        [self setupUI];
    }
}

- (void)dealloc {
    [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
}

- (UIDeviceOrientation)getCurrentOrientation {
    UIDeviceOrientation deviceOrientation = [[UIDevice currentDevice] orientation];
    UIInterfaceOrientation statusBarOrientation = UIApplication.sharedApplication.statusBarOrientation;

    if (deviceOrientation == UIDeviceOrientationUnknown || deviceOrientation == UIDeviceOrientationFaceUp || deviceOrientation == UIDeviceOrientationFaceDown) {
        if (statusBarOrientation == UIInterfaceOrientationLandscapeLeft) {
            return UIDeviceOrientationLandscapeRight;
        } else if (statusBarOrientation == UIInterfaceOrientationLandscapeRight) {
            return UIDeviceOrientationLandscapeLeft;
        } else {
            return UIDeviceOrientationPortrait;
        }
    }
    return deviceOrientation;
}

- (void)setVideoOrientationForOrientation:(UIDeviceOrientation)orientation {
    AVCaptureVideoOrientation videoOrientation;

    if (orientation == UIDeviceOrientationPortrait) {
        videoOrientation = AVCaptureVideoOrientationPortrait;
    } else if (orientation == UIDeviceOrientationLandscapeLeft) {
        videoOrientation = AVCaptureVideoOrientationLandscapeRight;
    } else if (orientation == UIDeviceOrientationLandscapeRight) {
        videoOrientation = AVCaptureVideoOrientationLandscapeLeft;
    } else {
        videoOrientation = AVCaptureVideoOrientationPortrait;
    }

    if (self.previewLayer.connection.isVideoOrientationSupported) {
        self.previewLayer.connection.videoOrientation = videoOrientation;
    }
}

- (void)handleDeviceOrientationChange {
    UIDeviceOrientation deviceOrientation = [self getCurrentOrientation];
    BOOL isSupportedLayoutOrientation = (deviceOrientation == UIDeviceOrientationPortrait ||
                                         deviceOrientation == UIDeviceOrientationLandscapeLeft ||
                                         deviceOrientation == UIDeviceOrientationLandscapeRight);

    if (isSupportedLayoutOrientation) {
        [self setVideoOrientationForOrientation:deviceOrientation];

        [UIView animateWithDuration:0.3 animations:^{
            self.previewLayer.frame = self.view.bounds;
            [self updateButtonFrames];
        }];
    }
}

- (void)setupCameraWithWidth:(NSString *)width height:(NSString *)height {
    self.captureSession = [[AVCaptureSession alloc] init];
    NSString *presetString = [NSString stringWithFormat:@"AVCaptureSessionPreset%@x%@", width, height];
    if(self.isStreaming) presetString = @"AVCaptureSessionPreset640x480";

    if ([self.captureSession canSetSessionPreset:presetString]) {
        self.captureSession.sessionPreset = presetString;
    } else return;

    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    NSError *error = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (!input) return;
    [self.captureSession addInput:input];

    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    output.videoSettings = @{(NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)};
    output.alwaysDiscardsLateVideoFrames = YES;
    [output setSampleBufferDelegate:self queue:dispatch_get_main_queue()];
    [self.captureSession addOutput:output];

    self.previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.captureSession];
    self.previewLayer.videoGravity = AVLayerVideoGravityResizeAspect;
    [self.view.layer addSublayer:self.previewLayer];
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
    if(self.isStreaming){
        videoHeight = 480;
        videoWidth = 640;
    }
    
    // Create a folder for the day within the documents directory
    NSURL *documentsDirectory = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    NSURL *dayFolderURL = [documentsDirectory URLByAppendingPathComponent:self.dayFolderName];
    
    // Ensure the directory exists
    NSError *error = nil;
    if (![[NSFileManager defaultManager] fileExistsAtPath:dayFolderURL.path]) {
        [[NSFileManager defaultManager] createDirectoryAtURL:dayFolderURL withIntermediateDirectories:YES attributes:nil error:&error];
        if (error) return;
    }
    
    NSURL *outputURL = [dayFolderURL URLByAppendingPathComponent:[NSString stringWithFormat:@"output_%@.mp4", finalTimestamp]];
    self.assetWriter = [AVAssetWriter assetWriterWithURL:outputURL fileType:AVFileTypeMPEG4 error:&error];
    if (error) return;
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
    }
    
    if (self.isStreaming) {
        NSMutableDictionary *mutableSettings = [videoSettings mutableCopy];
        mutableSettings[AVVideoCompressionPropertiesKey] = @{
            AVVideoAverageBitRateKey: @(500000)
        };
        videoSettings = [mutableSettings copy];
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
        return;
    }
    
    self.startTime = kCMTimeInvalid;
    self.currentTime = kCMTimeZero;
    self.isRecording = YES;
    [self.assetWriter startWriting];
    [self.assetWriter startSessionAtSourceTime:kCMTimeZero];
    if(![self ensureFreeDiskSpace]) [self stopRecording];
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
    self.fpsLabel = [[UILabel alloc] init];
    self.fpsLabel.backgroundColor = [UIColor colorWithWhite:0 alpha:0.5];
    self.fpsLabel.textColor = [UIColor whiteColor];
    self.fpsLabel.font = [UIFont boldSystemFontOfSize:14];
    self.fpsLabel.text = @"FPS: 0";
    [self.fpsLabel sizeToFit]; // Size to fit the text
    self.fpsLabel.frame = CGRectMake(10, 50, self.fpsLabel.frame.size.width + 8, self.fpsLabel.frame.size.height + 4);
    [self.view addSubview:self.fpsLabel];

    self.recordButton = [UIButton buttonWithType:UIButtonTypeCustom];
    CGFloat buttonSize = 80;
    self.recordButton.frame = CGRectMake(0, 0, buttonSize, buttonSize);
    self.recordButton.backgroundColor = [UIColor clearColor];
    self.recordButton.clipsToBounds = YES;
    CAShapeLayer *whiteRing = [CAShapeLayer layer];
    whiteRing.path = [UIBezierPath bezierPathWithOvalInRect:CGRectMake(5, 5, buttonSize - 10, buttonSize - 10)].CGPath;
    whiteRing.fillColor = [UIColor clearColor].CGColor;
    whiteRing.strokeColor = [UIColor whiteColor].CGColor;
    whiteRing.lineWidth = 4.0;
    [self.recordButton.layer addSublayer:whiteRing];

    CAShapeLayer *redShape = [CAShapeLayer layer];
    redShape.path = [UIBezierPath bezierPathWithOvalInRect:CGRectMake(10, 10, buttonSize - 20, buttonSize - 20)].CGPath;
    redShape.fillColor = [UIColor redColor].CGColor;
    redShape.name = @"redShape";
    [self.recordButton.layer addSublayer:redShape];

    [self.recordButton setTitle:@"Record" forState:UIControlStateNormal];
    self.recordButton.titleLabel.alpha = 0;
    [self.recordButton addTarget:self action:@selector(toggleRecording) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.recordButton];

    self.settingsButton = [UIButton buttonWithType:UIButtonTypeSystem];
    UIImage *gearIcon = [UIImage systemImageNamed:@"gear"];
    [self.settingsButton setImage:gearIcon forState:UIControlStateNormal];
    CGFloat gearButtonSize = 50;
    self.settingsButton.frame = CGRectMake(0, 0, gearButtonSize, gearButtonSize);
    self.settingsButton.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.3];
    self.settingsButton.layer.cornerRadius = gearButtonSize / 2;
    self.settingsButton.clipsToBounds = YES;
    self.settingsButton.tintColor = [UIColor whiteColor];
    [self.settingsButton addTarget:self action:@selector(openSettings) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.settingsButton];

    self.galleryButton = [UIButton buttonWithType:UIButtonTypeSystem];
    UIImage *galleryIcon = [UIImage systemImageNamed:@"photo.on.rectangle"];
    [self.galleryButton setImage:galleryIcon forState:UIControlStateNormal];
    CGFloat galleryButtonSize = 50;
    self.galleryButton.frame = CGRectMake(0, 0, galleryButtonSize, galleryButtonSize);
    self.galleryButton.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.3];
    self.galleryButton.layer.cornerRadius = galleryButtonSize / 2;
    self.galleryButton.clipsToBounds = YES;
    self.galleryButton.tintColor = [UIColor whiteColor];
    [self.galleryButton addTarget:self action:@selector(galleryButtonPressed) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.galleryButton];

    [self updateButtonFrames]; // Just call it once
}

- (void)updateButtonFrames {
    CGFloat recordButtonSize = 80;
    CGFloat settingsButtonSize = 50;
    CGFloat galleryButtonSize = 50;
    CGFloat spacing = 60;

    UIEdgeInsets safeAreaInsets = self.view.safeAreaInsets;
    CGFloat screenWidth = self.view.bounds.size.width;
    CGFloat screenHeight = self.view.bounds.size.height;

    self.recordButton.layer.cornerRadius = recordButtonSize / 2;
    self.settingsButton.layer.cornerRadius = settingsButtonSize / 2;
    self.galleryButton.layer.cornerRadius = galleryButtonSize / 2;

    UIDeviceOrientation orientation = [self getCurrentOrientation];

    if (orientation == UIDeviceOrientationPortraitUpsideDown ||
        orientation == UIDeviceOrientationFaceUp ||
        orientation == UIDeviceOrientationFaceDown) {
        return;
    }
    if (orientation == UIDeviceOrientationLandscapeRight) {
        CGFloat leftMargin = MAX(20, safeAreaInsets.left + 10);
        self.recordButton.frame = CGRectMake(leftMargin, screenHeight / 2 - recordButtonSize / 2, recordButtonSize, recordButtonSize);
        self.settingsButton.frame = CGRectMake(leftMargin + (recordButtonSize / 2) - (settingsButtonSize / 2),
                                               screenHeight / 2 - recordButtonSize / 2 - settingsButtonSize - spacing,
                                               settingsButtonSize, settingsButtonSize);
        self.galleryButton.frame = CGRectMake(leftMargin + (recordButtonSize / 2) - (galleryButtonSize / 2),
                                              screenHeight / 2 + recordButtonSize / 2 + spacing,
                                              galleryButtonSize, galleryButtonSize);
        self.fpsLabel.frame = CGRectMake(screenWidth - self.fpsLabel.frame.size.width - MAX(20, safeAreaInsets.right + 10),
                                        30,
                                        self.fpsLabel.frame.size.width,
                                        self.fpsLabel.frame.size.height);
    } else if (orientation == UIDeviceOrientationLandscapeLeft) {
        CGFloat rightMargin = MAX(20, safeAreaInsets.right + 10);
        self.recordButton.frame = CGRectMake(screenWidth - recordButtonSize - rightMargin, screenHeight / 2 - recordButtonSize / 2, recordButtonSize, recordButtonSize);
        self.settingsButton.frame = CGRectMake(screenWidth - recordButtonSize - rightMargin + (recordButtonSize / 2) - (settingsButtonSize / 2),
                                               screenHeight / 2 - recordButtonSize / 2 - settingsButtonSize - spacing,
                                               settingsButtonSize, settingsButtonSize);
        self.galleryButton.frame = CGRectMake(screenWidth - recordButtonSize - rightMargin + (recordButtonSize / 2) - (galleryButtonSize / 2),
                                              screenHeight / 2 + recordButtonSize / 2 + spacing,
                                              galleryButtonSize, galleryButtonSize);
        self.fpsLabel.frame = CGRectMake(MAX(10, safeAreaInsets.left + 5),
                                        30,
                                        self.fpsLabel.frame.size.width,
                                        self.fpsLabel.frame.size.height);
    } else {
        CGFloat bottomMargin = MAX(20, safeAreaInsets.bottom + 10);
        CGFloat recordX = (screenWidth - recordButtonSize) / 2;
        self.recordButton.frame = CGRectMake(recordX, screenHeight - recordButtonSize - bottomMargin, recordButtonSize, recordButtonSize);
        self.settingsButton.frame = CGRectMake(recordX + recordButtonSize + spacing,
                                               screenHeight - bottomMargin - recordButtonSize / 2 - settingsButtonSize / 2,
                                               settingsButtonSize, settingsButtonSize);
        self.galleryButton.frame = CGRectMake(recordX - galleryButtonSize - spacing,
                                              screenHeight - bottomMargin - recordButtonSize / 2 - galleryButtonSize / 2,
                                              galleryButtonSize, galleryButtonSize);
        self.fpsLabel.frame = CGRectMake(10, 50, self.fpsLabel.frame.size.width, self.fpsLabel.frame.size.height);
    }
}

- (void)galleryButtonPressed {
    [self.captureSession stopRunning];
    GalleryViewController *galleryVC = [[GalleryViewController alloc] init];
    [self.navigationController pushViewController:galleryVC animated:YES];
}

- (void)toggleRecording {
    CAShapeLayer *redShape = nil;
    for (CALayer *layer in self.recordButton.layer.sublayers) {
        if ([layer.name isEqualToString:@"redShape"]) {
            redShape = (CAShapeLayer *)layer;
            break;
        }
    }

    if ([[self.recordButton titleForState:UIControlStateNormal] isEqualToString:@"Record"]) {
        self.recordPressed = YES;
        [self.recordButton setTitle:@"Stop" forState:UIControlStateNormal];
        // Change to red square
        if (redShape) {
            [CATransaction begin];
            [CATransaction setAnimationDuration:0.2];
            redShape.path = [UIBezierPath bezierPathWithRect:CGRectMake(25, 25, 30, 30)].CGPath;
            [CATransaction commit];
        }
    } else {
        self.recordPressed = NO;
        [self.recordButton setTitle:@"Record" forState:UIControlStateNormal];
        // Change back to red circle
        if (redShape) {
            [CATransaction begin];
            [CATransaction setAnimationDuration:0.2];
            redShape.path = [UIBezierPath bezierPathWithOvalInRect:CGRectMake(10, 10, 60, 60)].CGPath;
            [CATransaction commit];
        }
    }
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    self.previewLayer.frame = self.view.bounds;
    [self updateButtonFrames];
}

- (void)openSettings {
    if(self.recordPressed) [self toggleRecording];
    [self.captureSession stopRunning];
    SettingsViewController *settingsVC = [[SettingsViewController alloc] init];
    [self.navigationController pushViewController:settingsVC animated:YES];
}

- (void)viewWillLayoutSubviews {
    [super viewWillLayoutSubviews];
    self.previewLayer.frame = self.view.bounds;
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [self.captureSession stopRunning];
    // Enable auto-locking (important!)
    [UIApplication sharedApplication].idleTimerDisabled = NO;
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
    if (!(self.isRecording && self.assetWriter.status == AVAssetWriterStatusWriting)) return;
    self.isRecording = NO;
    [self.videoWriterInput markAsFinished];
}


- (void)finishRecording {
    NSLog(@"rory finish recording %ld %@", (long)[FileServer sharedInstance].segment_length, self.isStreaming ? @"YES" : @"NO");
    if (!(self.isRecording && self.assetWriter.status == AVAssetWriterStatusWriting)) return;
    
    NSString *segmentsDirectory = [[NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject] stringByAppendingPathComponent:self.dayFolderName];
    [[NSFileManager defaultManager] createDirectoryAtPath:segmentsDirectory withIntermediateDirectories:YES attributes:nil error:nil];
    NSCalendar *calendar = [NSCalendar currentCalendar];
    
    dispatch_async(self.finishRecordingQueue, ^{
        while (self.isProcessingCoreData) {
            [NSThread sleepForTimeInterval:0.001];
        }
        self.isProcessingCoreData = YES;
        self.isRecording = NO;
        [self.videoWriterInput markAsFinished];
        
        [self.assetWriter finishWritingWithCompletionHandler:^{
            if (!self.assetWriter.outputURL) {
                NSLog(@"❌ No output URL for asset writer");
                dispatch_async(dispatch_get_main_queue(), ^{
                    self.isProcessingCoreData = NO;
                    [self startNewRecording];
                });
                return;
            }
            
            AVAsset *asset = [AVAsset assetWithURL:self.assetWriter.outputURL];
            CMTime time = asset.duration;
            NSString *segmentURL = [NSString stringWithFormat:@"%@/%@", [[self.assetWriter.outputURL URLByDeletingLastPathComponent] lastPathComponent], self.assetWriter.outputURL.lastPathComponent];
            NSDateComponents *components = [calendar components:NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay fromDate:self.current_file_timestamp];
            NSTimeInterval timeStamp = [self.current_file_timestamp timeIntervalSinceDate:[calendar dateFromComponents:components]];
            NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
            [formatter setDateFormat:@"yyyy-MM-dd"];
            NSString *thisDayFoler = [formatter stringFromDate:self.current_file_timestamp];
            
            if(self.isStreaming){
                // Prepare file for upload
                NSString *tempFilePath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"segment.mp4.aes"];
                NSError *fileError = nil;
                NSFileManager *fileManager = [NSFileManager defaultManager];
                if ([fileManager fileExistsAtPath:tempFilePath]) {
                    [fileManager removeItemAtPath:tempFilePath error:&fileError];
                    if (fileError) {
                        NSLog(@"❌ Failed to remove existing temp file: %@", fileError);
                    }
                }
                
                // Copy or move the video file to the temp path
                if ([fileManager copyItemAtURL:self.assetWriter.outputURL toURL:[NSURL fileURLWithPath:tempFilePath] error:&fileError]) {
                    NSLog(@"✅ Copied video to temp path for upload: %@", tempFilePath);
                } else {
                    NSLog(@"❌ Failed to copy video to temp path: %@", fileError);
                }
                
                // Trigger upload if file was successfully copied
                if ([fileManager fileExistsAtPath:tempFilePath]) {
                    [self uploadSegment];
                } else {
                    NSLog(@"❌ Temp file not found, skipping upload");
                }
            }
            
            [self.backgroundContext performBlock:^{
                // Save background context
                NSError *error = nil;
                if ([self.backgroundContext save:&error]) {
                    // Save parent context only if necessary
                    NSError *parentError = nil;
                    [self.fileServer.context save:&parentError];
                } else {
                    NSLog(@"❌ Failed to save background context: %@", error);
                }
                
                // Create a local copy of current_segment_squares
                __block NSArray *segmentSquaresCopy;
                dispatch_sync(self.segmentQueue, ^{
                    segmentSquaresCopy = [self.current_segment_squares copy];
                    [self.current_segment_squares removeAllObjects];
                });
                
                // Fetch or create DayEntity efficiently
                NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"DayEntity"];
                fetchRequest.predicate = [NSPredicate predicateWithFormat:@"date == %@", thisDayFoler];
                fetchRequest.fetchLimit = 1;
                
                NSArray *fetchedDays = [self.backgroundContext executeFetchRequest:fetchRequest error:&error];
                NSManagedObject *dayEntity;
                
                if (error) {
                    NSLog(@"❌ Fetch request failed: %@", error);
                    return;
                }
                
                if (fetchedDays.count > 0) {
                    dayEntity = fetchedDays.firstObject;
                } else {
                    dayEntity = [NSEntityDescription insertNewObjectForEntityForName:@"DayEntity" inManagedObjectContext:self.backgroundContext];
                    [dayEntity setValue:self.dayFolderName forKey:@"date"];
                }
                
                // Insert new SegmentEntity
                NSManagedObject *newSegment = [NSEntityDescription insertNewObjectForEntityForName:@"SegmentEntity" inManagedObjectContext:self.backgroundContext];
                [newSegment setValuesForKeysWithDictionary:@{
                    @"url": segmentURL,
                    @"timeStamp": @(timeStamp),
                    @"duration": @(CMTimeGetSeconds(time)),
                    @"orientation": @((int16_t)self.previewLayer.connection.videoOrientation)
                }];
                NSMutableArray<NSManagedObject *> *segmentFrames = [NSMutableArray arrayWithCapacity:segmentSquaresCopy.count];
                
                for (NSDictionary *frameData in segmentSquaresCopy) {
                    NSManagedObject *newFrame = [NSEntityDescription insertNewObjectForEntityForName:@"FrameEntity" inManagedObjectContext:self.backgroundContext];
                    [newFrame setValuesForKeysWithDictionary:@{
                        @"frame_timeStamp": frameData[@"frame_timeStamp"] ?: @0,
                        @"aspect_ratio": frameData[@"aspect_ratio"] ?: @0,
                        @"res": frameData[@"res"] ?: @0
                    }];
                    
                    NSArray *squaresData = frameData[@"squares"];
                    NSMutableArray<NSManagedObject *> *frameSquares = [NSMutableArray arrayWithCapacity:squaresData.count];
                    
                    for (NSDictionary *squareData in squaresData) {
                        NSManagedObject *newSquare = [NSEntityDescription insertNewObjectForEntityForName:@"SquareEntity" inManagedObjectContext:self.backgroundContext];
                        [newSquare setValuesForKeysWithDictionary:squareData];
                        [frameSquares addObject:newSquare];
                    }
                    
                    [newFrame setValue:[NSOrderedSet orderedSetWithArray:frameSquares] forKey:@"squares"];
                    [segmentFrames addObject:newFrame];
                }
                
                [newSegment setValue:[NSOrderedSet orderedSetWithArray:segmentFrames] forKey:@"frames"];
                [[dayEntity mutableOrderedSetValueForKey:@"segments"] addObject:newSegment];
                
                // Save the context again after modifications
                NSError *saveError = nil;
                if ([self.backgroundContext save:&saveError]) {
                    NSLog(@"✅ Saved segment to Core Data");
                } else {
                    NSLog(@"❌ Failed to save segment to Core Data: %@", saveError);
                }
            }];
                        
            dispatch_async(dispatch_get_main_queue(), ^{
                self.isProcessingCoreData = NO;
                [self startNewRecording];
            });
        }];
    });
}

- (void)uploadSegment {
    NSString *filePath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"segment.mp4.aes"];
    NSData *fileData = [NSData dataWithContentsOfFile:filePath];

    if (!fileData) {
        NSLog(@"❌ Failed to read segment.mp4 for upload");
        return;
    }

    // Replace with your actual signed URL
    NSString *signedUrl = self.streamLink;
    
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:[NSURL URLWithString:signedUrl]];
    request.HTTPMethod = @"PUT";
    [request setValue:@"video/mp4" forHTTPHeaderField:@"Content-Type"];
    [request setValue:[NSString stringWithFormat:@"%lu", (unsigned long)fileData.length] forHTTPHeaderField:@"Content-Length"];
    fileData = [self encryptData:fileData withKey:@"open_please"];
    
    NSInteger targetSize = 200*1024;
    NSMutableData *mutableFileData = [fileData mutableCopy];
    if (mutableFileData.length < targetSize) {
        NSUInteger paddingNeeded = targetSize - mutableFileData.length;
        NSMutableData *padding = [NSMutableData dataWithLength:paddingNeeded];
        [mutableFileData appendData:padding];
    }
    
    [[[NSURLSession sharedSession] uploadTaskWithRequest:request
                                               fromData:mutableFileData
                                      completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
        NSLog(@"size of segment.mp4.aes: %.2f KB", (float)mutableFileData.length / 1024.0);
        if (error) {
            NSLog(@"❌ Upload failed: %@", error);
        } else if (httpResponse.statusCode >= 200 && httpResponse.statusCode < 300) {
            NSLog(@"✅ Uploaded segment.mp4.aes: %.2f KB", (float)mutableFileData.length / 1024.0);
        } else {
            NSLog(@"❌ Upload failed with status %ld", (long)httpResponse.statusCode);
        }
    }] resume];
}

#define MAGIC_NUMBER 0x4D41474943ULL // "MAGIC" in ASCII as a 64-bit value
#define HEADER_SIZE (sizeof(uint64_t)) // Size of the magic number (8 bytes)
#define AES_BLOCK_SIZE kCCBlockSizeAES128
#define AES_KEY_SIZE kCCKeySizeAES256
- (NSData *)encryptData:(NSData *)data withKey:(NSString *)key {
    if (!data || !key) return nil;

    uint64_t magic = MAGIC_NUMBER;
    uint64_t originalLength = data.length;

    NSMutableData *plaintext = [NSMutableData data];
    [plaintext appendBytes:&magic length:sizeof(magic)];
    [plaintext appendBytes:&originalLength length:sizeof(originalLength)];
    [plaintext appendData:data];

    char keyPtr[AES_KEY_SIZE];
    bzero(keyPtr, sizeof(keyPtr));
    if (![key getCString:keyPtr maxLength:sizeof(keyPtr) encoding:NSUTF8StringEncoding]) return nil;

    uint8_t iv[AES_BLOCK_SIZE];
    if (SecRandomCopyBytes(kSecRandomDefault, sizeof(iv), iv) != errSecSuccess) return nil;

    size_t bufferSize = plaintext.length + AES_BLOCK_SIZE;
    void *buffer = malloc(bufferSize);
    if (!buffer) return nil;

    size_t numBytesEncrypted = 0;
    CCCryptorStatus status = CCCrypt(kCCEncrypt,
                                     kCCAlgorithmAES,
                                     kCCOptionPKCS7Padding,
                                     keyPtr,
                                     AES_KEY_SIZE,
                                     iv,
                                     plaintext.bytes,
                                     plaintext.length,
                                     buffer,
                                     bufferSize,
                                     &numBytesEncrypted);

    if (status != kCCSuccess) {
        free(buffer);
        return nil;
    }

    NSMutableData *final = [NSMutableData dataWithBytes:iv length:sizeof(iv)];
    [final appendBytes:buffer length:numBytesEncrypted];
    free(buffer);
    return final;
}

- (NSString *)jsonStringFromDictionary:(NSDictionary *)dictionary {
    NSError *error;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:dictionary options:0 error:&error];
    if (!error) {
        return [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
    } else {
        return nil;
    }
}

- (uint64_t)getFreeDiskSpace {
    NSError *error = nil;
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfFileSystemForPath:NSHomeDirectory() error:&error];

    if (error) return 0;

    uint64_t freeSpace = [[attributes objectForKey:NSFileSystemFreeSize] unsignedLongLongValue];
    return freeSpace;
}

- (BOOL)ensureFreeDiskSpace {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    NSInteger lastDeletedDayIndex = [defaults integerForKey:@"LastDeletedDayIndex"];
    NSInteger lastDeletedSegmentIndex = [defaults integerForKey:@"LastDeletedSegmentIndex"];

    @try {
        if ((double)[[[[NSFileManager defaultManager] attributesOfFileSystemForPath:NSHomeDirectory() error:nil] objectForKey:NSFileSystemFreeSize] unsignedLongLongValue] / (1024.0 * 1024.0) < MIN_FREE_SPACE_MB) {
            // Fetch all DayEntities, sorted by date
            NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"DayEntity"];
            fetchRequest.sortDescriptors = @[[NSSortDescriptor sortDescriptorWithKey:@"date" ascending:YES]];

            NSError *fetchError = nil;
            NSArray *dayEntities = [self.fileServer.context executeFetchRequest:fetchRequest error:&fetchError];

            if (fetchError) return YES;

            if (dayEntities.count == 0 || lastDeletedDayIndex >= dayEntities.count) {
                [defaults removeObjectForKey:@"LastDeletedDayIndex"];
                [defaults removeObjectForKey:@"LastDeletedSegmentIndex"];
                [defaults synchronize];
                return YES;
            }

            // Fetch event timestamps from Core Data
            NSArray *eventDataArray = [self.fileServer fetchEventDataFromCoreData:self.fileServer.context];
            NSMutableArray<NSNumber *> *eventTimestamps = [NSMutableArray array];
            NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
            [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];

            NSCalendar *calendar = [NSCalendar currentCalendar];
            for (NSDictionary *eventData in eventDataArray) {
                NSString *timestampString = eventData[@"timeStamp"];
                NSDate *eventDate = [dateFormatter dateFromString:timestampString];

                if (eventDate) {
                    NSDateComponents *components = [calendar components:(NSCalendarUnitHour | NSCalendarUnitMinute | NSCalendarUnitSecond) fromDate:eventDate];
                    double eventSecondsSinceMidnight = (components.hour * 3600) + (components.minute * 60) + components.second;
                    [eventTimestamps addObject:@(eventSecondsSinceMidnight)];
                }
            }

            BOOL deletedFile = NO;
            NSManagedObjectContext *context = self.fileServer.context;

            for (NSInteger dayIndex = lastDeletedDayIndex; dayIndex < dayEntities.count; dayIndex++) {
                NSManagedObject *dayEntity = dayEntities[dayIndex];
                NSArray *segments = [[dayEntity valueForKey:@"segments"] allObjects];

                if (lastDeletedSegmentIndex >= segments.count) {
                    lastDeletedSegmentIndex = 0; // Reset segment index if we move to the next day
                }

                // Collect segments to delete in a separate array to avoid modifying the collection during iteration
                NSMutableArray<NSManagedObject *> *segmentsToDelete = [NSMutableArray array];

                // Inside the dayIndex loop
                for (NSInteger segmentIndex = lastDeletedSegmentIndex; segmentIndex < segments.count; segmentIndex++) {
                    NSManagedObject *segmentEntity = segments[segmentIndex];
                    NSString *segmentURL = [segmentEntity valueForKey:@"url"];

                    NSNumber *segmentTimeStampNumber = [segmentEntity valueForKey:@"timeStamp"];
                    if (!segmentTimeStampNumber) {
                        continue;
                    }

                    double segmentSecondsSinceMidnight = segmentTimeStampNumber.doubleValue;
                    BOOL shouldSkipDeletion = NO;
                    for (NSNumber *eventTimestamp in eventTimestamps) {
                        double eventSecondsSinceMidnight = eventTimestamp.doubleValue;
                        double timeDifference = fabs(segmentSecondsSinceMidnight - eventSecondsSinceMidnight);
                        if (timeDifference <= 60) { // Within one minute
                            shouldSkipDeletion = YES;
                            break;
                        }
                    }

                    if (shouldSkipDeletion) {
                        continue;
                    }

                    // Mark this segment for deletion
                    [segmentsToDelete addObject:segmentEntity];
                }

                // Delete the collected segments and their files
                for (NSManagedObject *segmentEntity in segmentsToDelete) {
                    NSString *segmentURL = [segmentEntity valueForKey:@"url"]; // Declare segmentURL here
                    @try {
                        NSString *filePath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:segmentURL];

                        NSError *deleteError = nil;
                        if ([[NSFileManager defaultManager] fileExistsAtPath:filePath]) {
                            if ([[NSFileManager defaultManager] removeItemAtPath:filePath error:&deleteError]) {
                                // Delete the segmentEntity from Core Data
                                [context deleteObject:segmentEntity];

                                deletedFile = YES;
                                if ((double)[[[[NSFileManager defaultManager] attributesOfFileSystemForPath:NSHomeDirectory() error:nil] objectForKey:NSFileSystemFreeSize] unsignedLongLongValue] / (1024.0 * 1024.0) >= MIN_FREE_SPACE_MB + 500) {
                                    // Save the context before returning
                                    NSError *saveError = nil;
                                    if ([context save:&saveError]) {
                                        [defaults setInteger:dayIndex forKey:@"LastDeletedDayIndex"];
                                        [defaults setInteger:lastDeletedSegmentIndex forKey:@"LastDeletedSegmentIndex"];
                                        [defaults synchronize];
                                        return YES;
                                    }
                                }
                            }
                        }
                    } @catch (NSException *exception) {
                        NSLog(@"Exception while deleting file: %@, reason: %@", segmentURL, exception.reason); // segmentURL is now in scope
                    }
                }

                // Save the context after processing each day
                NSError *saveError = nil;
                if ([context save:&saveError]) {
                    [defaults setInteger:dayIndex forKey:@"LastDeletedDayIndex"];
                    [defaults setInteger:lastDeletedSegmentIndex forKey:@"LastDeletedSegmentIndex"];
                    [defaults synchronize];
                }

                // Reset segment index for the next day
                lastDeletedSegmentIndex = 0;
            }

            // If nothing was deleted, reset the indexes
            if (!deletedFile) {
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
            if ([[NSDate date] timeIntervalSince1970] - self.last_check_time > 10.0) {
                NSLog(@"Making request at %.2f %.2f seconds", [[NSDate date] timeIntervalSince1970],self.last_check_time);
                self.last_check_time = [[NSDate date] timeIntervalSince1970];
                NSString *deviceName = [[NSUserDefaults standardUserDefaults] stringForKey:@"device_name"];
                NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
                
                NSLog(@"session token = %@", sessionToken);

                NSString *encodedDeviceName = [deviceName stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];
                NSString *encodedSessionToken = [sessionToken stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];
                NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/get_stream_upload_link"];
                components.queryItems = @[
                    [NSURLQueryItem queryItemWithName:@"name" value:encodedDeviceName],
                    [NSURLQueryItem queryItemWithName:@"session_token" value:encodedSessionToken]
                ];

                NSURL *url = components.URL;
                NSLog(@"Final URL: %@", url);

                NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithURL:url
                                                                         completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
                    if (error) {
                        NSLog(@"❌ Error: %@", error.localizedDescription);
                    } else {
                        NSError *jsonError;
                        NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
                        
                        if (jsonError) {
                            NSLog(@"⚠️ JSON Parsing Error: %@", jsonError.localizedDescription);
                        } else {
                            NSString *uploadLink = json[@"upload_link"];
                            if ((uploadLink && ![uploadLink isKindOfClass:[NSNull class]])) {
                                if(self.isStreaming == NO){
                                    NSLog(@"📤 Upload link received: %@", uploadLink);
                                    //self.streamLink = uploadLink;
                                    self.isStreaming = YES;
                                    [FileServer sharedInstance].segment_length = 2;
                                    dispatch_async(dispatch_get_main_queue(), ^{
                                        [self refreshView];
                                    });
                                }
                            } else {
                                if(self.isStreaming){ //todo, messy
                                    self.isStreaming = NO;
                                    [FileServer sharedInstance].segment_length = 1;
                                    dispatch_async(dispatch_get_main_queue(), ^{
                                        [self refreshView];
                                    });
                                    NSLog(@"no link");
                                }
                            }
                        }
                    }
                }];
                [task resume];
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
            if(self.recordPressed) [self.scene processOutput:output withImage:ciImage orientation:self.previewLayer.connection.videoOrientation];
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
            [self.fpsLabel sizeToFit]; // Adjust size to fit new text
            self.fpsLabel.frame = CGRectMake(self.fpsLabel.frame.origin.x,
                                           self.fpsLabel.frame.origin.y,
                                           self.fpsLabel.frame.size.width + 8,
                                           self.fpsLabel.frame.size.height + 4); // Add padding
            self.frameCount = 0;
            self.lastFrameTime = currentTime;
        }
    } else {
        self.lastFrameTime = currentTime;
    }
}
@end
