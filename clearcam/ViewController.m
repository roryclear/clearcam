#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <Metal/Metal.h>
#import "AppDelegate.h"
#import <CoreData/CoreData.h>
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
@property (nonatomic, assign) NSDate *current_file_timestamp;
@property (nonatomic, strong) NSMutableDictionary *digits;
@property (nonatomic, strong) NSMutableArray *current_segment_squares;
@property (nonatomic, strong) NSLock *segmentLock;
@property (nonatomic, strong) NSManagedObjectContext *context;

#define MIN_FREE_SPACE_MB 200  //threshold to start deleting

@end

@implementation ViewController

NSMutableDictionary *classColorMap;

- (void)viewDidLoad {
    [super viewDidLoad];
    
    // coredata stuff
    AppDelegate *appDelegate = (AppDelegate *)[[UIApplication sharedApplication] delegate];
    self.context = appDelegate.persistentContainer.viewContext;

    // Save a new string
    NSLog(@"core data segments?:");
    //[self deleteAllSegmentsInContext:self.context]; //to delete all
    // Fetch all saved strings
    [self fetchAllSegmentsInContext:self.context];
    // coredata stuff
        
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
        if ([file hasPrefix:@"opp"]) continue;
        if ([file hasPrefix:@"batch_req"]) continue;
        if ([file hasPrefix:@"2025-01-27"]) continue;

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

- (void)saveURL:(NSString *)string inContext:(NSManagedObjectContext *)context {
    // Create a new StringEntity object
    NSManagedObject *newString = [NSEntityDescription insertNewObjectForEntityForName:@"SegmentEntity" inManagedObjectContext:context];
    
    // Set the value attribute
    [newString setValue:string forKey:@"url"];
    
    // Save the context
    NSError *error = nil;
    if (![context save:&error]) {
        NSLog(@"Failed to save string: %@", error.localizedDescription);
    } else {
        NSLog(@"String saved successfully");
    }
}

- (void)fetchAllSegmentsInContext:(NSManagedObjectContext *)context {
    // Create a fetch request for SegmentEntity
    NSFetchRequest *fetchRequest = [[NSFetchRequest alloc] initWithEntityName:@"SegmentEntity"];
    
    // Execute the fetch request
    NSError *error = nil;
    NSArray *results = [context executeFetchRequest:fetchRequest error:&error];
    
    if (error) {
        NSLog(@"Failed to fetch segments: %@", error.localizedDescription);
    } else {
        NSLog(@"Fetched segments:");
        for (NSManagedObject *obj in results) {
            NSString *url = [obj valueForKey:@"url"];
            double timeStamp = [[obj valueForKey:@"timeStamp"] doubleValue];
            double duration = [[obj valueForKey:@"duration"] doubleValue];
            
            NSLog(@"Segment - URL: %@, TimeStamp: %f, Duration: %f", url, timeStamp, duration);
        }
    }
}

- (void)deleteAllSegmentsInContext:(NSManagedObjectContext *)context {
    // Create a fetch request for all SegmentEntity objects
    NSFetchRequest *fetchRequest = [NSFetchRequest fetchRequestWithEntityName:@"SegmentEntity"];
    
    // Fetch all SegmentEntity objects
    NSError *fetchError = nil;
    NSArray *segments = [context executeFetchRequest:fetchRequest error:&fetchError];
    
    if (fetchError) {
        NSLog(@"Failed to fetch SegmentEntity objects: %@", fetchError.localizedDescription);
        return;
    }
    
    // Loop through all fetched objects and delete them
    for (NSManagedObject *segment in segments) {
        [context deleteObject:segment];
    }
    
    // Save the context after deletion
    NSError *saveError = nil;
    if (![context save:&saveError]) {
        NSLog(@"Failed to delete objects: %@", saveError.localizedDescription);
    } else {
        NSLog(@"All segments deleted successfully");
    }
}

- (void)setupCamera {
    self.captureSession = [[AVCaptureSession alloc] init];
    self.captureSession.sessionPreset = AVCaptureSessionPreset1920x1080;
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
    [self ensureFreeDiskSpace];
    NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
    [formatter setDateFormat:@"yyyy-MM-dd"];
    NSString *dayFolderName = [formatter stringFromDate:[NSDate date]];
    
    [formatter setDateFormat:@"yyyy-MM-dd_HH:mm:ss:SSS"];
    self.current_file_timestamp = [NSDate date];
    NSString *timestamp = [formatter stringFromDate:self.current_file_timestamp];
    NSString *segNumberString = [NSString stringWithFormat:@"_%05ld_", (long)self.seg_number];
    self.seg_number += 1;
    NSString *finalTimestamp = [NSString stringWithFormat:@"%@%@%@",
                                [[timestamp componentsSeparatedByString:@"_"] firstObject],
                                segNumberString,
                                [[timestamp componentsSeparatedByString:@"_"] lastObject]];
    
    // Create a folder for the day within the documents directory
    NSURL *documentsDirectory = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    NSURL *dayFolderURL = [documentsDirectory URLByAppendingPathComponent:dayFolderName];
    
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
    
    NSDictionary *videoSettings = @{
        AVVideoCodecKey: AVVideoCodecTypeH264,
        AVVideoWidthKey: @1280,
        AVVideoHeightKey: @720,
        AVVideoScalingModeKey: AVVideoScalingModeResizeAspectFill
    };
    self.videoWriterInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:videoSettings];
    self.videoWriterInput.expectsMediaDataInRealTime = YES;
    if(self.previewLayer.connection.videoOrientation == AVCaptureVideoOrientationPortrait){
        self.videoWriterInput.transform = CGAffineTransformMakeRotation(M_PI_2);//does this work?
    } else if (self.previewLayer.connection.videoOrientation == AVCaptureVideoOrientationLandscapeLeft) {
        self.videoWriterInput.transform = CGAffineTransformMakeRotation(M_PI);
    }
    
    
    NSDictionary *sourcePixelBufferAttributes = @{
        (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (NSString *)kCVPixelBufferWidthKey: @1280,
        (NSString *)kCVPixelBufferHeightKey: @720
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

        AVAsset *asset = [AVAsset assetWithURL:self.assetWriter.outputURL];
        CMTime time = asset.duration;

        NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
        [dateFormatter setDateFormat:@"yyyy-MM-dd"];
        NSString *dateFolderName = [dateFormatter stringFromDate:self.current_file_timestamp];

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

        NSCalendar *calendar = [NSCalendar currentCalendar];
        NSDateComponents *components = [calendar components:NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay | NSCalendarUnitHour fromDate:self.current_file_timestamp];
        NSString *dateKey = [NSString stringWithFormat:@"%04ld-%02ld-%02ld", (long)components.year, (long)components.month, (long)components.day];

        if (!self.fileServer.segmentsDict[dateKey]) {
            NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
            NSString *segmentsDirectory = [documentsDirectory stringByAppendingPathComponent:dateKey];
            NSString *segmentsFilePath = [segmentsDirectory stringByAppendingPathComponent:@"segments.txt"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:segmentsFilePath]) {
                NSData *data = [NSData dataWithContentsOfFile:segmentsFilePath];
                NSArray *fileSegments = [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:nil];
                if (fileSegments) {
                    NSLog(@"found segments %lu",(unsigned long)fileSegments.count);
                    self.fileServer.segmentsDict[dateKey] = [fileSegments mutableCopy];
                }
            }
            if(!self.fileServer.segmentsDict[dateKey]) self.fileServer.segmentsDict[dateKey] = [[NSMutableArray alloc] init];
        }


        NSMutableDictionary *segmentEntry = [NSMutableDictionary dictionaryWithDictionary:@{ //does this fix last segment in hour/day?
            @"url": [NSString stringWithFormat:@"%@/%@", [[self.assetWriter.outputURL URLByDeletingLastPathComponent] lastPathComponent], self.assetWriter.outputURL.lastPathComponent],
            @"duration": @(CMTimeGetSeconds(time)),
            @"frames":[self.current_segment_squares copy]
        }];
                    
        //timestamp of every segment now
        calendar = [NSCalendar currentCalendar];
        components = [calendar components:NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay fromDate:self.current_file_timestamp];
        NSDate *midnight = [calendar dateFromComponents:components];
        NSTimeInterval timeStamp = [self.current_file_timestamp timeIntervalSinceDate:midnight];
        segmentEntry[@"timeStamp"] = @(timeStamp);
            
            
        //new coredata way!
        [self saveURL:[NSString stringWithFormat:@"%@/%@", [[self.assetWriter.outputURL URLByDeletingLastPathComponent] lastPathComponent], self.assetWriter.outputURL.lastPathComponent] inContext:self.context];
            
        NSManagedObject *newString = [NSEntityDescription insertNewObjectForEntityForName:@"SegmentEntity" inManagedObjectContext:self.context];
        
        // Set the value attribute
        [newString setValue:[NSString stringWithFormat:@"%@/%@", [[self.assetWriter.outputURL URLByDeletingLastPathComponent] lastPathComponent], self.assetWriter.outputURL.lastPathComponent] forKey:@"url"];
        [newString setValue:@(timeStamp) forKey:@"timeStamp"];
        [newString setValue:@(CMTimeGetSeconds(time)) forKey:@"duration"];
        // Save the context
        if (![self.context save:&error]) {
            NSLog(@"Failed to save string: %@", error.localizedDescription);
        } else {
            NSLog(@"String saved successfully");
        }
        //core data
            

        [self.fileServer.segmentsDict[dateKey] addObject:segmentEntry];
        
        [self startNewRecording];
        NSString *segmentsFilePath = [segmentsDirectory stringByAppendingPathComponent:@"segments.txt"];
        [self saveSegmentEntry:segmentEntry toFile:segmentsFilePath];
        [self.segmentLock lock];
        [self.current_segment_squares removeAllObjects];
        [self.segmentLock unlock];
        }];
    } else {
        NSLog(@"Cannot finish writing. Asset writer status: %ld", (long)self.assetWriter.status);
    }
}

- (void)saveSegmentEntry:(NSDictionary *)segmentEntry toFile:(NSString *)filePath {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSError *error = nil;

    // Check if file exists
    if (![fileManager fileExistsAtPath:filePath]) {
        // Create a new file with the initial JSON array
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:@[segmentEntry]
                                                           options:NSJSONWritingPrettyPrinted
                                                             error:&error];
        if (!error) {
            BOOL success = [jsonData writeToFile:filePath atomically:YES];
            if (!success) {
                NSLog(@"Failed to write JSON to file.");
            }
        } else {
            NSLog(@"Error writing JSON to file: %@", error.localizedDescription);
        }
    } else {
        // Open the file for updating
        NSFileHandle *fileHandle = [NSFileHandle fileHandleForUpdatingAtPath:filePath];
        if (!fileHandle) {
            NSLog(@"Error opening file for writing.");
            return;
        }
        
        @try {
            // Seek to the end of the file minus 1 byte (to remove the last ']')
            [fileHandle seekToEndOfFile];
            unsigned long long fileLength = [fileHandle offsetInFile];
            if (fileLength < 1) {
                NSLog(@"File is too short to be valid JSON.");
                return;
            }
            [fileHandle seekToFileOffset:fileLength - 1];
            
            // Read the last character to ensure it's a ']'
            NSData *lastCharData = [fileHandle readDataOfLength:1];
            NSString *lastChar = [[NSString alloc] initWithData:lastCharData encoding:NSUTF8StringEncoding];
            if (![lastChar isEqualToString:@"]"]) {
                NSLog(@"File does not end with a valid JSON array closing bracket.");
                return;
            }
            
            // Truncate the file to remove the last ']'
            [fileHandle truncateFileAtOffset:fileLength - 1];
            
            // Create a comma to separate the new JSON object
            NSString *commaString = @",";
            NSData *commaData = [commaString dataUsingEncoding:NSUTF8StringEncoding];
            
            // Write the comma to the file
            [fileHandle writeData:commaData];
            
            // Serialize the new JSON object
            NSData *newJsonData = [NSJSONSerialization dataWithJSONObject:segmentEntry options:0 error:&error];
            if (!newJsonData || error) {
                NSLog(@"Error serializing new JSON: %@", error.localizedDescription);
                return;
            }
            
            // Write the new JSON object to the file
            [fileHandle writeData:newJsonData];
            
            // Insert closing bracket at the end of file
            NSString *closingBracketString = @"]";
            NSData *closingBracketData = [closingBracketString dataUsingEncoding:NSUTF8StringEncoding];
            [fileHandle writeData:closingBracketData];
        }
        @finally {
            [fileHandle closeFile];
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

- (void)ensureFreeDiskSpace {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), ^{
        double freeSpace = (double)[[[[NSFileManager defaultManager] attributesOfFileSystemForPath:NSHomeDirectory() error:nil] objectForKey:NSFileSystemFreeSize] unsignedLongLongValue] / (1024.0 * 1024.0 * 1024.0);
        NSLog(@"Current free space: %.2f GB", freeSpace);
    });
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
            if (self.fileServer.segment_length == 1 && [[NSDate now] timeIntervalSinceDate:self.fileServer.last_req_time] > 60){
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
            
            NSMutableDictionary *frame = [[NSMutableDictionary alloc] init];//todo, init elsewhere
            NSMutableDictionary *frameSquare = [[NSMutableDictionary alloc] init];;
            NSMutableArray *frameSquares = [[NSMutableArray alloc] init];;
            

            NSCalendar *calendar = [NSCalendar currentCalendar];
            NSDate *now = [NSDate date];
            NSDateComponents *components = [calendar components:NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay fromDate:now];
            NSDate *midnight = [calendar dateFromComponents:components];
            NSTimeInterval timeStamp = [now timeIntervalSinceDate:midnight];
            
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
            CGImageRelease(cgImage);
            
            __weak typeof(self) weak_self = self;
            dispatch_async(dispatch_get_main_queue(), ^{
                @try {
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

                    // Ensure thread safety and check for nil
                    @synchronized(weak_self) {
                        if (frame) {
                            [self.segmentLock lock];
                            [weak_self.current_segment_squares addObject:frame];
                            [self.segmentLock unlock];
                        } else {
                            NSLog(@"Warning: Attempted to insert nil frame into array.");
                        }
                    }

                    [weak_self updateFPS];
                    weak_self.isProcessing = NO;
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

- (CVPixelBufferRef)addTimeStampToPixelBuffer:(CVPixelBufferRef)pixelBuffer{
    NSInteger pixelSize = 6;
    NSInteger spaceSize = 3;
    NSInteger digitOriginX = spaceSize;
    NSInteger digitOriginY = spaceSize;
    NSInteger height = spaceSize*2 + pixelSize*5;

    NSDate *currentDate = [NSDate date];
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
    NSString *timestamp = [dateFormatter stringFromDate:currentDate];
        
    
    //check roatation
    CGAffineTransform transform = self.videoWriterInput.transform;
    CGFloat angle = atan2(transform.b, transform.a);

    // Convert radians to degrees
    CGFloat degrees = angle * (180.0 / M_PI);
    
    size_t width_res = CVPixelBufferGetWidth(pixelBuffer);
    size_t height_res = CVPixelBufferGetHeight(pixelBuffer);
    if (degrees == 90 || degrees == -90) {
        pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer withColor:[UIColor blackColor] originX:0 originY:height_res-(pixelSize*3+spaceSize)*timestamp.length width:height height:(pixelSize*3+spaceSize)*timestamp.length opacity:0.4];
    } else if (degrees == 180 || degrees == -180) {
        pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer withColor:[UIColor blackColor] originX:width_res-(pixelSize*3+spaceSize)*timestamp.length originY:height_res-height width:(pixelSize*3+spaceSize)*timestamp.length height:height opacity:0.4];
    } else {
        //no rotation
        pixelBuffer = [self addColoredRectangleToPixelBuffer:pixelBuffer withColor:[UIColor blackColor] originX:0 originY:0 width:(pixelSize*3+spaceSize)*timestamp.length height:height opacity:0.4];
    }
    
    
    for (NSUInteger k = 0; k < [timestamp length]; k++) {
        unichar character = [timestamp characterAtIndex:k];
        if(character == ' '){
            digitOriginX += pixelSize*3;
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
                                                            originX: width_res - (digitOriginX + [self.digits[key][i][0] doubleValue] * pixelSize) - ([self.digits[key][i][2] doubleValue] * pixelSize)
                                                            originY: height_res - (digitOriginY + [self.digits[key][i][1] doubleValue] * pixelSize) - ([self.digits[key][i][3] doubleValue] * pixelSize)
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

