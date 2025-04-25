#import "DeviceStreamViewController.h"
#import "StoreManager.h" // For session token
#import <AVFoundation/AVFoundation.h>
#import <Security/Security.h>
#import <CommonCrypto/CommonCryptor.h>

@interface DeviceStreamViewController ()
@property (nonatomic, strong) AVQueuePlayer *player;
@property (nonatomic, strong) AVPlayerLayer *playerLayer;
@property (nonatomic, strong) NSTimer *downloadTimer;
@property (nonatomic, strong) NSData *lastSegmentData;
@property (nonatomic, assign) NSUInteger segmentIndex;
@property (nonatomic, strong) NSString *downloadLink;
@end

@implementation DeviceStreamViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor blackColor];

    self.player = [AVQueuePlayer queuePlayerWithItems:@[]];
    self.player.volume = 1.0;

    self.playerLayer = [AVPlayerLayer playerLayerWithPlayer:self.player];
    self.playerLayer.frame = self.view.bounds;
    self.playerLayer.videoGravity = AVLayerVideoGravityResizeAspect;
    [self.view.layer addSublayer:self.playerLayer];

    [self.player play];
    [self fetchDownloadLinkAndStartStreaming];

    NSLog(@"device streaming from = %@", self.deviceName);
}

- (void)fetchDownloadLinkAndStartStreaming {
    NSString *deviceName = self.deviceName;
    NSString *sessionToken = [[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];

    if (!deviceName || !sessionToken) {
        NSLog(@"❌ Missing device name or session token");
        return;
    }

    NSString *encodedDeviceName = [deviceName stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];
    NSString *encodedSessionToken = [sessionToken stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]];

    NSURLComponents *components = [NSURLComponents componentsWithString:@"https://rors.ai/get_stream_download_link"];
    components.queryItems = @[
        [NSURLQueryItem queryItemWithName:@"name" value:encodedDeviceName],
        [NSURLQueryItem queryItemWithName:@"session_token" value:encodedSessionToken]
    ];

    NSURL *url = components.URL;

    NSURLSessionDataTask *linkTask = [[NSURLSession sharedSession] dataTaskWithURL:url completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"❌ Error fetching stream link: %@", error.localizedDescription);
            return;
        }


        NSError *jsonError;
        NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError) {
            NSLog(@"⚠️ JSON Parsing Error: %@", jsonError.localizedDescription);
            return;
        }

        NSString *downloadLink = json[@"download_link"];
        if (![downloadLink isKindOfClass:[NSString class]] || downloadLink.length == 0) {
            NSLog(@"🚫 Invalid downloadLink link");
            return;
        } else {
            self.downloadLink = downloadLink;
        }

        NSLog(@"✅ Got download link: %@", self.downloadLink);

        dispatch_async(dispatch_get_main_queue(), ^{
            [self downloadAndQueueSegment];
            [self startDownloadTimer];
        });
    }];
    [linkTask resume];
}

- (void)startDownloadTimer {
    self.downloadTimer = [NSTimer scheduledTimerWithTimeInterval:1.0
                                                          target:self
                                                        selector:@selector(downloadAndQueueSegment)
                                                        userInfo:nil
                                                         repeats:YES];
}

- (void)downloadAndQueueSegment {
    NSString *urlString = self.downloadLink;
    NSURL *remoteURL = [NSURL URLWithString:urlString];

    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithURL:remoteURL
                                                             completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error || !data) {
            NSLog(@"❌ Download error: %@", error.localizedDescription);
            return;
        }

        if ([self.lastSegmentData isEqualToData:data]) {
            NSLog(@"🔁 Same segment, skipping");
            return;
        }

        self.lastSegmentData = data;
        data = [self decryptData:data withKey:@"open_please"];
        NSString *fileName = [NSString stringWithFormat:@"segment_%lu.mp4", (unsigned long)(self.segmentIndex++ % 2)];
        NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:fileName];
        NSURL *localURL = [NSURL fileURLWithPath:tempPath];

        if ([data writeToURL:localURL atomically:YES]) {
            NSLog(@"✅ New segment saved: %@", fileName);
            dispatch_async(dispatch_get_main_queue(), ^{
                AVURLAsset *asset = [AVURLAsset assetWithURL:localURL];
                AVPlayerItem *item = [AVPlayerItem playerItemWithAsset:asset];
                [self.player insertItem:item afterItem:nil];
            });
        } else {
            NSLog(@"❌ Write failed");
        }
    }];
    [task resume];
}

#define MAGIC_NUMBER 0x4D41474943ULL // "MAGIC" in ASCII as a 64-bit value
#define HEADER_SIZE (sizeof(uint64_t)) // Size of the magic number (8 bytes)
#define AES_BLOCK_SIZE kCCBlockSizeAES128
#define AES_KEY_SIZE kCCKeySizeAES256
- (NSData *)decryptData:(NSData *)encryptedDataWithIv withKey:(NSString *)key {
    if (!encryptedDataWithIv || !key) return nil;
    if (encryptedDataWithIv.length <= AES_BLOCK_SIZE) return nil;

    NSData *ivData = [encryptedDataWithIv subdataWithRange:NSMakeRange(0, AES_BLOCK_SIZE)];
    NSData *cipherData = [encryptedDataWithIv subdataWithRange:NSMakeRange(AES_BLOCK_SIZE, encryptedDataWithIv.length - AES_BLOCK_SIZE)];

    char keyPtr[AES_KEY_SIZE + 1];
    bzero(keyPtr, sizeof(keyPtr));
    if (![key getCString:keyPtr maxLength:sizeof(keyPtr) encoding:NSUTF8StringEncoding]) return nil;

    size_t bufferSize = cipherData.length + AES_BLOCK_SIZE;
    void *buffer = malloc(bufferSize);
    if (!buffer) return nil;

    size_t numBytesDecrypted = 0;
    CCCryptorStatus status = CCCrypt(kCCDecrypt,
                                     kCCAlgorithmAES,
                                     kCCOptionPKCS7Padding,
                                     keyPtr,
                                     AES_KEY_SIZE,
                                     ivData.bytes,
                                     cipherData.bytes,
                                     cipherData.length,
                                     buffer,
                                     bufferSize,
                                     &numBytesDecrypted);

    if (status != kCCSuccess) {
        free(buffer);
        return nil;
    }

    if (numBytesDecrypted < sizeof(uint64_t) * 2) {
        free(buffer);
        return nil;
    }

    uint64_t magic;
    uint64_t originalLength;
    memcpy(&magic, buffer, sizeof(uint64_t));
    memcpy(&originalLength, buffer + sizeof(uint64_t), sizeof(uint64_t));

    if (magic != MAGIC_NUMBER || originalLength > (numBytesDecrypted - sizeof(uint64_t) * 2)) {
        free(buffer);
        return nil;
    }

    NSData *result = [NSData dataWithBytes:(buffer + sizeof(uint64_t) * 2) length:originalLength];
    free(buffer);
    return result;
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    self.playerLayer.frame = self.view.bounds;
}

- (void)dealloc {
    [self.downloadTimer invalidate];
}

@end
