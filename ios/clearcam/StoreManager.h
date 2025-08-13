#import <StoreKit/StoreKit.h>

NS_ASSUME_NONNULL_BEGIN

// Define the notification name
extern NSString *const StoreManagerSubscriptionStatusDidChangeNotification;

@interface StoreManager : NSObject <SKProductsRequestDelegate, SKPaymentTransactionObserver>
@property (nonatomic, assign) NSTimeInterval last_check_time;

+ (instancetype)sharedInstance;

- (void)fetchAndPurchaseProductWithCompletion:(void (^)(BOOL success, NSError * _Nullable error))completion;
- (void)verifySubscriptionWithCompletion:(void (^)(BOOL isActive, NSDate * _Nullable expiryDate))completion;
- (void)verifySubscriptionWithCompletionIfSubbed:(void (^)(BOOL isActive, NSDate * _Nullable expiryDate))completion;
- (NSString *)retrieveSessionTokenFromKeychain;
- (void)getPremiumProductInfo:(void (^)(SKProduct * _Nullable product, NSError * _Nullable error))completion;
- (void)restorePurchases;
- (void)storeSessionTokenInKeychain:(NSString *)sessionToken;
- (void)clearSessionTokenFromKeychain;

@end

NS_ASSUME_NONNULL_END
