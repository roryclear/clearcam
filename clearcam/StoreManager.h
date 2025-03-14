#import <StoreKit/StoreKit.h>

NS_ASSUME_NONNULL_BEGIN

// Define the notification name
extern NSString *const StoreManagerSubscriptionStatusDidChangeNotification;

@interface StoreManager : NSObject <SKProductsRequestDelegate, SKPaymentTransactionObserver>

+ (instancetype)sharedInstance;

- (void)fetchAndPurchaseProduct;
- (void)verifySubscriptionWithCompletion:(void (^)(BOOL isActive, NSDate * _Nullable expiryDate))completion;

@end

NS_ASSUME_NONNULL_END
