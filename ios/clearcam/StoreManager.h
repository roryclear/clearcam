#import <StoreKit/StoreKit.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

// Define the notification name
extern NSString *const StoreManagerSubscriptionStatusDidChangeNotification;

@interface StoreManager : NSObject <SKProductsRequestDelegate, SKPaymentTransactionObserver>
@property (nonatomic, assign) NSTimeInterval last_check_time;
@property (nonatomic, strong) SKProduct *premiumProduct;

+ (instancetype)sharedInstance;
- (void)verifySubscriptionWithCompletion:(void (^)(BOOL isActive, NSDate * _Nullable expiryDate))completion;
- (void)verifySubscriptionWithCompletionIfSubbed:(void (^)(BOOL isActive, NSDate * _Nullable expiryDate))completion;
- (void)verifySessionOnlyWithCompletion:(void (^)(BOOL isActive, NSDate *expiryDate))completion;
- (void)showUpgradePopupInViewController:(UIViewController *)presentingVC
                               darkMode:(BOOL)isDarkMode
                             completion:(void (^)(BOOL success))completion;
- (void)fetchAndPurchaseProductWithCompletion:(void (^)(BOOL success, NSError * _Nullable error))completion;
- (void)restorePurchasesWithCompletion:(void (^)(BOOL success))completion;
- (void)getPremiumProductInfo:(void (^)(SKProduct * _Nullable product, NSError * _Nullable error))completion;
- (NSString *)retrieveSessionTokenFromKeychain;
- (void)storeSessionTokenInKeychain:(NSString *)sessionToken;
- (void)clearSessionTokenFromKeychain;
- (void)showUpgradePopupInViewController:(UIViewController *)presentingVC;
- (void)showUpgradePopupInViewController:(UIViewController *)presentingVC
                             completion:(void (^)(BOOL success))completion;

@end

NS_ASSUME_NONNULL_END
