#import "StoreManager.h"
#import "FileServer.h"

// Define the notification name
NSString *const StoreManagerSubscriptionStatusDidChangeNotification = @"StoreManagerSubscriptionStatusDidChangeNotification";

@interface StoreManager ()
@property (nonatomic, strong) SKProductsRequest *productsRequest;
@property (nonatomic, strong) SKProduct *premiumProduct;
@end

@implementation StoreManager

+ (instancetype)sharedInstance {
    static StoreManager *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        [[SKPaymentQueue defaultQueue] addTransactionObserver:self];
    }
    return self;
}

- (void)fetchAndPurchaseProduct {
    NSSet *productIdentifiers = [NSSet setWithObject:@"monthly.premium"];
    self.productsRequest = [[SKProductsRequest alloc] initWithProductIdentifiers:productIdentifiers];
    self.productsRequest.delegate = self;
    [self.productsRequest start];
}

#pragma mark - SKProductsRequestDelegate

- (void)productsRequest:(SKProductsRequest *)request didReceiveResponse:(SKProductsResponse *)response {
    if (response.products.count == 0) return;
    self.premiumProduct = response.products.firstObject;
    [self purchaseProduct];
}

#pragma mark - Purchase Product

- (void)purchaseProduct {
    if ([SKPaymentQueue canMakePayments] && self.premiumProduct) {
        SKPayment *payment = [SKPayment paymentWithProduct:self.premiumProduct];
        [[SKPaymentQueue defaultQueue] addPayment:payment];
    }
}

#pragma mark - SKPaymentTransactionObserver

- (void)paymentQueue:(SKPaymentQueue *)queue updatedTransactions:(NSArray<SKPaymentTransaction *> *)transactions {
    for (SKPaymentTransaction *transaction in transactions) {
        switch (transaction.transactionState) {
            case SKPaymentTransactionStatePurchased:
                [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
                // After a successful purchase, verify the subscription status
                [self verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
                    if (isActive) {
                        // Post the notification to inform listeners of the status change
                        [[NSNotificationCenter defaultCenter] postNotificationName:StoreManagerSubscriptionStatusDidChangeNotification object:nil];
                    }
                }];
                break;
                
            case SKPaymentTransactionStateFailed:
                [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
                break;
                
            case SKPaymentTransactionStateRestored:
                [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
                // After restoring purchases, verify the subscription status
                [self verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
                    if (isActive) {
                        // Post the notification to inform listeners of the status change
                        [[NSNotificationCenter defaultCenter] postNotificationName:StoreManagerSubscriptionStatusDidChangeNotification object:nil];
                    }
                }];
                break;
                
            default:
                break;
        }
    }
}

- (void)verifySubscriptionWithCompletion:(void (^)(BOOL isActive, NSDate *expiryDate))completion {
    static BOOL isRequestInProgress = NO;

    if (isRequestInProgress) return;

    isRequestInProgress = YES;  // Mark request as in progress

    NSString *storedReceipt = [[NSUserDefaults standardUserDefaults] stringForKey:@"subscriptionReceipt"];
    
    if (!storedReceipt) {
        NSURL *receiptURL = [[NSBundle mainBundle] appStoreReceiptURL];
        NSData *receiptData = [NSData dataWithContentsOfURL:receiptURL];

        if (!receiptData) {
            [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"isSubscribed"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            isRequestInProgress = NO;  // Reset flag
            completion(NO, nil);
            return;
        }

        storedReceipt = [receiptData base64EncodedStringWithOptions:0];

        [[NSUserDefaults standardUserDefaults] setObject:storedReceipt forKey:@"subscriptionReceipt"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }

    NSDictionary *requestDict = @{@"receipt": storedReceipt};
    NSError *error;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:requestDict options:0 error:&error];

    if (error) {
        [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"isSubscribed"];
        [[NSUserDefaults standardUserDefaults] synchronize];
        isRequestInProgress = NO;  // Reset flag
        completion(NO, nil);
        return;
    }

    [FileServer performPostRequestWithURL:@"https://rors.ai/verify_receipt"
                                       method:@"POST"
                                  contentType:@"application/json"
                                         body:jsonData
                            completionHandler:^(NSData *data, NSHTTPURLResponse *response, NSError *error) {
        isRequestInProgress = NO;
        if (error) {
            [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"isSubscribed"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            completion(NO, nil);
            return;
        }
        NSDictionary *jsonResponse = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if (!jsonResponse || ![jsonResponse isKindOfClass:[NSDictionary class]]) {
            [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"isSubscribed"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            completion(NO, nil);
            return;
        }
        BOOL isSubscribed = [jsonResponse[@"valid"] boolValue];
        if (isSubscribed) {
            NSString *sessionToken = jsonResponse[@"session_token"];
            if (sessionToken && [sessionToken isKindOfClass:[NSString class]]) {
                [self storeSessionTokenInKeychain:sessionToken];
                [[NSUserDefaults standardUserDefaults] setObject:[NSDate dateWithTimeIntervalSinceNow:45 * 60] forKey:@"expiry"];
            }
            [[NSNotificationCenter defaultCenter] postNotificationName:StoreManagerSubscriptionStatusDidChangeNotification object:nil];
        } else {
            [self clearSessionTokenFromKeychain];
        }
        [[NSUserDefaults standardUserDefaults] setBool:isSubscribed forKey:@"isSubscribed"];
        [[NSUserDefaults standardUserDefaults] synchronize];
        completion(isSubscribed, nil);
    }];
}


- (void)storeSessionTokenInKeychain:(NSString *)sessionToken {
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: @"com.clearcam.session",
        (__bridge id)kSecAttrAccount: @"sessionToken",
        (__bridge id)kSecValueData: [sessionToken dataUsingEncoding:NSUTF8StringEncoding]
    };

    // Delete any existing token first
    SecItemDelete((__bridge CFDictionaryRef)query);
    // Add the new token
    OSStatus status = SecItemAdd((__bridge CFDictionaryRef)query, NULL);
}

- (void)clearSessionTokenFromKeychain {
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: @"com.clearcam.session",
        (__bridge id)kSecAttrAccount: @"sessionToken"
    };
    SecItemDelete((__bridge CFDictionaryRef)query);}

// Optional: Retrieve the session token if needed later
- (NSString *)retrieveSessionTokenFromKeychain {
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: @"com.clearcam.session",
        (__bridge id)kSecAttrAccount: @"sessionToken",
        (__bridge id)kSecReturnData: @YES,
        (__bridge id)kSecMatchLimit: (__bridge id)kSecMatchLimitOne
    };

    CFTypeRef dataRef = NULL;
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, &dataRef);
    if (status == errSecSuccess) {
        NSData *data = (__bridge_transfer NSData *)dataRef;
        return [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    }
    return nil;
}

@end
