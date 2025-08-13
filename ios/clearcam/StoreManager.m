#import "StoreManager.h"
#import "FileServer.h"
#import <StoreKit/StoreKit.h>

// Define the notification name
NSString *const StoreManagerSubscriptionStatusDidChangeNotification = @"StoreManagerSubscriptionStatusDidChangeNotification";

@interface StoreManager ()
@property (nonatomic, strong) SKProductsRequest *productsRequest;
@property (nonatomic, strong) SKProduct *premiumProduct;
@property (nonatomic, strong) void (^productInfoCompletionHandler)(SKProduct * _Nullable product, NSError * _Nullable error); // Store completion handler for fetching info
@property (nonatomic, copy) void (^purchaseCompletionHandler)(BOOL success, NSError * _Nullable error);
// Implement SKPaymentTransactionObserver method to handle transaction updates
- (void)getPremiumProductInfo:(void (^)(SKProduct * _Nullable product, NSError * _Nullable error))completion;
@end

@implementation StoreManager

- (void)restorePurchases {
    [[SKPaymentQueue defaultQueue] restoreCompletedTransactions];
}

+ (StoreManager *)sharedInstance {
    static StoreManager *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[StoreManager alloc] init];
        [[SKPaymentQueue defaultQueue] addTransactionObserver:sharedInstance];
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

- (void)fetchAndPurchaseProductWithCompletion:(void (^)(BOOL success, NSError * _Nullable error))completion {
    [self getPremiumProductInfo:^(SKProduct * _Nullable product, NSError * _Nullable error) {
        if (!product) {
            if (completion) {
                completion(NO, error ?: [NSError errorWithDomain:@"StoreManager" code:-1 userInfo:@{NSLocalizedDescriptionKey: @"Failed to fetch product information"}]);
            }
            return;
        }
        
        // Initiate the purchase
        SKPayment *payment = [SKPayment paymentWithProduct:product];
        [[SKPaymentQueue defaultQueue] addPayment:payment];
        
        // Store the completion handler to call it when the transaction finishes
        self.purchaseCompletionHandler = completion;
    }];
}

- (void)getPremiumProductInfo:(void (^)(SKProduct * _Nullable product, NSError * _Nullable error))completion {
    if (self.premiumProduct) {
        if (completion) {
            completion(self.premiumProduct, nil);
        }
        return;
    }
    
    // Store the completion handler
    self.productInfoCompletionHandler = completion;
    
    // Fetch product info from App Store
    NSSet *productIdentifiers = [NSSet setWithObject:@"monthly.premium"]; // Your product ID
    self.productsRequest = [[SKProductsRequest alloc] initWithProductIdentifiers:productIdentifiers];
    self.productsRequest.delegate = self; // Delegate methods will handle the response
    [self.productsRequest start];
}

#pragma mark - SKProductsRequestDelegate

- (void)productsRequest:(SKProductsRequest *)request didReceiveResponse:(SKProductsResponse *)response {
    NSLog(@"Received product response.");
    if (response.products.count > 0) {
        self.premiumProduct = response.products.firstObject;
        if (self.productInfoCompletionHandler) {
            dispatch_async(dispatch_get_main_queue(), ^{
                self.productInfoCompletionHandler(self.premiumProduct, nil);
                self.productInfoCompletionHandler = nil;
            });
        }
    } else {
        NSError *error = [NSError errorWithDomain:@"StoreManagerError" code:101 userInfo:@{NSLocalizedDescriptionKey: @"No products found."}];
        if (self.productInfoCompletionHandler) {
            dispatch_async(dispatch_get_main_queue(), ^{
                self.productInfoCompletionHandler(nil, error);
                self.productInfoCompletionHandler = nil;
            });
        }
    }
    self.productsRequest.delegate = nil;
    self.productsRequest = nil;
}

- (void)request:(SKRequest *)request didFailWithError:(NSError *)error {
    if (self.productInfoCompletionHandler) {
        dispatch_async(dispatch_get_main_queue(), ^{
            self.productInfoCompletionHandler(nil, error);
            self.productInfoCompletionHandler = nil; // Clear the handler
        });
    }
    self.productsRequest.delegate = nil;
    self.productsRequest = nil;
}

#pragma mark - Purchase Product

- (void)purchaseProduct:(SKProduct *)product {
    if ([SKPaymentQueue canMakePayments] && product) {
        SKPayment *payment = [SKPayment paymentWithProduct:product];
        [[SKPaymentQueue defaultQueue] addPayment:payment];
    }
}

#pragma mark - SKPaymentTransactionObserver

- (void)paymentQueue:(SKPaymentQueue *)queue updatedTransactions:(NSArray<SKPaymentTransaction *> *)transactions {
    for (SKPaymentTransaction *transaction in transactions) {
        switch (transaction.transactionState) {
            case SKPaymentTransactionStatePurchased: {
                [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
                [self verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
                    if (isActive) {
                        [[NSNotificationCenter defaultCenter] postNotificationName:StoreManagerSubscriptionStatusDidChangeNotification object:nil];
                        if (self.purchaseCompletionHandler) {
                            self.purchaseCompletionHandler(YES, nil);
                            self.purchaseCompletionHandler = nil;
                        }
                    } else if (self.purchaseCompletionHandler) {
                        self.purchaseCompletionHandler(NO, [NSError errorWithDomain:@"StoreManager" code:-1 userInfo:@{NSLocalizedDescriptionKey: @"Subscription verification failed"}]);
                        self.purchaseCompletionHandler = nil;
                    }
                }];
                break;
            }
            case SKPaymentTransactionStateFailed: {
                [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
                if (self.purchaseCompletionHandler) {
                    self.purchaseCompletionHandler(NO, transaction.error);
                    self.purchaseCompletionHandler = nil;
                }
                if (transaction.error.code != SKErrorPaymentCancelled) {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Purchase Failed"
                                                                                       message:transaction.error.localizedDescription
                                                                                preferredStyle:UIAlertControllerStyleAlert];
                        [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
                        UIViewController *topController = [UIApplication sharedApplication].keyWindow.rootViewController;
                        while (topController.presentedViewController) {
                            topController = topController.presentedViewController;
                        }
                        [topController presentViewController:alert animated:YES completion:nil];
                    });
                }
                break;
            }
            case SKPaymentTransactionStateRestored: {
                [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
                [self verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        if (isActive) {
                            [[NSNotificationCenter defaultCenter] postNotificationName:StoreManagerSubscriptionStatusDidChangeNotification object:nil];
                            UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Purchases Restored"
                                                                                           message:@"Your previous purchases have been successfully restored."
                                                                                    preferredStyle:UIAlertControllerStyleAlert];
                            [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
                            UIViewController *topController = [UIApplication sharedApplication].keyWindow.rootViewController;
                            while (topController.presentedViewController) {
                                topController = topController.presentedViewController;
                            }
                            [topController presentViewController:alert animated:YES completion:nil];
                            if (self.purchaseCompletionHandler) {
                                self.purchaseCompletionHandler(YES, nil);
                                self.purchaseCompletionHandler = nil;
                            }
                        } else {
                            UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"No Purchases Found"
                                                                                           message:@"No previous purchases were found to restore."
                                                                                    preferredStyle:UIAlertControllerStyleAlert];
                            [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
                            UIViewController *topController = [UIApplication sharedApplication].keyWindow.rootViewController;
                            while (topController.presentedViewController) {
                                topController = topController.presentedViewController;
                            }
                            [topController presentViewController:alert animated:YES completion:nil];
                            if (self.purchaseCompletionHandler) {
                                self.purchaseCompletionHandler(NO, [NSError errorWithDomain:@"StoreManager" code:-1 userInfo:@{NSLocalizedDescriptionKey: @"No valid subscriptions found"}]);
                                self.purchaseCompletionHandler = nil;
                            }
                        }
                    });
                }];
                break;
            }
            case SKPaymentTransactionStatePurchasing:
                // No action needed while purchasing
                break;
            default:
                break;
        }
    }
}

- (void)paymentQueueRestoreCompletedTransactionsFinished:(SKPaymentQueue *)queue {
    NSLog(@"Restore completed transactions finished.");
}

- (void)paymentQueue:(SKPaymentQueue *)queue restoreCompletedTransactionsFailedWithError:(NSError *)error {
    dispatch_async(dispatch_get_main_queue(), ^{
        UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Restore Failed"
                                                                       message:error.localizedDescription
                                                                preferredStyle:UIAlertControllerStyleAlert];
        [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
        UIViewController *topController = [UIApplication sharedApplication].keyWindow.rootViewController;
        while (topController.presentedViewController) {
            topController = topController.presentedViewController;
        }
        [topController presentViewController:alert animated:YES completion:nil];
    });
}

- (void)verifySubscriptionWithCompletionIfSubbed:(void (^)(BOOL isActive, NSDate *expiryDate))completion {
    NSDate *expiry = [[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"];
    BOOL isSubscribed = [[NSUserDefaults standardUserDefaults] boolForKey:@"isSubscribed"];
    NSTimeInterval lastCheck = [StoreManager sharedInstance].last_check_time;
    BOOL sessionExpired = !expiry || [expiry compare:[NSDate date]] != NSOrderedDescending;
    BOOL sessionTokenMissing = ![[StoreManager sharedInstance] retrieveSessionTokenFromKeychain];
    BOOL shouldCheckRemote = sessionExpired || sessionTokenMissing || ([[NSDate date] timeIntervalSince1970] - lastCheck > 120.0);

    if (shouldCheckRemote) {
        [self verifySubscriptionWithCompletion:^(BOOL isActive, NSDate *expiryDate) {
            if (completion) completion(isActive, expiryDate);
        }];
    } else {
        if (completion) completion(isSubscribed, expiry);
    }
}

- (void)verifySubscriptionWithCompletion:(void (^)(BOOL isActive, NSDate *expiryDate))completion {
    self.last_check_time = [[NSDate date] timeIntervalSince1970];
    static BOOL isRequestInProgress = NO;

    if (isRequestInProgress) return;
    isRequestInProgress = YES;
    NSString *sessionToken = [self retrieveSessionTokenFromKeychain];
    if (sessionToken && sessionToken.length > 0) {
        NSString *urlString = [NSString stringWithFormat:@"https://rors.ai/validate_user?session_token=%@", sessionToken];
        NSURL *url = [NSURL URLWithString:urlString];

        NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithURL:url
                                                                 completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
            NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
            if (!error && httpResponse.statusCode >= 200 && httpResponse.statusCode < 300) {
                [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"isSubscribed"];
                [[NSUserDefaults standardUserDefaults] synchronize];
                isRequestInProgress = NO;
                dispatch_async(dispatch_get_main_queue(), ^{
                    completion(YES, [[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"]);
                });
                return;
            }
            [self performReceiptVerificationWithCompletion:completion requestFlag:&isRequestInProgress];
        }];
        [task resume];
        return;
    }
    [self performReceiptVerificationWithCompletion:completion requestFlag:&isRequestInProgress];
}

- (void)performReceiptVerificationWithCompletion:(void (^)(BOOL isActive, NSDate *expiryDate))completion
                                      requestFlag:(BOOL *)flagPtr {
    NSString *storedReceipt = [[NSUserDefaults standardUserDefaults] stringForKey:@"subscriptionReceipt"];

    if (!storedReceipt) {
        NSURL *receiptURL = [[NSBundle mainBundle] appStoreReceiptURL];
        NSData *receiptData = [NSData dataWithContentsOfURL:receiptURL];

        if (!receiptData) {
            [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"isSubscribed"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            *flagPtr = NO;
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
        *flagPtr = NO;
        completion(NO, nil);
        return;
    }

    [FileServer performPostRequestWithURL:@"https://www.rors.ai/verify_receipt"
                                   method:@"POST"
                              contentType:@"application/json"
                                     body:jsonData
                        completionHandler:^(NSData *data, NSHTTPURLResponse *response, NSError *error) {
        *flagPtr = NO;
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
    SecItemDelete((__bridge CFDictionaryRef)query);
}

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
