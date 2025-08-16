#import "StoreManager.h"
#import "FileServer.h"
#import <StoreKit/StoreKit.h>
#import <objc/runtime.h>

// Define the notification name
NSString *const StoreManagerSubscriptionStatusDidChangeNotification = @"StoreManagerSubscriptionStatusDidChangeNotification";

@interface StoreManager ()
@property (nonatomic, strong) SKProductsRequest *productsRequest;
@property (nonatomic, strong) void (^productInfoCompletionHandler)(SKProduct * _Nullable product, NSError * _Nullable error);
@property (nonatomic, copy) void (^purchaseCompletionHandler)(BOOL success, NSError * _Nullable error);
@property (nonatomic, copy) void (^restoreCompletionHandler)(BOOL success);
@end

@implementation StoreManager

+ (instancetype)sharedInstance {
    static StoreManager *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[StoreManager alloc] init];
        [[SKPaymentQueue defaultQueue] addTransactionObserver:sharedInstance];
    });
    return sharedInstance;
}

#pragma mark - Upgrade Popup

- (void)showUpgradePopupInViewController:(UIViewController *)presentingVC
                               darkMode:(BOOL)isDarkMode
                             completion:(void (^)(BOOL success))completion {
    
    [self getPremiumProductInfo:^(SKProduct * _Nullable product, NSError * _Nullable error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (!product) {
                NSLog(@"Error fetching product info: %@", error.localizedDescription);
                if (completion) completion(NO);
                return;
            }

            // Format price
            NSNumberFormatter *formatter = [[NSNumberFormatter alloc] init];
            [formatter setNumberStyle:NSNumberFormatterCurrencyStyle];
            [formatter setLocale:product.priceLocale];
            NSString *localizedPrice = [formatter stringFromNumber:product.price] ?: NSLocalizedString(@"price_unknown", @"Fallback for unknown price");

            // Colors
            UIColor *cardBackground = isDarkMode ? [UIColor colorWithWhite:0.1 alpha:1.0] : [UIColor colorWithWhite:1.0 alpha:1.0];
            UIColor *textColor = isDarkMode ? UIColor.whiteColor : UIColor.blackColor;

            // Overlay
            UIView *overlay = [[UIView alloc] initWithFrame:presentingVC.view.bounds];
            overlay.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.7];
            overlay.tag = 999;
            [presentingVC.view addSubview:overlay];

            // Card
            UIView *card = [[UIView alloc] init];
            card.translatesAutoresizingMaskIntoConstraints = NO;
            card.backgroundColor = cardBackground;
            card.layer.cornerRadius = 20;
            card.clipsToBounds = YES;
            [overlay addSubview:card];

            // Title
            UILabel *title = [[UILabel alloc] init];
            title.translatesAutoresizingMaskIntoConstraints = NO;
            title.text = NSLocalizedString(@"get_premium_title", @"Title for premium upgrade popup");
            title.textColor = [UIColor colorWithRed:1.0 green:0.84 blue:0 alpha:1.0]; // gold
            title.font = [UIFont boldSystemFontOfSize:24];
            title.textAlignment = NSTextAlignmentCenter;

            // Features
            UIStackView *featureStack = [[UIStackView alloc] init];
            featureStack.translatesAutoresizingMaskIntoConstraints = NO;
            featureStack.axis = UILayoutConstraintAxisVertical;
            featureStack.spacing = 8;

            NSArray *features = @[
                NSLocalizedString(@"premium_feature_1", @"Feature 1 description for premium"),
                NSLocalizedString(@"premium_feature_2", @"Feature 2 description for premium"),
                NSLocalizedString(@"premium_feature_3", @"Feature 3 description for premium"),
                NSLocalizedString(@"premium_feature_4", @"Feature 4 description for premium")
            ];

            for (NSString *item in features) {
                UILabel *label = [[UILabel alloc] init];
                label.text = [NSString stringWithFormat:@"• %@", item];
                label.font = [UIFont systemFontOfSize:16 weight:UIFontWeightSemibold];
                label.textColor = textColor;
                label.numberOfLines = 0;
                [featureStack addArrangedSubview:label];
            }

            // Upgrade button
            UIButton *upgradeBtn = [UIButton buttonWithType:UIButtonTypeSystem];
            upgradeBtn.translatesAutoresizingMaskIntoConstraints = NO;

            NSString *line1 = NSLocalizedString(@"1_week_free_trial", @"Free trial label");
            NSString *line2 = [NSString stringWithFormat:NSLocalizedString(@"upgrade_button", @"Subscription price after trial"), localizedPrice];
            NSString *fullText = [NSString stringWithFormat:@"%@\n%@", line1, line2];

            NSMutableParagraphStyle *paragraphStyle = [[NSMutableParagraphStyle alloc] init];
            paragraphStyle.alignment = NSTextAlignmentCenter;

            NSMutableAttributedString *attributedTitle = [[NSMutableAttributedString alloc] initWithString:fullText attributes:@{
                NSParagraphStyleAttributeName: paragraphStyle,
                NSForegroundColorAttributeName: UIColor.blackColor
            }];

            NSRange line1Range = [fullText rangeOfString:line1];
            [attributedTitle addAttribute:NSFontAttributeName value:[UIFont boldSystemFontOfSize:17] range:line1Range];

            NSRange line2Range = [fullText rangeOfString:line2];
            [attributedTitle addAttribute:NSFontAttributeName value:[UIFont systemFontOfSize:14] range:line2Range];

            [upgradeBtn setAttributedTitle:attributedTitle forState:UIControlStateNormal];
            upgradeBtn.titleLabel.numberOfLines = 2;
            upgradeBtn.titleLabel.textAlignment = NSTextAlignmentCenter;
            upgradeBtn.backgroundColor = [UIColor colorWithRed:1.0 green:0.84 blue:0 alpha:1.0];
            upgradeBtn.layer.cornerRadius = 12;
            
            // Handle upgrade tap
            __weak typeof(self) weakSelf = self;
            [upgradeBtn addTarget:weakSelf action:@selector(handleUpgradeTap:) forControlEvents:UIControlEventTouchUpInside];
            objc_setAssociatedObject(upgradeBtn, @"completionBlock", completion, OBJC_ASSOCIATION_COPY_NONATOMIC);
            objc_setAssociatedObject(upgradeBtn, @"presentingVC", presentingVC, OBJC_ASSOCIATION_ASSIGN);
            objc_setAssociatedObject(upgradeBtn, @"overlayView", overlay, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

            // Cancel button
            UIButton *cancelBtn = [UIButton buttonWithType:UIButtonTypeSystem];
            cancelBtn.translatesAutoresizingMaskIntoConstraints = NO;
            [cancelBtn setTitle:NSLocalizedString(@"not_now", @"Not now button") forState:UIControlStateNormal];
            [cancelBtn setTitleColor:textColor forState:UIControlStateNormal];
            cancelBtn.titleLabel.font = [UIFont systemFontOfSize:15];
            [cancelBtn addTarget:self action:@selector(dismissUpgradePopup:) forControlEvents:UIControlEventTouchUpInside];
            objc_setAssociatedObject(cancelBtn, @"overlayView", overlay, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

            // Disclaimer label
            UILabel *disclaimer = [[UILabel alloc] init];
            disclaimer.translatesAutoresizingMaskIntoConstraints = NO;
            disclaimer.text = NSLocalizedString(@"premium_disclaimer", @"Disclaimer for premium subscription limits");
            disclaimer.font = [UIFont systemFontOfSize:13];
            disclaimer.textColor = [textColor colorWithAlphaComponent:0.6];
            disclaimer.numberOfLines = 0;
            disclaimer.textAlignment = NSTextAlignmentCenter;

            // Add all to card
            [card addSubview:title];
            [card addSubview:featureStack];
            [card addSubview:upgradeBtn];
            [card addSubview:cancelBtn];
            [card addSubview:disclaimer];

            // Constraints
            [NSLayoutConstraint activateConstraints:@[
                [card.centerXAnchor constraintEqualToAnchor:overlay.centerXAnchor],
                [card.centerYAnchor constraintEqualToAnchor:overlay.centerYAnchor],
                [card.widthAnchor constraintEqualToConstant:320],

                [title.topAnchor constraintEqualToAnchor:card.topAnchor constant:24],
                [title.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:20],
                [title.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-20],

                [featureStack.topAnchor constraintEqualToAnchor:title.bottomAnchor constant:20],
                [featureStack.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:20],
                [featureStack.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-20],

                [upgradeBtn.topAnchor constraintEqualToAnchor:featureStack.bottomAnchor constant:24],
                [upgradeBtn.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:20],
                [upgradeBtn.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-20],
                [upgradeBtn.heightAnchor constraintEqualToConstant:48],

                [cancelBtn.topAnchor constraintEqualToAnchor:upgradeBtn.bottomAnchor constant:16],
                [cancelBtn.centerXAnchor constraintEqualToAnchor:card.centerXAnchor],

                [disclaimer.topAnchor constraintEqualToAnchor:cancelBtn.bottomAnchor constant:14],
                [disclaimer.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:20],
                [disclaimer.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-20],
                [disclaimer.bottomAnchor constraintEqualToAnchor:card.bottomAnchor constant:-20]
            ]];
        });
    }];
}

- (void)showUpgradePopupInViewController:(UIViewController *)presentingVC
                             completion:(void (^)(BOOL success))completion {
    [self showUpgradePopupInViewController:presentingVC
                                 darkMode:(presentingVC.traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark)
                               completion:completion];
}

- (void)handleUpgradeTap:(UIButton *)sender {
    void (^completion)(BOOL) = objc_getAssociatedObject(sender, @"completionBlock");
    UIViewController *presentingVC = objc_getAssociatedObject(sender, @"presentingVC");
    UIView *overlay = objc_getAssociatedObject(sender, @"overlayView");
    
    // Create and show a loading spinner
    UIActivityIndicatorView *spinner = [[UIActivityIndicatorView alloc] initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleLarge];
    spinner.translatesAutoresizingMaskIntoConstraints = NO;
    spinner.color = UIColor.whiteColor;
    [overlay addSubview:spinner];
    
    // Center the spinner in the overlay
    [NSLayoutConstraint activateConstraints:@[
        [spinner.centerXAnchor constraintEqualToAnchor:overlay.centerXAnchor],
        [spinner.centerYAnchor constraintEqualToAnchor:overlay.centerYAnchor]
    ]];
    
    [spinner startAnimating];
    
    // Disable user interaction on the overlay to prevent multiple taps
    overlay.userInteractionEnabled = NO;
    
    // Fetch and initiate the purchase
    [self fetchAndPurchaseProductWithCompletion:^(BOOL success, NSError * _Nullable error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            // Stop the spinner and re-enable interaction
            [spinner stopAnimating];
            [spinner removeFromSuperview];
            overlay.userInteractionEnabled = YES;
            
            // Dismiss the upgrade popup
            [overlay removeFromSuperview];
            
            if (!success && error) {
                // Show error alert if purchase failed
                UIAlertController *alert = [UIAlertController alertControllerWithTitle:NSLocalizedString(@"purchase_failed", @"Title for purchase failed alert")
                                                                               message:error.localizedDescription
                                                                        preferredStyle:UIAlertControllerStyleAlert];
                [alert addAction:[UIAlertAction actionWithTitle:NSLocalizedString(@"ok", @"OK button") style:UIAlertActionStyleDefault handler:nil]];
                [presentingVC presentViewController:alert animated:YES completion:nil];
            }
            
            if (completion) {
                completion(success);
            }
        });
    }];
}

- (void)dismissUpgradePopup:(UIButton *)sender {
    UIView *overlay = objc_getAssociatedObject(sender, @"overlayView");
    [overlay removeFromSuperview];
    
    void (^completion)(BOOL) = objc_getAssociatedObject(sender, @"completionBlock");
    if (completion) {
        completion(NO);
    }
}

- (void)restorePurchasesWithCompletion:(void (^)(BOOL success))completion {
    self.restoreCompletionHandler = completion;
    [[SKPaymentQueue defaultQueue] restoreCompletedTransactions];
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
                            if (self.restoreCompletionHandler) {
                                self.restoreCompletionHandler(YES);
                                self.restoreCompletionHandler = nil;
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
                            if (self.restoreCompletionHandler) {
                                self.restoreCompletionHandler(NO);
                                self.restoreCompletionHandler = nil;
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


- (void)restorePurchases {
    [[SKPaymentQueue defaultQueue] restoreCompletedTransactions];
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

- (void)verifySessionOnlyWithCompletion:(void (^)(BOOL isActive, NSDate *expiryDate))completion {
    [self verifySessionOrSubscription:NO completion:completion];
}

- (void)verifySubscriptionWithCompletion:(void (^)(BOOL isActive, NSDate *expiryDate))completion {
    [self verifySessionOrSubscription:YES completion:completion];
}

- (void)verifySessionOrSubscription:(BOOL)checkSubscription
                         completion:(void (^)(BOOL isActive, NSDate *expiryDate))completion
{
    if (checkSubscription) {
        self.last_check_time = [[NSDate date] timeIntervalSince1970];
    }
    
    static BOOL isRequestInProgress = NO;
    if (checkSubscription && isRequestInProgress) {
        return;
    }
    if (checkSubscription) {
        isRequestInProgress = YES;
    }
    
    NSString *sessionToken = [self retrieveSessionTokenFromKeychain];
    
    if (!sessionToken || sessionToken.length == 0) {
        if (checkSubscription) {
            [self performReceiptVerificationWithCompletion:completion requestFlag:&isRequestInProgress];
        } else {
            completion(NO, nil);
        }
        return;
    }
    
    NSString *urlString = [NSString stringWithFormat:@"https://rors.ai/validate_user?session_token=%@", sessionToken];
    NSURL *url = [NSURL URLWithString:urlString];
    
    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithURL:url
                                                             completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
        
        if (!error && httpResponse.statusCode >= 200 && httpResponse.statusCode < 300) {
            if (checkSubscription) {
                [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"isSubscribed"];
                [[NSUserDefaults standardUserDefaults] synchronize];
                isRequestInProgress = NO;
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(YES, [[NSUserDefaults standardUserDefaults] objectForKey:@"expiry"]);
            });
        } else {
            if (checkSubscription) {
                [self performReceiptVerificationWithCompletion:completion requestFlag:&isRequestInProgress];
            } else {
                dispatch_async(dispatch_get_main_queue(), ^{
                    completion(NO, nil);
                });
            }
        }
    }];
    
    [task resume];
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
