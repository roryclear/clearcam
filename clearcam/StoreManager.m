#import "StoreManager.h"

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
    NSLog(@"✅ Received response from App Store.");

    if (response.invalidProductIdentifiers.count > 0) {
        NSLog(@"🚨 Invalid product IDs: %@", response.invalidProductIdentifiers);
    }

    if (response.products.count == 0) {
        NSLog(@"🚨 No valid products found! Double-check subscription status in App Store Connect.");
        return;
    }

    // Print details of the subscription
    for (SKProduct *product in response.products) {
        NSLog(@"✅ Found product: %@", product.productIdentifier);
        NSLog(@"Title: %@", product.localizedTitle);
        NSLog(@"Description: %@", product.localizedDescription);
        NSLog(@"Price: %@", product.price);
    }

    self.premiumProduct = response.products.firstObject;
    [self purchaseProduct];
}

- (void)request:(SKRequest *)request didFailWithError:(NSError *)error {
    NSLog(@"Failed to fetch products: %@", error.localizedDescription);
}

#pragma mark - Purchase Product

- (void)purchaseProduct {
    if ([SKPaymentQueue canMakePayments] && self.premiumProduct) {
        SKPayment *payment = [SKPayment paymentWithProduct:self.premiumProduct];
        [[SKPaymentQueue defaultQueue] addPayment:payment];
    } else {
        NSLog(@"Purchases are disabled on this device.");
    }
}

#pragma mark - SKPaymentTransactionObserver

- (void)paymentQueue:(SKPaymentQueue *)queue updatedTransactions:(NSArray<SKPaymentTransaction *> *)transactions {
    for (SKPaymentTransaction *transaction in transactions) {
        switch (transaction.transactionState) {
            case SKPaymentTransactionStatePurchased:
                NSLog(@"Purchase successful!");
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
                NSLog(@"Purchase failed: %@", transaction.error.localizedDescription);
                [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
                break;
                
            case SKPaymentTransactionStateRestored:
                NSLog(@"Purchase restored.");
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
    // Check if we have a stored receipt
    NSString *storedReceipt = [[NSUserDefaults standardUserDefaults] stringForKey:@"subscriptionReceipt"];
    
    if (storedReceipt) {
        NSLog(@"📜 Using cached receipt.");
    } else {
        NSLog(@"🔄 Fetching new receipt from App Store.");
        NSURL *receiptURL = [[NSBundle mainBundle] appStoreReceiptURL];
        NSData *receiptData = [NSData dataWithContentsOfURL:receiptURL];

        if (!receiptData) {
            NSLog(@"No receipt found. User may not have purchased any subscriptions.");
            [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"isSubscribed"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            completion(NO, nil);
            return;
        }

        storedReceipt = [receiptData base64EncodedStringWithOptions:0];

        // Save the receipt in NSUserDefaults
        [[NSUserDefaults standardUserDefaults] setObject:storedReceipt forKey:@"subscriptionReceipt"];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }

    // Prepare the request to the new verification endpoint
    NSDictionary *requestDict = @{@"receipt": storedReceipt};
    NSError *error;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:requestDict options:0 error:&error];

    if (error) {
        NSLog(@"JSON serialization error: %@", error.localizedDescription);
        [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"isSubscribed"];
        [[NSUserDefaults standardUserDefaults] synchronize];
        completion(NO, nil);
        return;
    }

    NSURL *url = [NSURL URLWithString:@"https://rors.ai/verify_receipt"];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    request.HTTPMethod = @"POST";
    request.HTTPBody = jsonData;
    [request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];

    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"Error verifying receipt: %@", error.localizedDescription);
            [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"isSubscribed"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            completion(NO, nil);
            return;
        }

        NSDictionary *jsonResponse = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        NSLog(@"Response from server = %@", jsonResponse);

        if (!jsonResponse || ![jsonResponse isKindOfClass:[NSDictionary class]]) {
            NSLog(@"Invalid response from server.");
            [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"isSubscribed"];
            [[NSUserDefaults standardUserDefaults] synchronize];
            completion(NO, nil);
            return;
        }

        BOOL isSubscribed = [jsonResponse[@"valid"] boolValue];

        [[NSUserDefaults standardUserDefaults] setBool:isSubscribed forKey:@"isSubscribed"];
        [[NSUserDefaults standardUserDefaults] synchronize];

        if (isSubscribed) {
            NSLog(@"✅ Subscription is active.");
            [[NSNotificationCenter defaultCenter] postNotificationName:StoreManagerSubscriptionStatusDidChangeNotification object:nil];
        } else {
            NSLog(@"🚨 Subscription has expired.");
        }

        completion(isSubscribed, nil); // No expiry date since the server doesn't return it
    }];

    [task resume];
}

@end
