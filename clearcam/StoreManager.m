#import "StoreManager.h"

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
                break;
                
            case SKPaymentTransactionStateFailed:
                NSLog(@"Purchase failed: %@", transaction.error.localizedDescription);
                [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
                break;
                
            case SKPaymentTransactionStateRestored:
                NSLog(@"Purchase restored.");
                [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
                break;
                
            default:
                break;
        }
    }
}

@end

