//
//  StoreManager.h
//  clearcam
//
//  Created by Rory Clear on 13/03/2025.
//


#import <Foundation/Foundation.h>
#import <StoreKit/StoreKit.h>

@interface StoreManager : NSObject <SKProductsRequestDelegate, SKPaymentTransactionObserver>

+ (instancetype)sharedInstance;
- (void)fetchAndPurchaseProduct;
- (void)verifySubscriptionWithCompletion:(void (^)(BOOL isActive, NSDate *expiryDate))completion;

@end
