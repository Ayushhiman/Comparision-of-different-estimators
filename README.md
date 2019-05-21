Prediction models have been developed for the loan dataset, to determine whether a client will pay loan or not.
The estimators compared were LoR,KNN,SVM, Decision trees and random forrests.
As all the features had quiet some correlation with target variable feature selection has not been done(rfecv was tried and the results did not improve).
For KNN the ideal amount of neighbours was calculated.
For SVM grid parameter search was done to get the best pair of parameters C and gamma.
For random forrests the number of estimators used were 600.
All the models were used to predct on test data. It was found that all of them has similar results, and they were quiet unsuccessful in prediciting whe the loan was not fully paid.
