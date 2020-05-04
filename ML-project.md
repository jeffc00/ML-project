---
title: "ML Project"
author: "Jeff C"
date: "5/3/2020"
output:
  html_document:
    keep_md: yes
---



## Introduction
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and train a model to predict the manner in which the exercise was done.

## Model Building

```r
training <- read.csv("coursera/ml-project/pml-training.csv")
testing <- read.csv("coursera/ml-project/pml-testing.csv")
```

We start by appending together the training and testing datasets to make data cleaning easier. Also, a new column called `train` is created to help keep track of these datasets, which will later be split back again into training and testing components.  

```r
training$train <- 1
testing$train <- 0
classe <- factor(training$classe)
predictors <- rbind(training[, -160], testing[, -160])
dim(predictors)
```

```
## [1] 19642   160
```

Let's drop all the columns that have mostly missing values, since they don't add much information.

```r
predictors <- predictors[, colSums(is.na(predictors)) == 0]
dim(predictors)
```

```
## [1] 19642    60
```

Next we drop the columns 1 through 7, which have little or no relationship with the response variable.

```r
predictors <- predictors[, -c(1:7)]
dim(predictors)
```

```
## [1] 19642    53
```

Now that the data is clean, we split the `predictors` dataframe back into training and testing sets.

```r
training_full <- predictors[predictors$train == 1, -53]
testing <- predictors[predictors$train == 0, -53]
```

We are ready to build our model. Let's start by splitting the full training set into training and validation sets, using a 60/40 split and random seed to 42 in order to guarantee reproducibility of the results.

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
set.seed(42)
inTrain <- createDataPartition(classe, p = 0.6, list = FALSE)
training <- cbind(training_full, classe)[inTrain, ]
validation <- cbind(training_full, classe)[-inTrain, ]
```

### Multinomial Logistic Regression
Since the response variable is categorical and has multiple levels, a good place to start is by fitting a multinomial logistic regression model. Since this is a rather simple linear model, we expect it to have a higher bias but lower variance.

```r
m <- train(classe ~ .,
           method = "multinom",
           data = training,
           preProc = c("center", "scale"),
           trace = FALSE,
           allowParallel = TRUE,
           trControl = trainControl(method = "boot", number = 10))
```

In order to look for signs of over or underfitting, we compare the model's accuracy on the training set and on the validation set.  

* Training set confusion matrix:

```r
confusionMatrix(training$classe, predict(m, training[, -53]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2857   86  202  168   35
##          B  281 1474  227   73  224
##          C  208  174 1456  125   91
##          D  135   80  224 1389  102
##          E   97  289  153  187 1439
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7316          
##                  95% CI : (0.7235, 0.7396)
##     No Information Rate : 0.3038          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6598          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7985   0.7009   0.6437   0.7152   0.7610
## Specificity            0.9401   0.9168   0.9371   0.9450   0.9266
## Pos Pred Value         0.8533   0.6468   0.7089   0.7197   0.6647
## Neg Pred Value         0.9145   0.9338   0.9171   0.9438   0.9530
## Prevalence             0.3038   0.1786   0.1921   0.1649   0.1606
## Detection Rate         0.2426   0.1252   0.1236   0.1180   0.1222
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8693   0.8088   0.7904   0.8301   0.8438
```

* Validation set confusion matrix:

```r
confusionMatrix(validation$classe, predict(m, validation[, -53]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1919   67  138   91   17
##          B  205  979  139   47  148
##          C  122  127  984   85   50
##          D   88   56  142  933   67
##          E   63  182   91  102 1004
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7417          
##                  95% CI : (0.7318, 0.7513)
##     No Information Rate : 0.3055          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6724          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8006   0.6938   0.6586   0.7417   0.7807
## Specificity            0.9426   0.9162   0.9395   0.9464   0.9332
## Pos Pred Value         0.8598   0.6449   0.7193   0.7255   0.6963
## Neg Pred Value         0.9149   0.9317   0.9213   0.9505   0.9560
## Prevalence             0.3055   0.1798   0.1904   0.1603   0.1639
## Detection Rate         0.2446   0.1248   0.1254   0.1189   0.1280
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8716   0.8050   0.7991   0.8440   0.8570
```

### Random Forest
As we can see above, both accuracy scores are quite close but maybe we can do better with a nonlinear model. Let's try to fit a Random Forest model this time.

```r
m <- train(classe ~ .,
           method = "rf",
           data = training,
           preProc = c("center", "scale"),
           trace = FALSE,
           allowParallel = TRUE,
           trControl = trainControl(method = "boot", number = 10))
```

* Training set confusion matrix:

```r
confusionMatrix(training$classe, predict(m, training[, -53]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

* Validation set confusion matrix:

```r
confusionMatrix(validation$classe, predict(m, validation[, -53]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2226    4    2    0    0
##          B   13 1504    1    0    0
##          C    0    9 1353    6    0
##          D    0    0   11 1271    4
##          E    0    0    2    1 1439
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9912, 0.9949)
##     No Information Rate : 0.2854          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9915          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9942   0.9914   0.9883   0.9945   0.9972
## Specificity            0.9989   0.9978   0.9977   0.9977   0.9995
## Pos Pred Value         0.9973   0.9908   0.9890   0.9883   0.9979
## Neg Pred Value         0.9977   0.9979   0.9975   0.9989   0.9994
## Prevalence             0.2854   0.1933   0.1745   0.1629   0.1839
## Detection Rate         0.2837   0.1917   0.1724   0.1620   0.1834
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9966   0.9946   0.9930   0.9961   0.9984
```

# Conclusion
As we can see from the results above, the Random Forest model is far superior than the simple Multinomial Logistic Regression, having much better fit and accuracy. Hence we shall keep that as our final model.
