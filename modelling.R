###
#
# MLAA Assignment 2 Part A
#
# Students:
# Ivan Cheung - 13975420
# Ryan Yeo - 1234
# Dinh Tran -
# John Rho - 24509337
#
#
#
#
###

## Load Data - Ivan
## Data Cleaning - Ivan



### Model Analysis

## Model 1 Dinh

## Model 2 Ivan

## Model 3 John
##train logistic regression model 
#transform default values to no=0 and yes= 1
trainset$default<-as.factor(trainset$default)
levels(trainset$default)<-c("0","1")
testset$default<-as.factor(testset$default)
levels(testset$default)<-c("0","1")

AT2.glm = glm(formula = default~ .,
              data = trainset,
              family = "binomial")
summary(AT2.glm )

#Try training model with variables with significant p value 
AT2.glm_sig = glm(formula = default~ LIMIT_BAL+MARRIAGE+AGE+PAY_0 +PAY_2+PAY_AMT3+NO_PAY_DELAY,
                  data = trainset,
                  family = "binomial")
summary(AT2.glm_sig)

#we get a better AIC score from model wtih all variables. Meaning that AT2.glm is fitting the data better than AT2.glm_sig

testset$probability = predict(AT2.glm, newdata = testset, type = "response")
testset$prediction = "0"
testset[testset$probability >= 0.5, "prediction"] = "1"

table(testset$prediction)
#confusion matrix 
AT2_cfm <- table(predicted=testset$prediction,true=testset$default)
AT2_cfm

#Accuracy
accuracy <- (AT2_cfm[1,1]+AT2_cfm[2,2])/sum(AT2_cfm)
accuracy

#Precision 
precision <- AT2_cfm[1,1]/(AT2_cfm[1,1]+AT2_cfm[1,2])
precision

#Recall
recall <- AT2_cfm[1,1]/(AT2_cfm[1,1]+AT2_cfm[2,1])
recall

#F1 
f1 <- 2*(precision*recall/(precision+recall))
f1

#AUC 
library(pROC)
roc_object <- roc(testset$default,testset$probability)
auc(roc_object)


## Model 4 Ryan

### Model Evaluations

## Confusion Matrix
## AUC of each model
## ROC

### Model Optimisations

## Final Model - GBM

## Final Model - Random Forest

## Produce validation output

##
