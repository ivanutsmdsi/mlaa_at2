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
#train logistic regression model 
AT2.glm = glm(formula = default~ .,
              data = train,
              family = "binomial")
summary(AT2.glm )

test$probability = predict(AT2.glm, newdata = test, type = "response")
test$prediction = "0"
test[test$probability >= 0.5, "prediction"] = "1"

table(test$prediction)

#confusion matrix 
AT2_cfm <- table(predicted=test$prediction,true=test$default)
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
roc_object <- roc( test$default, test$probability)
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
