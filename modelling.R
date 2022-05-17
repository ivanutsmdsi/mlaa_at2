###
#
# MLAA Assignment 2 Part A ####
#
# Students:
# Ivan Cheung - 13975420
# Ryan Yeo - 14328254
# Dinh Tran -
# John Rho - 24509337
#
###
#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Libraries ####
#install.packages("AMR")
library(dplyr)
library(caret)
library(gbm)
library(parallel)
library(magrittr)
library(AMR)
library(ROCR)
library(ggplot2)

library(e1071)

## Load Data - Ivan

rm(list = ls())
df <- read.csv('AT2_credit_train.csv')
df_raw <- df

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# EDA ####

str(df) # check shape
df$ID <- NULL # drop ID

## Dependent variable ####
# Check target var <- default
sum(is.na(df)) # check missing value - null
unique(df$default) # check target var
df$default[df$default == "Y"] <- 1 # converting to match data dictionary
df$default[df$default == "N"] <- 0 # converting to match data dictionary
df$default <- as.integer(df$default) # set as integer
df$default <- as.factor(df$default) # set as factor
levels(df$default) # check factor, 1 = yes
tb_default <- table(df$default) # check target distribution
tb_default # 0:1 = 16974:6127

# checking the ratio
tbl_prop <- prop.table(tb_default)
tbl_prop # we have unbalanced data. approx 0:1 = 3:1

## Independent variables ####
# check other independent variables
df %>% summarise_all(n_distinct) 
# SEX has 6 unique values - should be only 2
# Education has 7 values - should be 6,
# Marriage has 4, should be 3 only
# PAY_AMT could be transformed as binary

# LIMIT_BAL
p<-ggplot(data=df, aes(x=LIMIT_BAL)) +
  geom_histogram()
p

# SEX
unique(df$SEX) # SEX contains wrong inputs
tb_sex <- table(df$SEX)
tb_sex # count 4

p<-ggplot(data=df, aes(x=SEX)) +
  geom_histogram(stat="count")
p

# MARRIAGE
tb_marriage <- table(df$MARRIAGE) # check distribution
tb_marriage # 0 <- 3

p<-ggplot(data=df, aes(x=MARRIAGE)) +
  geom_histogram()
p

# AGE
unique(df$AGE)
tb_age <- table(df$AGE) # check distribution
tb_age # 
summary(df$AGE)

p<-ggplot(data=df, aes(x=AGE)) +
  geom_histogram()
p

# EDUCATION
tb_edu <- table(df$EDUCATION) # check distribution
tb_edu # 0 <- 4(others), 6(unknown) <- 5(unknown)

p<-ggplot(data=df, aes(x=EDUCATION)) +
  geom_histogram()
p

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Data Cleaning ####

# LIMIT_BAL
# remove observations with limit bal less than 0
df$LIMIT_BAL <-  as.integer(df$LIMIT_BAL) # change LIMIT_BAL to integer
df <- subset(df, df$LIMIT_BAL > 0) 
# obs removed = 19
# Higher AUC on test set without. (Compared results with vs without)

# SEX
# remove observations with sex != 1 or 2
# obs removed = 0 -- The obs with invalid sex entries were picked up in the previous cleaning step
df <- subset(df, df$SEX == 1 | df$SEX == 2)

# MARRIAGE
# remove observations with marriage != 1,2 or 3
# obs removed = 45
#df <- subset(df, df$MARRIAGE == 1 | df$MARRIAGE == 2 | df$MARRIAGE == 3) 
##(COMPARE RESULTS 0+3 vs 3)
## update - the class 0 = others e.g. de facto, separated etc

# AGE
# remove observations with age > 75
# obs removed = 2
df <- subset(df, df$AGE <= 75)

# EDUCATION
# 0 <- 4(others), 6(unknown) <- 5(unknown)
df$EDUCATION[df$EDUCATION == 0] <- 4 # reducing class
df$EDUCATION[df$EDUCATION == 6] <- 5
table(df$EDUCATION)

# total obs removed from raw dataset = 21 (0.09% of raw data removed)

# Convert default to factor
df$default  <- as.factor(df$default)

# Reduce PAY_AMT columns to 0 and 1
df$PAY_AMT1[df$PAY_AMT1 > 0] <- 1
df$PAY_AMT2[df$PAY_AMT2 > 0] <- 1
df$PAY_AMT3[df$PAY_AMT3 > 0] <- 1
df$PAY_AMT4[df$PAY_AMT4 > 0] <- 1
df$PAY_AMT5[df$PAY_AMT5 > 0] <- 1
df$PAY_AMT6[df$PAY_AMT6 > 0] <- 1

## Build additional factors

# Age Band
#df$AGE_BAND[df$AGE <= 30] <- 1
#df$AGE_BAND[df$AGE > 30 & df$AGE <= 40] <- 2
#df$AGE_BAND[df$AGE > 40 & df$AGE <= 50] <- 3
#df$AGE_BAND[df$AGE > 50 & df$AGE <= 60] <- 4
#df$AGE_BAND[df$AGE > 60 & df$AGE <= 70] <- 5
#df$AGE_BAND[df$AGE > 70] <- 6

#p<-ggplot(data=df, aes(x=AGE_BAND)) +
#  geom_histogram()
#p
#table(df$AGE_BAND)

# NO_PAY_DELAY
df$NO_PAY_DELAY <- case_when(df$PAY_0 > 0 | 
                             df$PAY_2 > 0 | 
                             df$PAY_3 > 0 |
                             df$PAY_4 > 0 |
                             df$PAY_5 > 0 |
                             df$PAY_6 > 0 ~ 0,
                           TRUE ~ 1)

table(df$NO_PAY_DELAY)
# I believe we lose a lot of information here. This is like all or nothing approach. e.g. may miss one but make the rest on time. 

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Partitioning ####

# Split data into testing and training with 80% for training on stratified method
#Dinh: REWORTE THE CODE ABOVE TO SPLIT INTO TRAINSET AND TESTSET USING CARET PACKAGE
set.seed(20220504)
intrain  <- createDataPartition(y=df$default,p=0.8,list=FALSE)
trainset <- df[intrain,]
testset  <- df[-intrain,]


# Validation of train and test set
nrow(trainset) + nrow(testset)
nrow(df)

tab_temp1<-table(trainset[,"default"])
print(prop.table(tab_temp1))
tab_temp2<-table(testset[,"default"])
print(prop.table(tab_temp2))

# subsampling the training_df
set.seed(7)
levels(training_df$default) <- c("no", "yes") # prep for caret
levels(testing_df$default) <- c("no", "yes") # prep for caret
## upsampling
training_up <- upSample(x=trainset[,-ncol(trainset)],# changed to trainset from traing_df
                        y= trainset$default)#changed to trainset from traing_df
str(training_up)
colnames(training_up)[24] <- "default"
table(training_up$default)

## downsampling
training_dn <- downSample(x=trainset[,-ncol(trainset)], #changed to trainset from traing_df
                          y= trainset$default) #changed to trainset from traing_df
str(training_dn)
colnames(training_dn)[24] <- "default"
table(training_dn$default)

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Modelling ####

## Random Forest ####

## Model 1 Dinh  Random  Forest & gbm
## AUTO TUNE MODEL  ##
set.seed(42)
fitControl <- trainControl(method = 'cv', number = 10, 
                           savePredictions = 'final', classProbs = TRUE, summaryFunction = twoClassSummary)

trainDinh <-trainset
testDinh<-testset

#changing for the caret package

trainDinh <-trainset
testDinh<-testset

levels(trainDinh$default) <- c("no", "yes")
levels(testDinh$default) <- c("no", "yes")

## Change variables PAY_ATMX into factors
trainDinh[, c(18:23)] <- lapply(trainDinh[, c(18:23)], factor)
testDinh[, c(18:23)] <- lapply(testDinh[, c(18:23)], factor)

#model
forest1 <- train(default~., data=trainDinh, method="rf", trControl=fitControl, Importance=TRUE, metric="ROC")
print(forest1)
plot(forest1)

#confusion matrix for test set
p1 <- predict(forest1, testDinh, type="raw")
confusionMatrix(p1, testDinh$default, positive = "yes")

plot(varImp(forest1), main="Important features plot") 

#### Manual tunning
tunegrid <- expand.grid(.mtry=13)
modellist <- list()
for (ntree in c(200, 250, 300, 350)) {
  set.seed(42)
  fit <- train(default~., data=trainDinh, method="rf",  tuneGrid=tunegrid, trControl=fitControl, ntree=ntree,Importance=TRUE,metric="ROC")
  key <- toString(ntree)
  modellist[[key]] <- fit
}
results <- resamples(modellist)
summary(results)
dotplot(results)

#_________________________________________________________________________#
#_________________________________________________________________________#

## SVM ####
## Model 2 Ivan
svm_model<- 
  svm(default ~ LIMIT_BAL + MARRIAGE + AGE + NO_PAY_DELAY + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6
      , data=trainset, type="C-classification", kernel="linear", scale = TRUE, probability=TRUE)

pred_train <- predict(svm_model,newdata = trainset)
mean(pred_train==trainset$default)

pred_test <- predict(svm_model,testset)
predict_probability <- predict(svm_model,newdata = testset, na.action = na.pass, probability=TRUE)

svm_probabilties <- attr(predict_probability, "probabilities")[,"Y"]
mean(pred_test==testset$default)

cfm <- confusionMatrix(pred_test, testset$default, "Y")

cfm[["overall"]][["Accuracy"]]
cfm[["byClass"]][["Precision"]]
cfm[["byClass"]][["Recall"]]

##AUC/ROC
roc_svm <- roc(testset$default,svm_probabilties, plot=TRUE)
plot(roc_svm, col="red", main="SVM ROC Curve")
auc(roc_svm)

#_________________________________________________________________________#
#_________________________________________________________________________#

## GLM ####
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

testset$prediction<-as.factor(testset$prediction)
#confusion matrix 
cfm <- confusionMatrix(testset$prediction, testset$default, "1")
cfm

#AUC 
library(pROC)
roc_object <- roc(testset$default,testset$probability)
auc(roc_object)


#_________________________________________________________________________#
#_________________________________________________________________________#

## GBM ####
## Model 4 Ryan
############ Train GBM model

fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10,
  summaryFunction=twoClassSummary, classProbs=T,
  savePredictions = T
)

gbmGrid_v1 <-  expand.grid(interaction.depth = 5, 
                        n.trees = 100, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

gbmGrid_v2 <-  expand.grid(interaction.depth = 4, 
                        n.trees = 100, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)



gbmFit1 <- train(default ~ ., data = training_df, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE,
                 metric = "ROC",
                 tuneGrid = gbmGrid_v2)
gbmFit1

# fit using downsample
gbmFit2 <- train(default ~ ., data = training_dn, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 metric = "ROC",
                 tuneGrid = gbmGrid_v2)
gbmFit2

# fit using upsample
gbmFit3 <- train(default ~ ., data = training_up, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE,
                 metric = "ROC",
                 tuneGrid = gbmGrid_v2)
gbmFit3


varImp(gbmFit3)

#Let's get our predictions, confusion matrix and auc
testing_df$predictions = predict(gbmFit3, newdata = testing_df)
#Let us check the confusion matrix
confusionMatrix(data = testing_df$predictions, reference = testing_df$default,
                mode = "everything", positive="yes")

testing_df$probability <- predict(gbmFit3, newdata = testing_df, type = "prob")

pred = prediction(testing_df$probability[,2], testing_df$default)

#Let us look at the AUC
auc = performance(pred, "auc")@y.values[[1]]
auc

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Modelling Summary ####

## Confusion Matrix
## AUC of each model
# add a table
# add ROCs for all 4 models


## ROC
plot(roc_object, col="blue", main="ROC Curve")
plot(roc_svm,  col = "red", add = TRUE)
legend("right", legend = c("glm", "svm"), col = c("blue", "red"), lty=1:1)

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Model Optimisations ####

##John GBM Testing## 
control_JR <- trainControl(method = "cv",
                           number = 5,
                           search="grid",
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE,
                           allowParallel = TRUE
)

trainset_JR<-trainset
testset_JR<-testset

str(trainset_JR)
str(testset_JR)

trainset_JR$SEX<-as.factor(trainset_JR$SEX)

levels(trainset_JR$default) <- c("no", "yes")
levels(testset_JR$default) <- c("no", "yes")

#random search Tune length=5
gbm_fit_rand_JR = train(x = trainset_JR[, -length(trainset_JR)], 
                        y = trainset_JR$default, 
                        method = "gbm", 
                        trControl = control_JR,
                        tuneLength = 5, #Here we set how many to sample
                        verbose = T,
                        metric = "ROC"
                        
)


print(gbm_fit_rand_JR)

#Tune length= 10
gbm_fit_rand_JR10 = train(x = trainset_JR[, -length(trainset_JR)], 
                          y = trainset_JR$default, 
                          method = "gbm", 
                          trControl = control_JR,
                          tuneLength = 10,
                          verbose = T,
                          metric = "ROC")
print(gbm_fit_rand_JR)
print(gbm_fit_rand_JR10)

#both random serarch gives model with best ROC = n tree=50, depth=1, shrinkage=0.1, nminobsinnode=10

#get  ROC for gbm_fit_rand_JR10 
gbm_fit_rand_grid <-  expand.grid(interaction.depth = 1, 
                                  n.trees = 50, 
                                  shrinkage = 0.1,
                                  n.minobsinnode = 10)

gbmFitgrand <- train(default ~ ., data = trainset_JR, 
                     method = "gbm", 
                     trControl = control_JR, 
                     verbose = FALSE,
                     metric = "ROC",
                     tuneGrid = gbm_fit_rand_grid)

gbmFitgrand #ROC 0.75

testset_JR$predictions=predict(gbmFitgrand, newdata=testset_JR)

#prediction on table
table(testset_JR$predictions)

#CFM
confusionMatrix(data = testset_JR$predictions, reference = testset_JR$default,
                mode = "everything", positive="yes")

### upsample###
trainup_JR<-upSample(x=trainset[,-ncol(trainset)],
                     y= trainset$default)
str(trainup_JR)
colnames(trainup_JR)[24] <- "default"
table(trainup_JR$default)

testup_JR<-upSample(x=testset[,-ncol(testset)],
                    y= testset$default)
str(testup_JR)
colnames(testup_JR)[24] <- "default"
table(testup_JR$default)

trainup_JR$SEX<-as.factor(trainup_JR$SEX)

levels(trainup_JR$default) <- c("no", "yes")
levels(testup_JR$default) <- c("no", "yes")


#Tune length= 10
gbm_fit_uprand_JR10 = train(x = trainup_JR[, -length(trainup_JR)], 
                            y = trainup_JR$default, 
                            method = "gbm", 
                            trControl = control_JR,
                            tuneLength = 10,
                            verbose = T,
                            metric = "ROC")

print(gbm_fit_uprand_JR10)

#same parameters as out put from previous random search on unbalanced data set


#### Dinh tried gbm 
### GBM Model 
set.seed(42)
control<- trainControl(method = "cv",
                       number = 10, #Making a simple cv for speed here
                       search="grid",
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       allowParallel = TRUE
)

names(trainDinh)
#####default grid
gbm = train( x = trainDinh[, -24],  y = trainDinh$default,  
             method = "gbm", 
             trControl = control, verbose = FALSE, metric="ROC")


gbm

p1_gbm <-predict(gbm, testDinh)
cfm_gbm <- confusionMatrix(data=p1_gbm, reference=as.factor(testDinh$default), positive="yes")
cfm_gbm
imp = varImp(gbm)

#As a table
imp
# As a plot
plot(imp, main="Important variables in the gbm model")


#EXPAND GRID
# #EXPAND GRID
# gbm_grid =  expand.grid(
#   interaction.depth = c(3,4),
#   n.trees = (10:15)*100, 
#   shrinkage = 0.01,
#   n.minobsinnode = 10
# )
gbm_grid =  expand.grid(
  interaction.depth = c(3,4,5),
  n.trees = 2000, 
  shrinkage = c(0.005,0.01),
  n.minobsinnode = 10
)
set.seed(42)
gbm_1 = train( x = trainDinh[, -24],  y = trainDinh$default, 
               method = "gbm", 
               trControl = control, verbose = FALSE,
               tuneGrid= gbm_grid, metric="ROC")

gbm_1


p1_gbm <-predict(gbm_1, testDinh)
cfm_gbm <- confusionMatrix(data=p1_gbm, reference=as.factor(tesDinh$default), positive="yes")
cfm_gbm


gbm_prob <- predict(gbm_1, testDinh, type = "prob")$"Y"
perf_gbm <- prediction(gbm_prob, testDinh$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_gbm, col = "blue", lty = 2, main="ROC curve using gbm")
abline(a=0,b=1)


gbm_auc <-performance(prediction(gbm_prob, testDinh$default),"auc")
gbm_auc<-unlist(slot(gbm_auc, "y.values"))
gbm_auc

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Optimisation Summary ####

## AUC of each version of GBM models
# add a table
# add ROCs for key GBM models showing improvements after tuning

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Final GBM Model ####



#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Output ####
## Produce validation output

##
