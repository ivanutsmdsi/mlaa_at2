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
library(pROC)
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

# Reduce PAY_AMT columns to 0 and 1
df$PAY_AMT1[df$PAY_AMT1 > 0] <- 1
df$PAY_AMT2[df$PAY_AMT2 > 0] <- 1
df$PAY_AMT3[df$PAY_AMT3 > 0] <- 1
df$PAY_AMT4[df$PAY_AMT4 > 0] <- 1
df$PAY_AMT5[df$PAY_AMT5 > 0] <- 1
df$PAY_AMT6[df$PAY_AMT6 > 0] <- 1

str(df)
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
# - assumption no longer vaild
# df$NO_PAY_DELAY <- case_when(df$PAY_0 > 0 | 
#                             df$PAY_2 > 0 | 
#                             df$PAY_3 > 0 |
#                             df$PAY_4 > 0 |
#                             df$PAY_5 > 0 |
#                             df$PAY_6 > 0 ~ 0,
#                           TRUE ~ 1)

# table(df$NO_PAY_DELAY)

# Convert categorical variables to factors
df$default  <- as.factor(df$default)
df$SEX <- as.factor(df$SEX)
df$EDUCATION <- as.factor(df$EDUCATION)
df$MARRIAGE <- as.factor(df$MARRIAGE)
df$PAY_AMT1 <- as.factor(df$PAY_AMT1)
df$PAY_AMT2 <- as.factor(df$PAY_AMT2)
df$PAY_AMT3 <- as.factor(df$PAY_AMT3)
df$PAY_AMT4 <- as.factor(df$PAY_AMT4)
df$PAY_AMT5 <- as.factor(df$PAY_AMT5)
df$PAY_AMT6 <- as.factor(df$PAY_AMT6)

str(df)

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
levels(trainset$default) <- c("no", "yes") # prep for caret
levels(testset$default) <- c("no", "yes") # prep for caret

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


levels(trainDinh$default) <- c("no", "yes")
levels(testDinh$default) <- c("no", "yes")

## Change variables PAY_ATMX into factors
trainDinh[, c(18:23)] <- lapply(trainDinh[, c(18:23)], factor)
testDinh[, c(18:23)] <- lapply(testDinh[, c(18:23)], factor)

trainDinh$SEX <- as.integer(trainDinh$SEX)
testDinh$SEX <- as.integer(testDinh$SEX)



#model
forest1 <- train(default~., data=trainDinh, method="rf", trControl=fitControl, Importance=TRUE, metric="ROC")
print(forest1)
plot(forest1)

#confusion matrix for train set
p1_train <- predict(forest1, trainDinh, type="raw")
confusionMatrix(p1_train, trainDinh$default, positive = "yes")


#confusion matrix for test set
p1_test<- predict(forest1, testDinh, type="raw")
confusionMatrix(p1_test, testDinh$default, positive = "yes")

test_pred1 <- predict(forest1, testDinh, type="prob")
train_pred1 <- predict(forest1, trainDinh, type="prob")

library(pROC)
#ROC curve on train and test sets
plot(roc(testDinh$default, test_pred1[[2]]), print.auc=TRUE, col="red", 
     xlim=c(0,1), main="Model 1 with trainset")
plot(roc(trainDinh$default, train_pred1[[2]]), print.auc=TRUE, col="blue", 
     xlim=c(0,1))

plot(varImp(forest1), main="Important features plot") 


### Remove AGE_BAND, NO_PAY_DELAY  variables ad they have high correlations with other variables
#model
forest2 <- train(default~.-AGE_BAND-NO_PAY_DELAY, data=trainDinh, method="rf", trControl=fitControl, Importance=TRUE, metric="ROC")
print(forest2)
plot(forest2)

#confusion matrix for test set
p2_test<- predict(forest2, testDinh, type="raw")
confusionMatrix(p2_test, testDinh$default, positive = "yes")

test_pred2 <- predict(forest2, testDinh, type="prob")
train_pred2 <- predict(forest2, trainDinh, type="prob")

#ROC curve on train and test sets
plot(roc(testDinh$default, test_pred2[[2]]), print.auc=TRUE, col="red", 
     xlim=c(0,1), main="model 2 trainset")
plot(roc(trainDinh$default, train_pred2[[2]]), print.auc=TRUE, col="blue", 
     xlim=c(0,1))
#### model 2 perform a bit better with an auc =0.792 whild model 1 has auc=0.791



#### Drop some unimportant variables
forest3 <- train(default~.-AGE_BAND-NO_PAY_DELAY-PAY_AMT1-PAY_AMT2-PAY_AMT3-PAY_AMT4-PAY_AMT5-PAY_AMT6, data=trainDinh, method="rf", trControl=fitControl, Importance=TRUE, metric="ROC")
print(forest3)
plot(forest3)

#confusion matrix for test set
p3_test<- predict(forest3, testDinh, type="raw")
confusionMatrix(p3_test, testDinh$default, positive = "yes")

test_pred3 <- predict(forest3, testDinh, type="prob")
train_pred3 <- predict(forest3, trainDinh, type="prob")


#ROC curve on train and test sets
plot(roc(testDinh$default, test_pred3[[2]]), print.auc=TRUE, col="red", 
     xlim=c(0,1), main="model 3 with test set")

#### model 3 perform a bit better with an auc =0.793 while model 1 has auc=0.791, model 2 has auc=0.792


#### Manual tunning
# tunegrid <- expand.grid(.mtry=c(5:12))
# modellist <- list()
# for (ntree in c(200, 300)) {
#   set.seed(42)
#   fit <- train(default~.-AGE_BAND-NO_PAY_DELAY-PAY_AMT1-PAY_AMT2-PAY_AMT3-PAY_AMT4-PAY_AMT5-PAY_AMT6, data=trainDinh, method="rf",  tuneGrid=tunegrid, trControl=fitControl, ntree=ntree,Importance=TRUE,metric="ROC")
#   key <- toString(ntree)
#   modellist[[key]] <- fit
# }
# results <- resamples(modellist)
# summary(results)
# dotplot(results)

tunegrid <- expand.grid(.mtry=c(1:12))
forest4 <- train(default~.-AGE_BAND-NO_PAY_DELAY-PAY_AMT1-PAY_AMT2-PAY_AMT3-PAY_AMT4-PAY_AMT5-PAY_AMT6, data=trainDinh, method="rf",  tuneGrid=tunegrid, trControl=fitControl, ntree=200,Importance=TRUE,metric="ROC")

print(forest4)
plot(forest4)

#confusion matrix for test set
p4_test<- predict(forest4, testDinh, type="raw")
confusionMatrix(p4_test, testDinh$default, positive = "yes")

test_pred4 <- predict(forest4, testDinh, type="prob")
train_pred4 <- predict(forest4, trainDinh, type="prob")


#ROC curve on train and test sets
plot(roc(testDinh$default, test_pred4[[2]]), print.auc=TRUE, col="red", 
     xlim=c(0,1), main="model 4 with test set")


### diferent ntree
tunegrid <- expand.grid(.mtry=c(4:8))
forest5 <- train(default~.-AGE_BAND-NO_PAY_DELAY-PAY_AMT1-PAY_AMT2-PAY_AMT3-PAY_AMT4-PAY_AMT5-PAY_AMT6, data=trainDinh, method="rf",  tuneGrid=tunegrid, trControl=fitControl, ntree=500,Importance=TRUE,metric="ROC")

print(forest5)
plot(forest5)

#confusion matrix for test set
p5_test<- predict(forest5, testDinh, type="raw")
confusionMatrix(p5_test, testDinh$default, positive = "yes")

test_pred5 <- predict(forest5, testDinh, type="prob")
train_pred5 <- predict(forest5, trainDinh, type="prob")


#ROC curve on train and test sets
plot(roc(testDinh$default,  test_pred5[[2]]),print.auc=TRUE, col="red", 
     xlim=c(0,1), main="model 5 with test set")

tunegrid <- expand.grid(.mtry=c(6:14))
forest6 <- train(default~.-AGE_BAND-NO_PAY_DELAY, data=trainDinh, method="rf",  tuneGrid=tunegrid, trControl=fitControl, ntree=100,Importance=TRUE,metric="ROC")


print(forest6)
plot(forest6)

test_pred6 <- predict(forest6, testDinh, type="prob")
train_pred6 <- predict(forest6, trainDinh, type="prob")


#ROC curve on train and test sets
plot(roc(testDinh$default,  test_pred6[[2]]),print.auc=TRUE, col="red", 
     xlim=c(0,1), main="model 5 with test set")



tunegrid <- expand.grid(.mtry=8)
forest7 <- train(default~.-AGE_BAND-NO_PAY_DELAY, data=trainDinh, method="rf",  tuneGrid=tunegrid, trControl=fitControl, ntree=500,Importance=TRUE, metric='ROC')


print(forest7)
plot(forest7)

test_pred7 <- predict(forest7, testDinh, type="prob")
train_pred7 <- predict(forest7, trainDinh, type="prob")


#ROC curve on train and test sets
plot(roc(testDinh$default,  test_pred7[[2]]),print.auc=TRUE, col="red", 
     xlim=c(0,1), main="model 7 with test set")



tunegrid <- expand.grid(.mtry=8)
forest8 <- train(default~.-AGE_BAND-NO_PAY_DELAY, data=trainDinh, method="rf",  tuneGrid=tunegrid, trControl=fitControl, ntree=1000,Importance=TRUE, metric='ROC')


print(forest8)


test_pred8 <- predict(forest8, testDinh, type="prob")
train_pred8 <- predict(forest8, trainDinh, type="prob")


#ROC curve on train and test sets
plot(roc(testDinh$default,  test_pred8[[2]]),print.auc=TRUE, col="red", 
     xlim=c(0,1), main="model 8 with test set")


test<-read.csv("AT2_credit_test.csv")


ID_column <- test$ID

ID_column
a<- c("no", "yes")


test$default <-sample(a,1)
test$default <-as.factor(test$default)
levels(test$default) <- c("no", "yes")
test$LIMIT_BAL <-  as.integer(test$LIMIT_BAL)
test$ID <- NULL # drop ID




# total obs removed from raw dataset = 78 (0.33% of raw data removed)

# Reduce PAY_AMT columns to 0 and 1
test$PAY_AMT1[test$PAY_AMT1 > 0] <- 1
test$PAY_AMT2[test$PAY_AMT2 > 0] <- 1
test$PAY_AMT3[test$PAY_AMT3 > 0] <- 1
test$PAY_AMT4[test$PAY_AMT4 > 0] <- 1
test$PAY_AMT5[test$PAY_AMT5 > 0] <- 1
test$PAY_AMT6[test$PAY_AMT6 > 0] <- 1




## Build additional factors

# Age Band
test$AGE_BAND[test$AGE <= 30] <- 1
test$AGE_BAND[test$AGE > 30 & test$AGE <= 40] <- 2
test$AGE_BAND[test$AGE > 40 & test$AGE <= 50] <- 3
test$AGE_BAND[test$AGE > 50 & test$AGE <= 60] <- 4
test$AGE_BAND[test$AGE > 60 & test$AGE <= 70] <- 5
test$AGE_BAND[test$AGE > 70] <- 6



# NO_PAY_DELAY
test$NO_PAY_DELAY <- case_when(test$PAY_0 > 0 | 
                                 test$PAY_2 > 0 | 
                                 test$PAY_3 > 0 |
                                 test$PAY_4 > 0 |
                                 test$PAY_5 > 0 |
                                 test$PAY_6 > 0 ~ 0,
                               TRUE ~ 1)


test[, c(18:23)] <- lapply(test[, c(18:23)], factor)


#test$SEX <- as.character(test$SEX)


#using forest model 
pred_rf<- predict(forest7, test, type="prob")
pred_rf

test_rf <-cbind(ID_column, pred_rf)

test_rf
dim(test_rf)

test_rf <-test_rf[,-2]

column_names <- c("ID", "default")
colnames(test_rf) <-column_names
View(test_rf)

is.data.frame(test_rf)
names(test_rf)
names(sample)

write.csv(test_rf, "rf_revisit.csv",row.names=FALSE)




gbm_grid1 =  expand.grid(
  interaction.depth = c(3,4,5),
  n.trees = 200, 
  shrinkage = c(0.1,0.01),
  n.minobsinnode = 10
)
set.seed(42)
gbm_1 = train( x = trainDinh[, -c(24,25,26)],  y = trainDinh$default, 
               method = "gbm", 
               trControl = control, verbose = FALSE,
               tuneGrid= gbm_grid1, metric="ROC")

gbm_1


p1_gbm <-predict(gbm_1, testDinh)
cfm_gbm1 <- confusionMatrix(data=p1_gbm, reference=as.factor(testDinh$default), positive="yes")
cfm_gbm1


gbm_prob1 <- predict(gbm_1, testDinh, type = "prob")$"yes"
perf_gbm1 <- prediction(gbm_prob1, testDinh$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_gbm1, col = "blue", lty = 2, main="ROC curveusing gbm")
abline(a=0,b=1)


gbm_auc1 <-performance(prediction(gbm_prob1, testDinh$default),"auc")
gbm_auc1<-unlist(slot(gbm_auc1, "y.values"))
gbm_auc1

#increase the number of trees
gbm_grid2 =  expand.grid(
  interaction.depth = c(3,4,5),
  n.trees = c(200,300,400,500), 
  shrinkage = c(0.1,0.05, 0.01),
  n.minobsinnode = c(5,10,15)
)
set.seed(42)
gbm_2 = train( x = trainDinh[, -c(24,25,26)],  y = trainDinh$default, 
               method = "gbm", 
               trControl = control, verbose = FALSE,
               tuneGrid= gbm_grid2, metric="ROC")

gbm_2


#gbm 2 gitROC was used to select the optimal model using the largest value.
#The final values used for the model were n.trees = 300, interaction.depth = 5, shrinkage = 0.05
#and n.minobsinnode = 15.

p_gbm2 <-predict(gbm_2, testDinh)
cfm_gbm2 <- confusionMatrix(data=p_gbm2, reference=as.factor(testDinh$default), positive="yes")
cfm_gbm2


gbm_prob2 <- predict(gbm_2, testDinh, type = "prob")$"yes"
perf_gbm2 <- prediction(gbm_prob2, testDinh$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_gbm2, col = "blue", lty = 2, main="ROC curveusing gbm2")
abline(a=0,b=1)


gbm_auc2 <-performance(prediction(gbm_prob2, testDinh$default),"auc")
gbm_auc2<-unlist(slot(gbm_auc2, "y.values"))
gbm_auc2



#increase the number of trees
gbm_grid3 =  expand.grid(
  interaction.depth = c(3,4,5),
  n.trees = c(200,300,400,500), 
  shrinkage = c(0.1,0.05, 0.01),
  n.minobsinnode = c(5,10)
)
set.seed(42)
gbm_3 = train( x = trainDinh[, -c(24,25,26)],  y = trainDinh$default, 
               method = "gbm", 
               trControl = control, verbose = FALSE,
               tuneGrid= gbm_grid3, metric="ROC")

gbm_3
# gbm3 ROC was used to select the optimal model using the largest value.
#The final values used for the model were n.trees = 500, interaction.depth = 5, shrinkage = 0.05
#and n.minobsinnode = 10.

p_gbm3 <-predict(gbm_3, testDinh)
cfm_gbm3 <- confusionMatrix(data=p_gbm3, reference=as.factor(testDinh$default), positive="yes")
cfm_gbm3


gbm_prob3 <- predict(gbm_3, testDinh, type = "prob")$"yes"
perf_gbm3 <- prediction(gbm_prob3, testDinh$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_gbm3, col = "blue", lty = 2, main="ROC curveusing gbm2")
abline(a=0,b=1)


gbm_auc3 <-performance(prediction(gbm_prob3, testDinh$default),"auc")
gbm_auc3<-unlist(slot(gbm_auc3, "y.values"))
gbm_auc3


###gbm4

gbm_grid4 =  expand.grid(
  interaction.depth = c(3,4,5),
  n.trees = c(200,300,400,500), 
  shrinkage = c(0.1,0.05),
  n.minobsinnode = c(5,10,15)
)
set.seed(42)
gbm_4 = train( x = trainDinh[, -c(24,25,26)],  y = trainDinh$default, 
               method = "gbm", 
               trControl = control, verbose = FALSE,
               tuneGrid= gbm_grid4, metric="ROC")

gbm_4


##gbm 4 ROC was used to select the optimal model using the largest value.
#The final values used for the model were n.trees = 200, interaction.depth = 5, shrinkage = 0.1
#and n.minobsinnode = 15.

p_gbm4 <-predict(gbm_4, testDinh)
cfm_gbm4 <- confusionMatrix(data=p_gbm4, reference=as.factor(testDinh$default), positive="yes")
cfm_gbm4


gbm_prob4 <- predict(gbm_4, testDinh, type = "prob")$"yes"
perf_gbm4 <- prediction(gbm_prob4, testDinh$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_gbm4, col = "blue", lty = 2, main="ROC curveusing gbm2")
abline(a=0,b=1)


gbm_auc4 <-performance(prediction(gbm_prob4, testDinh$default),"auc")
gbm_auc4<-unlist(slot(gbm_auc4, "y.values"))
gbm_auc4


#gbm2 give 0.79088, gbm3 gives 0.79044, gmb4 give 0.78...

# apply on test set for Kaggle
pred_gbm2<- predict(gbm_2, test, type="prob")
pred_gbm2

test_gbm2 <-cbind(ID_column, pred_gbm2)

test_gbm2
dim(test_gbm2)

test_gbm2 <-test_gbm2[,-2]

column_names <- c("ID", "default")
colnames(test_gbm2) <-column_names
View(test_gbm2)

is.data.frame(test_gbm2)
names(test_gbm2)
names(sample)

write.csv(test_gbm2, "gbm2_revisit.csv",row.names=FALSE)







pred_gbm3<- predict(gbm_3, test, type="prob")
pred_gbm3

test_gbm3 <-cbind(ID_column, pred_gbm3)

test_gbm3
dim(test_gbm3)

test_gbm3 <-test_gbm3[,-2]

column_names <- c("ID", "default")
colnames(test_gbm3) <-column_names
View(test_gbm3)

is.data.frame(test_gbm3)
names(test_gbm3)
names(sample)

write.csv(test_gbm3, "gbm_revisit.csv",row.names=FALSE)


pred_gbm4<- predict(gbm_4, test, type="prob")
pred_gbm4

test_gbm4 <-cbind(ID_column, pred_gbm4)

test_gbm4
dim(test_gbm4)

test_gbm4 <-test_gbm4[,-2]

column_names <- c("ID", "default")
colnames(test_gbm4) <-column_names
View(test_gbm4)

is.data.frame(test_gbm4)
names(test_gbm4)
names(sample)

write.csv(test_gbm4, "gbm4_revisit.csv",row.names=FALSE)


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

svm_probabilties <- attr(predict_probability, "probabilities")[,"1"]
mean(pred_test==testset$default)

cfm <- confusionMatrix(pred_test, testset$default, "1")

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



gbmFit1 <- train(default ~ ., data = trainset, 
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
testset$predictions = predict(gbmFit3, newdata = testset)
#Let us check the confusion matrix
confusionMatrix(data = testset$predictions, reference = testset$default,
                mode = "everything", positive="yes")

testset$probability <- predict(gbmFit3, newdata = testset, type = "prob")

pred = prediction(testset$probability[,2], testset$default)

#Let us look at the AUC
auc = performance(pred, "auc")@y.values[[1]]
auc

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Modelling Summary ####

# Ivan
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

trainset_JR$SEX<-as.factor(trainset_JR$SEX)

table(trainset_JR$default)

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

gbmFitgrand #}}}}ROC 0.7553836

testset_JR$predictions=predict(gbmFitgrand, newdata=testset_JR)

#prediction on table
table(testset_JR$predictions)

#CFM
confusionMatrix(data = testset_JR$predictions, reference = testset_JR$default,
                mode = "everything", positive="yes")

#AUC
testset_JR$probability <- predict(gbmFitgrand, newdata = testset_JR, type = "prob")

pred_JR = prediction(testset_JR$probability[,2], testset_JR$default)

auc = performance(pred_JR, "auc")@y.values[[1]]
auc#}}}}} AUC 0.7502347

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

pred <- prediction(testset$predictions, testset$labels )
pred2 <- prediction(abs(testset$predictions + 
                          rnorm(length(testset$predictions), 0, 0.1)), 
                    testset$labels)
perf <- performance( pred, "tpr", "fpr" )
perf2 <- performance(pred2, "tpr", "fpr")
plot(perf, colorize = TRUE)
plot(perf2, add = TRUE, colorize = TRUE)


## AUC of each version of GBM models
# add a table
# add ROCs for key GBM models showing improvements after tuning

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Final GBM Model ####

# Dihn


#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Output ####
# Prep the validation set
df_validation = read.csv("AT2_credit_test.csv")
levels(df$default) <- c("no", "yes") # prep for caret

df_validation$EDUCATION[df_validation$EDUCATION == 0] <- 4 # reducing class
df_validation$EDUCATION[df_validation$EDUCATION == 6] <- 5
df_validation$PAY_AMT1[df_validation$PAY_AMT1 != 0] <- 1
df_validation$PAY_AMT2[df_validation$PAY_AMT2 != 0] <- 1
df_validation$PAY_AMT3[df_validation$PAY_AMT3 != 0] <- 1
df_validation$PAY_AMT4[df_validation$PAY_AMT4 != 0] <- 1
df_validation$PAY_AMT5[df_validation$PAY_AMT5 != 0] <- 1
df_validation$PAY_AMT6[df_validation$PAY_AMT6 != 0] <- 1

# Define final categorical variables 
df_validation$SEX <- as.factor(df_validation$SEX)
df_validation$EDUCATION <- as.factor(df_validation$EDUCATION)
df_validation$MARRIAGE <- as.factor(df_validation$MARRIAGE)
df_validation$PAY_AMT1 <- as.factor(df_validation$PAY_AMT1)
df_validation$PAY_AMT2 <- as.factor(df_validation$PAY_AMT2)
df_validation$PAY_AMT3 <- as.factor(df_validation$PAY_AMT3)
df_validation$PAY_AMT4 <- as.factor(df_validation$PAY_AMT4)
df_validation$PAY_AMT5 <- as.factor(df_validation$PAY_AMT5)
df_validation$PAY_AMT6 <- as.factor(df_validation$PAY_AMT6)


## Produce validation output
pred_prob <- predict(gbm_fnl, newdata = df_validation, type = "prob",predict.all=TRUE)
target_prob <- pred_prob$yes 

df_validation$default <- target_prob # add probabilities as default

#prepare only ID, target_probability, target_class for export
output_export <- df_validation %>% dplyr::select(ID, default)

# Export as csv
write.csv(output_export,"~/GitHub/mlaa_at2/MLAA_AT2_output.csv", row.names = FALSE)


##
