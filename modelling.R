###
#
# MLAA Assignment 2 Part A ####
#
# Students:
# Ivan Cheung - 13975420
# Ryan Yeo - 14328254
# Dinh Tran - 14382497
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
library(randomForest)

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
levels(df$default) <- c("no", "yes")
df$default <- as.integer(df$default) # set as integer
df$default <- as.factor(df$default) # set as factor
levels(df$default) # check factor, 1 = yes
tb_default <- table(df$default) # check target distribution
tb_default # 0:1 = 16974:6127

# checking the ratio
tbl_prop <- prop.table(tb_default)
tbl_prop # we have unbalanced data. approx 0:1 = 3:1

# visualise
p<-ggplot(data=df)+aes(x=default, fill=default)+geom_bar()
p

## Independent variables ####

# check other independent variables
df %>% summarise_all(n_distinct) 
# SEX has 6 unique values - should be only 2
# Education has 7 values - should be 6,
# Marriage has 4, should be 3 only
# PAY_AMT could be transformed as binary

# LIMIT_BAL
# p<-ggplot(data=df, aes(x=LIMIT_BAL)) +
#   geom_histogram()
# p


p <- ggplot(df, aes(x = default, y = LIMIT_BAL, colour=default)) +
  geom_boxplot(size = .75) +
  geom_jitter(alpha = .5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))+labs(x="default payment next month", y="Limit Balance")
p


p <- ggplot(df, aes(x = default, y = LIMIT_BAL, colour=default)) +
  geom_boxplot(size = .75)+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))+labs(x="default payment next month", y="Limit Balance")
p

# SEX
unique(df$SEX) # SEX contains wrong inputs
tb_sex <- table(df$SEX)
tb_sex # count 4

# p<-ggplot(data=df, aes(x=SEX), fill="SEX") +
#   geom_histogram(stat="count")
p<-ggplot(data=df)+aes(x=SEX, fill=SEX)+geom_bar()+labs(title="Gender distribution" )
p

# MARRIAGE
tb_marriage <- table(df$MARRIAGE) # check distribution
tb_marriage # 0 <- 3

p<-ggplot(data=df)+ aes(x=as.factor(MARRIAGE), fill=as.factor(MARRIAGE)) +
     geom_bar() +labs( x="MARRIAGE",title="Marriage by categories", fill = "MARRIAGE" )
p

# AGE
unique(df$AGE)
tb_age <- table(df$AGE) # check distribution
tb_age # 
summary(df$AGE)

p<-ggplot(data=df, aes(x=AGE)) +
  geom_histogram(colour = "blue", fill = "white", 
                 binwidth = 10, boundary=20)+ scale_x_continuous( breaks = seq(20, 160, by = 10))+labs( x="AGE",title="Age distribution")
p

# EDUCATION
tb_edu <- table(df$EDUCATION) # check distribution
tb_edu # 0 <- 4(others), 6(unknown) <- 5(unknown)

# p<-ggplot(data=df, aes(x=EDUCATION)) +
#   geom_histogram( binwidth = 1)
# p

p<-ggplot(data=df)+ aes(x=as.factor(EDUCATION), fill=as.factor(EDUCATION)) +
  geom_bar() +labs( x="EDUCATION",title="Education by categories", fill = "EDUCATION" )

p

#default


p<-ggplot(data=df)+aes(x=default, fill=as.factor(SEX))+geom_bar() +labs(title="Default payment next month by gender", fill = "SEX" )
p


p<-ggplot(data=df)+aes(x=default, fill=as.factor(EDUCATION))+geom_bar() +labs(title="Default payment next month by education level", fill = "EDUCATION" )
p

p<-ggplot(data=df)+aes(x=as.factor(EDUCATION), fill=as.factor(default))+geom_bar() +labs(title="Default payment next month by education level", fill = "EDUCATION" )
p

p<-ggplot(data=df)+aes(x=as.factor(EDUCATION), fill=as.factor(EDUCATION))+geom_bar() +labs(title="Default payment next month by education level", fill = "EDUCATION" )+facet_wrap(~default)
p

 p<-ggplot(data=df)+aes(x=default, fill=as.factor(MARRIAGE))+geom_bar() +labs(title="Default payment next month by marriage status", fill = "MARRIAGE" )
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
df$SEX <- as.integer(df$SEX)

# MARRIAGE

df$MARRIAGE[df$MARRIAGE == 0] <- 3 # combine 0 & 3

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
#df$SEX <- as.factor(df$SEX)
#df$EDUCATION <- as.factor(df$EDUCATION)
#df$MARRIAGE <- as.factor(df$MARRIAGE)
df$PAY_AMT1 <- as.factor(df$PAY_AMT1)
df$PAY_AMT2 <- as.factor(df$PAY_AMT2)
df$PAY_AMT3 <- as.factor(df$PAY_AMT3)
df$PAY_AMT4 <- as.factor(df$PAY_AMT4)
df$PAY_AMT5 <- as.factor(df$PAY_AMT5)
df$PAY_AMT6 <- as.factor(df$PAY_AMT6)

# Convert categorical variables to factors #Dinh rewrote this to make it shorter

names <- names(df)
#finding column numbers for PAY_AMT1. PAY_ATM6
n1<- which(names=="PAY_AMT1")
n2<- which(names=="PAY_AMT6")

#change "SEX",  "EDUCATION" , "MARRIAGE"  into factor
df[, c(2:4)] <- lapply(df[, c(2:4)], factor)
#change columns from 18 to 24 (PAY_AMTX and default) into factor
df[, c(n1:n2+1)] <- lapply(df[, c(n1:n2+1)], factor)

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Partitioning ####

# Split data into testing and training with 80% for training on stratified method
#Dinh: REWROTE THE CODE ABOVE TO SPLIT INTO TRAINSET AND TESTSET USING CARET PACKAGE
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

## Create results df
cfm <- data.frame(model = character(), acc = numeric(), prec = numeric(), recall = numeric(), f1 = numeric())

auc <- data.frame(model = character(), auc = numeric())

## Random Forest ####

## Model 1 Dinh  Random  Forest 

set.seed(42)
fitControl <- trainControl(method = 'cv', number = 10, 
                           savePredictions = 'final', classProbs = TRUE, summaryFunction = twoClassSummary)

trainDinh <-trainset
testDinh<-testset

#changing for the caret package

levels(trainDinh$default) <- c("no", "yes")
levels(testDinh$default) <- c("no", "yes")

## Change variables SEX< EDUCATION, MARRIAGE back to integers
 trainDinh$SEX <- as.integer(trainDinh$SEX)
 trainDinh$EDUCATION <- as.integer(trainDinh$EDUCATION)
 trainDinh$MARRIAGE <- as.integer(trainDinh$MARRIAGE)
 
 ## Change variables SEX< EDUCATION, MARRIAGE back to integers
 testDinh$SEX <- as.integer(testDinh$SEX)
 testDinh$EDUCATION <- as.integer(testDinh$EDUCATION)
 testDinh$MARRIAGE <- as.integer(testDinh$MARRIAGE)
 
 #simple random forest
 rftrain <- randomForest(default~., data = trainDinh,  ntree=200)
rftrain

rftrain_pred <-predict(rftrain, testDinh)
 confusionMatrix(rftrain_pred, testDinh$default, positive = 'yes')
 
 rftrain_prob <- predict(rftrain,  type = "prob", newdata=testDinh)[,2]
 rftrain_auc <-performance(prediction(rftrain_prob, testDinh$default),"auc")
 rftrain_auc<-unlist(slot(rftrain_auc, "y.values"))
 rftrain_auc
#auc for this model is 0.785

 
 #grid search 
set.seed(42)

#using the suggested mtry=floor(sqrt(ncol(df))) for classification, changing number of trees
tunegrid <- expand.grid(.mtry = c(sqrt(ncol(trainDinh))))


rf_list <- list()

for (ntree in c(200,300)){
  set.seed(42)
  rf_fit <- train(default~.,
               data = trainDinh,
               method = 'rf',
               metric = 'ROC',
               tuneGrid = tunegrid,
               trControl = fitControl,
               ntree = ntree)
  key <- toString(ntree)
  rf_list[[key]] <- rf_fit
}

results1 <- resamples(rf_list)
summary(results1)

dotplot(results1)

rf_list[["200"]]
rf_list[["300"]]


#checking performance on the test set
rf_200_prob <- predict(rf_list[["200"]], testDinh, type = "prob")$"yes"
rf_200_auc <-performance(prediction(rf_200_prob, testDinh$default),"auc")
rf_200_auc<-unlist(slot(rf_200_auc, "y.values"))
rf_200_auc
#rf_list[['200']] gives 0.786 auc on the test set

rf_300_prob <- predict(rf_list[["300"]], testDinh, type = "prob")$"yes"
rf_300_auc <-performance(prediction(rf_300_prob, testDinh$default),"auc")
rf_300_auc<-unlist(slot(rf_300_auc, "y.values"))
rf_300_auc
#rf_list[['300']] gives 0.78828 auc on the test set








#default grid search
forest1 <- train(default~., data=trainDinh, method="rf", trControl=fitControl, Importance=TRUE, metric="ROC")
print(forest1)
#ROC was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 12.

#plot mtry vs ROC
plot(forest1)



#confusion matrix for test set
rf1_test<- predict(forest1, testDinh, type="raw")
cfm_rf1<-confusionMatrix(rf1_test, testDinh$default, positive = "yes")

cfm_temp <- confusionMatrix(rf1_test, testDinh$default, positive = "yes")

d <- c("forest", 
       cfm_temp[["byClass"]][["Balanced Accuracy"]],
       cfm_temp[["byClass"]][["Precision"]],
       cfm_temp[["byClass"]][["Recall"]],
       cfm_temp[["byClass"]][["F1"]])

d <- as.list(d)
names(d) <- names(cfm)

cfm <- rbind(cfm, d)

rf1_prob <- predict(forest1, testDinh, type = "prob")$"yes"
rf1_auc <-performance(prediction(rf1_prob, testDinh$default),"auc")
rf1_auc<-unlist(slot(rf1_auc, "y.values"))
rf1_auc

perf_rf1 <- prediction(rf1_prob, testDinh$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_rf1, col = "blue", lty = 2, main=" Random forest with default grid search")
abline(a=0,b=1)

plot(varImp(forest1), main="Important features plot for random forest with default grid search") 

#### forest1  on the test set  an auc =0.7863344



#### Drop some unimportant variables
set.seed(42)
forest2 <- train(default~.-PAY_AMT1-PAY_AMT2-PAY_AMT3-PAY_AMT4-PAY_AMT5-PAY_AMT6, data=trainDinh, method="rf", trControl=fitControl, Importance=TRUE, metric="ROC")
print(forest2)
plot(forest2)

#confusion matrix for test set
rf_test2<- predict(forest2, testDinh, type="raw")
confusionMatrix(rf_test2, testDinh$default, positive = "yes")

rf2_prob <- predict(forest2, testDinh, type = "prob")$"yes"
rf2_auc <-performance(prediction(rf2_prob, testDinh$default),"auc")
rf2_auc<-unlist(slot(rf2_auc, "y.values"))
rf2_auc

#### forest2  has an auc = 0.7856454

perf_rf2 <- prediction(rf2_prob, testDinh$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_rf2, col = "blue", lty = 2, main=" Random forest default grid search and without PAY_AMTX")
abline(a=0,b=1)


tunegrid <- expand.grid(.mtry=c(8:10))
set.seed(42)
forest3 <- train(default~., data=trainDinh, method="rf",  tuneGrid=tunegrid, trControl=fitControl, ntree=300,nodesize=5, Importance=TRUE,metric="ROC")

print(forest3)
plot(forest3)

#confusion matrix for test set
rf_test3<- predict(forest3, testDinh, type="raw")
confusionMatrix(rf_test3, testDinh$default, positive = "yes")


rf3_prob <- predict(forest3, testDinh, type = "prob")$"yes"
rf3_auc <-performance(prediction(rf3_prob, testDinh$default),"auc")
rf3_auc<-unlist(slot(rf3_auc, "y.values"))
rf3_auc

#forest3: auc  [1] 0.7880319

perf_rf3 <- prediction(rf3_prob, testDinh$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_rf3, col = "blue", lty = 2, main=" Random forest with mtry=c(8:10), ntrees=300, nodesize=10")
abline(a=0,b=1)



#---------------------------------#
# try to change the nodesize and tune for mtry from floor(sqrt(p)) to ceil(p/3)+1 where p is the number of features, 
#setting the number of trees to be 300 for running time

tunegrid <- expand.grid(.mtry = c(4:9))

rf_node2 <- list()

for (nodesize in c(1,5,10)){
  set.seed(42)
  rf_tune2 <- train(default~.,
                   data = trainDinh,
                   method = 'rf',
                   metric = 'ROC',
                   tuneGrid = tunegrid,
                   trControl = fitControl,
                   ntree = 300,
                   nodesize=nodesize)
  key <- toString(nodesize)
  rf_node2[[key]] <- rf_tune2
}

results22 <- resamples(rf_node2)
summary(results22)

dotplot(results22)

rf_node2[['1']]
rf_node2[['5']]
rf_node2[['10']]

rfnode2_1_prob <- predict(rf_node2[["1"]], testDinh, type = "prob")$"yes"
rfnode2_1_auc <-performance(prediction(rfnode2_1_prob, testDinh$default),"auc")
rfnode2_1_auc<-unlist(slot(rfnode2_1_auc, "y.values"))
rfnode2_1_auc
#rf_node[['1']] gives   0.7866337 auc on the test set

rfnode2_5_prob <- predict(rf_node2[["5"]], testDinh, type = "prob")$"yes"
rfnode2_5_auc <-performance(prediction(rfnode2_5_prob, testDinh$default),"auc")
rfnode2_5_auc<-unlist(slot(rfnode2_5_auc, "y.values"))
rfnode2_5_auc
#rf_node[['5']] gives   0.788137 auc on the test set

rfnode2_10_prob <- predict(rf_node2[["10"]], testDinh, type = "prob")$"yes"
rfnode2_10_auc <-performance(prediction(rfnode2_10_prob, testDinh$default),"auc")
rfnode2_10_auc<-unlist(slot(rfnode2_10_auc, "y.values"))
rfnode2_10_auc
#rf_node[['5']] gives   0.7873903 auc on the test set



#_________________________________________________________________________#
#_________________________________________________________________________#

## SVM ####
## Model 2 Ivan
svm_model<- svm(default ~ .,
                data=trainset, type="C-classification",
                kernel="linear", scale = TRUE, probability=TRUE)

pred_train <- predict(svm_model,newdata = trainset)
mean(pred_train==trainset$default)
summary(pred_train)
svm_model
pred_test <- predict(svm_model,newdata = testset, na.action = na.pass)
summary(pred_test)
predict_probability <- predict(svm_model,newdata = testset, na.action = na.pass, probability=TRUE)
summary(predict_probability)

svm_probabilties <- attr(predict_probability, "probabilities")[,1]
mean(pred_test==testset$default)


cfm_temp <- confusionMatrix(pred_test, testset$default, "yes")

d <- c("svm", 
  cfm_temp[["byClass"]][["Balanced Accuracy"]],
  cfm_temp[["byClass"]][["Precision"]],
  cfm_temp[["byClass"]][["Recall"]],
  cfm_temp[["byClass"]][["F1"]])

##AUC/ROC

roc_svm <- roc(testset$default,svm_probabilties, plot=TRUE)
plot(roc_svm, col="red", main="SVM ROC Curve")

a <- c("svm", auc_temp = auc(roc_svm))
a <- as.list(a)
names(a) <- names(auc)

auc <- rbind(auc, a)


#remove model variables
rm(d, cfm_temp)
rm(svm_probabilties, predict_probability, pred_test, pred_train, svm_model)

auc_svm <- auc(roc_svm)
auc_svm
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
AT2.glm_sig = glm(formula = default~ LIMIT_BAL+MARRIAGE+AGE+PAY_0 +PAY_2+PAY_AMT3,
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
cfm_temp <- confusionMatrix(testset$prediction, testset$default, "1")

d <- c("glm", 
       cfm_temp[["byClass"]][["Balanced Accuracy"]],
       cfm_temp[["byClass"]][["Precision"]],
       cfm_temp[["byClass"]][["Recall"]],
       cfm_temp[["byClass"]][["F1"]])

d <- as.list(d)

cfm <- rbind(cfm, d)



#AUC 

roc_glm <- roc(testset$default,testset$probability)
auc_glm <- auc(roc_glm)
auc_glm
plot(roc_glm, col="red", main="GLM ROC Curve")

a <- c("glm", auc_temp = auc(roc_glm))
a <- as.list(a)
names(a) <- names(auc)

auc <- rbind(auc, a)

#remove model variables
rm(d, cfm_temp, a)
rm(AT2.glm, AT2.glm_sig)

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
#Let's get our predictions, confusion matrix and auc
testset$predictions = predict(gbmFit1, newdata = testset)

testset$probability <- predict(gbmFit1, newdata = testset, type = "prob")

confusionMatrix(data = testset$predictions, reference = testset$default,
                mode = "everything", positive="yes")

pred_gbm1 = prediction(testset$probability[,2], testset$default)

#Let us look at the AUC
auc_gbm1 = performance(pred_gbm1, "auc")@y.values[[1]]
auc_gbm1


# fit using downsample
gbmFit2 <- train(default ~ ., data = training_dn, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 metric = "ROC",
                 tuneGrid = gbmGrid_v2)
gbmFit2
#Let's get our predictions, confusion matrix and auc
testset$predictions = predict(gbmFit2, newdata = testset)

testset$probability <- predict(gbmFit2, newdata = testset, type = "prob")

pred_gbm2 = prediction(testset$probability[,2], testset$default)

#Let us look at the AUC
auc_gbm2 = performance(pred_gbm2, "auc")@y.values[[1]]
auc_gbm2

gbm_prob_dn <- predict(gbmFit2, testset, type = "prob")$"yes"
roc_gbm_dn <- roc(testset$default,gbm_prob_dn, plot=TRUE)


# fit using upsample
gbmFit3 <- train(default ~ ., data = training_up, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE,
                 metric = "ROC",
                 tuneGrid = gbmGrid_v2)
gbmFit3


varImp(gbmFit3)
plot(varImp(gbmFit3), main="Variable importance - base GBM") 

#Let's get our predictions, confusion matrix and auc
testset$predictions = predict(gbmFit3, newdata = testset)

testset$probability <- predict(gbmFit3, newdata = testset, type = "prob")

pred_3 = prediction(testset$probability[,2], testset$default)

#Let us look at the AUC
auc_gbm3 = performance(pred_3, "auc")@y.values[[1]]
auc_gbm3

gbm_prob_up <- predict(gbmFit3, testset, type = "prob")$"yes"
roc_gbm_up <- roc(testset$default,gbm_prob_up, plot=TRUE)

testset$predictions = predict(gbmFit1, newdata = testset)

testset$probability <- predict(gbmFit1, newdata = testset, type = "prob")

pred_gbm1 = prediction(testset$probability[,2], testset$default)

#Let us look at the AUC
auc_gbm = performance(pred_gbm1, "auc")@y.values[[1]]
auc_gbm

gbm_prob <- predict(gbmFit1, testset, type = "prob")$"yes"

roc_gbm <- roc(testset$default,gbm_prob, plot=TRUE)

plot(roc_gbm, col="red", main="Base GBMs ROC Curves")
plot(roc_gbm_dn, col="orange", add=TRUE)
plot(roc_gbm_up, col="purple", add=TRUE)
legend("right", legend = c("unbalanced", "down-sampling", "up-sampling"), col = c("red", "orange","purple"), lty=1:1, box.lty=0)


#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Modelling Summary ####

## Confusion Matrix Results
cfm

## ROC and AUC of each model
plot(perf_rf3, col="red", main="Model ROC Curves")
plot(roc_svm, col="blue", add=TRUE)
plot(roc_glm, col="green", add=TRUE)
plot(roc_gbm, col="purple", add=TRUE)
legend("right", legend = c("forest", "svm", "glm", "gbm"), col = c("red", "blue", "green","purple"), lty=1:1, box.lty=0)

rf6_auc
auc_svm
auc_glm
auc_gbm

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

#both random search gives model with best ROC = n tree=50, depth=1, shrinkage=0.1, nminobsinnode=10

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



#------------------------------------------------------------------------------------#
#### Dinh tried gbm 
#tunning ntrees, interaction.depth

###gbm4

gbm_grid4 =  expand.grid(
  interaction.depth = c(5),
  n.trees = c(200), 
  shrinkage = c(0.1),
  n.minobsinnode = c(15)
)
set.seed(42)
gbm_4 = train( x = trainDinh[, -c(24)],  y = trainDinh$default, 
               method = "gbm", 
               trControl = fitControl, verbose = FALSE,
               tuneGrid= gbm_grid4, metric="ROC")

gbm_4


##gbm 4 ROC was used to select the optimal model using the largest value.
#The final values used for the model were n.trees = 200, interaction.depth = 5, shrinkage = 0.1
#and n.minobsinnode = 15.

gbm_grid3 =  expand.grid(
  interaction.depth = c(5),
  n.trees = c(500), 
  shrinkage = c(0.05),
  n.minobsinnode = c(5,10)
)

set.seed(42)
gbm_3 = train( x = trainDinh[, -c(24)],  y = trainDinh$default, 
               method = "gbm", 
               trControl = fitControl, verbose = FALSE,
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

gbm_grid2 =  expand.grid(
  interaction.depth = c(5),
  n.trees = c(300), 
  shrinkage = c(0.05),
  n.minobsinnode = c(15)
)
set.seed(42)
gbm_2 = train( x = trainDinh[, -c(24)],  y = trainDinh$default, 
               method = "gbm", 
               trControl = fitControl, verbose = FALSE,
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

#gbm2 gives 0.79088, gbm3 gives 0.79044, gmb4 gives 0.78...


# upsampling
set.seed(42)
final_Control <- trainControl(method = 'cv', number = 10, 
                              savePredictions = 'final', classProbs = TRUE, summaryFunction = twoClassSummary)
# v8 is our current best
gbm_grid_finalv8 =  expand.grid(
  interaction.depth = c(5),
  n.trees = c(350,450,550,650), 
  shrinkage = c(0.05),
  n.minobsinnode = c(10,15)
)

levels(df$default) <- c("no", "yes")

set.seed(42)
gbm_up = train( x = training_up[, -c(24)],  y = training_up$default, 
                   method = "gbm", 
                   trControl = final_Control, verbose = FALSE,
                   tuneGrid= gbm_grid_finalv8, metric="ROC")

gbm_up


#confusion matrix
p_gbm_up <-predict(gbm_up, df)
cfm_gbm_up <- confusionMatrix(data=p_gbm_up, reference=as.factor(df$default), positive="yes")
cfm_gbm_fnl

#plot roc curve
gbm_prob_up <- predict(gbm_up, df, type = "prob")$"yes"
perf_gbm_up <- prediction(gbm_prob_fnl, df$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_gbm_up, col = "blue", lty = 2, main="ROC curveusing gbm2")
abline(a=0,b=1)

#AUC
gbm_auc_up <-performance(prediction(gbm_prob_up, df$default),"auc")
gbm_auc_up<-unlist(slot(gbm_auc_up, "y.values"))
gbm_auc_up

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Final GBM Model ####

set.seed(42)
final_Control <- trainControl(method = 'cv', number = 10, 
                           savePredictions = 'final', classProbs = TRUE, summaryFunction = twoClassSummary)
# v8 is our current best
gbm_grid_finalv8 =  expand.grid(
  interaction.depth = c(5),
  n.trees = c(350,450,550,650), 
  shrinkage = c(0.05),
  n.minobsinnode = c(10,15)
)

levels(df$default) <- c("no", "yes")

set.seed(42)
gbm_final = train( x = df[, -c(24)],  y = df$default, 
               method = "gbm", 
               trControl = final_Control, verbose = FALSE,
               tuneGrid= gbm_grid_finalv8, metric="ROC")

gbm_final


#confusion matrix
p_gbm_fnl <-predict(gbm_final, df)
cfm_gbm_fnl <- confusionMatrix(data=p_gbm_fnl, reference=as.factor(df$default), positive="yes")
cfm_gbm_fnl

#plot roc curve
gbm_prob_fnl <- predict(gbm_final, df, type = "prob")$"yes"
perf_gbm_fnl <- prediction(gbm_prob_fnl, df$default) %>% performance(measure = "tpr", x.measure = "fpr")
plot(perf_gbm_fnl, col = "blue", lty = 2, main="ROC curveusing gbm2")
abline(a=0,b=1)

#AUC
gbm_auc_fnl <-performance(prediction(gbm_prob_fnl, df$default),"auc")
gbm_auc_fnl<-unlist(slot(gbm_auc_fnl, "y.values"))
gbm_auc_fnl

varImp(gbm_final)
plot(varImp(gbm_final), main="Variable importance - Final GBM") 

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Optimisation Summary ####
## AUC of each version of GBM models

## ROC and AUC of each model
plot(perf_gbm2, col="purple", main="Improving ROC Curves")
plot(perf_gbm3, col="blue", add=TRUE)
plot(perf_gbm_up, col="green", add=TRUE)
plot(perf_gbm_fnl, col="red", add=TRUE)
legend("right", legend = c("v2", "v3","upsampling","final"), col = c("purple", "blue","green", "red"), lty=1:1, box.lty=0)


gbm_auc3
gbm_auc2
gbm_auc_fnl

#_________________________________________________________________________#
#_________________________________________________________________________#
#_________________________________________________________________________#

# Output ####
# Prep the validation set
df_validation = read.csv("AT2_credit_test.csv")
levels(df$default) <- c("no", "yes") # prep for caret
levels(training_up$default) <- c("no", "yes") # prep for caret

df_validation$EDUCATION[df_validation$EDUCATION == 0] <- 4 # reducing class
df_validation$EDUCATION[df_validation$EDUCATION == 6] <- 5
df_validation$PAY_AMT1[df_validation$PAY_AMT1 != 0] <- 1
df_validation$PAY_AMT2[df_validation$PAY_AMT2 != 0] <- 1
df_validation$PAY_AMT3[df_validation$PAY_AMT3 != 0] <- 1
df_validation$PAY_AMT4[df_validation$PAY_AMT4 != 0] <- 1
df_validation$PAY_AMT5[df_validation$PAY_AMT5 != 0] <- 1
df_validation$PAY_AMT6[df_validation$PAY_AMT6 != 0] <- 1

# Define final categorical variables 
#df_validation$SEX <- as.factor(df_validation$SEX)
#df_validation$EDUCATION <- as.factor(df_validation$EDUCATION)
#df_validation$MARRIAGE <- as.factor(df_validation$MARRIAGE)
df_validation$PAY_AMT1 <- as.factor(df_validation$PAY_AMT1)
df_validation$PAY_AMT2 <- as.factor(df_validation$PAY_AMT2)
df_validation$PAY_AMT3 <- as.factor(df_validation$PAY_AMT3)
df_validation$PAY_AMT4 <- as.factor(df_validation$PAY_AMT4)
df_validation$PAY_AMT5 <- as.factor(df_validation$PAY_AMT5)
df_validation$PAY_AMT6 <- as.factor(df_validation$PAY_AMT6)


## Produce validation output
pred_prob <- predict(gbm_final, newdata = df_validation, type = "prob",predict.all=TRUE)
target_prob <- pred_prob$yes 

df_validation$default <- target_prob # add probabilities as default

#prepare only ID, target_probability, target_class for export
output_export <- df_validation %>% dplyr::select(ID, default)

# Export as csv
write.csv(output_export,"~/GitHub/mlaa_at2/MLAA_AT2_output_2005_v9.csv", row.names = FALSE)


##
