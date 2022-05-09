###
#
# MLAA Assignment 2 Part A
#
# Students:
# Ivan Cheung - 13975420
# Ryan Yeo - 14328254
# Dinh Tran -
# John Rho - 24509337
#
#
#
#
###

## Libraries
#install.packages("AMR")
library(dplyr)
library(caret)
library(gbm)
library(parallel)
library(magrittr)
library(AMR)
library(ROCR)
library(ggplot2)


## Load Data - Ivan
rm(list = ls())
df <- read.csv('AT2_credit_train.csv')
df_raw <- df


## EDA
######### Ryan
str(df)
df$LIMIT_BAL <-  as.integer(df$LIMIT_BAL)
df$ID <- NULL # drop ID
df %>% summarise_all(n_distinct) 
# SEX has 6 unique values - should be only 2
# Education has 7 values - should be 6,
# Marriage has 4, should be 3 only
# PAY_AMT is binary?

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

# LIMIT_BAL
p<-ggplot(data=df, aes(x=LIMIT_BAL)) +
  geom_histogram()
p # check

# SEX
p<-ggplot(data=df, aes(x=SEX)) +
  geom_histogram(stat="count")
p # # SEX contains wrong values

# MARRIAGE
p<-ggplot(data=df, aes(x=MARRIAGE)) +
  geom_histogram()
p

# AGE
p<-ggplot(data=df, aes(x=AGE)) +
  geom_histogram()
p

# EDUCATION
p<-ggplot(data=df, aes(x=EDUCATION)) +
  geom_histogram()
p

## Data Cleaning                                                                                ----

# remove observations with limit bal less than 0
# obs removed = 19
df <- subset(df, df$LIMIT_BAL > 0) ####(COMPARE RESULTS with vs without)

# remove observations with sex != 1 or 2
# obs removed = 0 -- The obs with invalid sex entries were picked up in the previous cleaning step
df <- subset(df, df$SEX == 1 | df$SEX == 2)

# remove observations with marriage != 1,2 or 3
# obs removed = 45
df <- subset(df, df$MARRIAGE == 1 | df$MARRIAGE == 2 | df$MARRIAGE == 3) 
####(COMPARE RESULTS 0+3 vs 3)

# remove observations with age > 75
# obs removed = 2
df <- subset(df, df$AGE <= 75)

# remove observation with education not in 1,2,3,4,5,6
# obs removed = 12
df <- subset(df, df$EDUCATION >= 1 & df$EDUCATION <= 6) ## There isn't any class above 6?
table(df$EDUCATION)

# we can just to below
#df$EDUCATION[df$EDUCATION == 0] <- 4 # reducing class
#df$EDUCATION[df$EDUCATION == 6] <- 5


# total obs removed from raw dataset = 78 (0.33% of raw data removed)

# Reduce PAY_AMT columns to 0 and 1
df$PAY_AMT1[df$PAY_AMT1 > 0] <- 1
df$PAY_AMT2[df$PAY_AMT2 > 0] <- 1
df$PAY_AMT3[df$PAY_AMT3 > 0] <- 1
df$PAY_AMT4[df$PAY_AMT4 > 0] <- 1
df$PAY_AMT5[df$PAY_AMT5 > 0] <- 1
df$PAY_AMT6[df$PAY_AMT6 > 0] <- 1

## Build additional factors

# Age Band
df$AGE_BAND[df$AGE <= 30] <- 1
df$AGE_BAND[df$AGE > 30 & df$AGE <= 40] <- 2
df$AGE_BAND[df$AGE > 40 & df$AGE <= 50] <- 3
df$AGE_BAND[df$AGE > 50 & df$AGE <= 60] <- 4
df$AGE_BAND[df$AGE > 60 & df$AGE <= 70] <- 5
df$AGE_BAND[df$AGE > 70] <- 6

p<-ggplot(data=df, aes(x=AGE_BAND)) +
  geom_histogram()
p
table(df$AGE_BAND)

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

## Build Train and Test Set                                                                     ----

# Split data into testing and training with 75% for training on stratified method
#trainIndex = createDataPartition(y = df$default, p = 0.75, list = F)
#training_df = df[trainIndex, ]
#testing_df = df[-trainIndex, ]
# easier this way?

# df_n <- subset(df, df$default == 'N')
# df_y <- subset(df, df$default == 'Y')
# train_n_size <- floor(0.80 * nrow(df_n))
# train_n_indices <- sample(seq_len(nrow(df_n)), size = train_n_size)
# train_y_size <- floor(0.80 * nrow(df_y))
# train_y_indices <- sample(seq_len(nrow(df_y)), size = train_y_size)
# 
# 
# train_n <- df_n[train_n_indices, ]
# test_n <- df_n[-train_n_indices, ]
# train_y <- df_y[train_y_indices, ]
# test_y <- df_y[-train_y_indices, ]
# 
# trainset <- rbind(train_n,train_y)
# testset <- rbind(test_n, test_y)
# 
# rm(train_n_indices, train_n_size, train_n, df_n, df_y, train_n, test_n, train_y, test_y, train_y_indices, train_y_size)
# 
# # Validation of train and test set
# table(trainset$default)
# table(testset$default)
# nrow(trainset) + nrow(testset)
# nrow(df)


#Dinh: REWORTE THE CODE ABOVE TO SPLIT INTO TRAINSET AND TESTSET USING CARET PACKAGE
set.seed(20220504)
intrain<-createDataPartition(y=df$default,p=0.8,list=FALSE)
trainset<-df[intrain,]
testset<-df[-intrain,]

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
training_up <- upSample(x=training_df[,-ncol(training_df)],
                        y= training_df$default)
str(training_up)
colnames(training_up)[24] <- "default"
table(training_up$default)

## downsampling
training_dn <- downSample(x=training_df[,-ncol(training_df)],
                          y= training_df$default)
str(training_dn)
colnames(training_dn)[24] <- "default"
table(training_dn$default)



#-------------------------------------------####---------------------------#
### Model Analysis

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

#### Dinh tried gbm ######################
### GBM Model #####
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
# As a table
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

#----------------------------------------------##---------------------------------------------------#

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
############ Train GBM model

fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10,
  summaryFunction=twoClassSummary, classProbs=T,
  savePredictions = T
)

gbmGrid <-  expand.grid(interaction.depth = 5, 
                        n.trees = 100, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)


gbmFit1 <- train(default ~ ., data = training_df, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE,
                 metric = "ROC",
                 tuneGrid = gbmGrid)
gbmFit1

# downsample
gbmFit2 <- train(default ~ ., data = training_dn, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 metric = "ROC",
                 tuneGrid = gbmGrid)
gbmFit2

#upsample
gbmFit3 <- train(default ~ ., data = training_up, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE,
                 metric = "ROC",
                 tuneGrid = gbmGrid)
gbmFit3


varImp(gbmFit3)

#Let's get our predictions, confusion matrix and auc
testing_df$predictions = predict(gbmFit3, newdata = testing_df)
#Let us check the confusion matrix
confusionMatrix(data = testing_df$predictions, reference = testing_df$default,
                mode = "everything", positive="yes")

#Note that the ROCR package offers some great metrics by using the 'prediction' function
testing_df$probability <- predict(gbmFit3, newdata = testing_df, type = "prob")

pred = prediction(testing_df$probability[,2], testing_df$default)

#Let us look at the AUC
auc = performance(pred, "auc")@y.values[[1]]
auc


### Model Evaluations

## Confusion Matrix
## AUC of each model
## ROC

### Model Optimisations

## Final Model - GBM

## Final Model - Random Forest

## Produce validation output

##
