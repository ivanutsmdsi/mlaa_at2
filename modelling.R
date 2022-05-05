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

## Libraries
library(ggplot2)
library(dplyr)

## Load Data - Ivan
rm(list = ls())
df <- read.csv('AT2_credit_train.csv')
df_raw <- df


## EDA
# LIMIT_BAL
p<-ggplot(data=df, aes(x=LIMIT_BAL)) +
  geom_histogram()
p

# SEX
p<-ggplot(data=df, aes(x=SEX)) +
  geom_histogram(stat="count")
p

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
df <- subset(df, df$LIMIT_BAL > 0)

# remove observations with sex != 1 or 2
# obs removed = 0 -- The obs with invalid sex entries were picked up in the previous cleaning step
df <- subset(df, df$SEX == 1 | df$SEX == 2)

# remove observations with marriage != 1,2 or 3
# obs removed = 45
df <- subset(df, df$MARRIAGE == 1 | df$MARRIAGE == 2 | df$MARRIAGE == 3)

# remove observations with age > 75
# obs removed = 2
df <- subset(df, df$AGE <= 75)

# remove observation with education not in 1,2,3,4,5,6
# obs removed = 12
df <- subset(df, df$EDUCATION >= 1 & df$EDUCATION <= 6)
table(df$EDUCATION)

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

## Build Train and Test Set                                                                     ----
set.seed(20220504)

df_n <- subset(df, df$default == 'N')
df_y <- subset(df, df$default == 'Y')
train_n_size <- floor(0.80 * nrow(df_n))
train_n_indices <- sample(seq_len(nrow(df_n)), size = train_n_size)
train_y_size <- floor(0.80 * nrow(df_y))
train_y_indices <- sample(seq_len(nrow(df_y)), size = train_y_size)


train_n <- df_n[train_n_indices, ]
test_n <- df_n[-train_n_indices, ]
train_y <- df_y[train_y_indices, ]
test_y <- df_y[-train_y_indices, ]

trainset <- rbind(train_n,train_y)
testset <- rbind(test_n, test_y)

rm(train_n_indices, train_n_size, train_n, df_n, df_y, train_n, test_n, train_y, test_y, train_y_indices, train_y_size)

# Validation of train and test set
table(trainset$default)
table(testset$default)
nrow(trainset) + nrow(testset)
nrow(df)

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
