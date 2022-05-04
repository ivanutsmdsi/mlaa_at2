## Import Libraries
library(ggplot2)
library(dplyr)
library(e1071)
library(pROC)
library(ROCR)

## Clear history
rm(list = ls())

## Load data
df <- read.csv('AT2_credit_train.csv')
df_raw <- df

## Data Cleaning / EDA
summary(df)

# Limit Balance - looking at values breakdown ------
p<-ggplot(data=df, aes(x=LIMIT_BAL)) +
  geom_histogram()
p

p1<-ggplot(data=df, aes(x=LIMIT_BAL)) +
  geom_histogram() +
  ylim(0,100)
p1

p2<-ggplot(data=df, aes(x=LIMIT_BAL)) +
  geom_histogram() +
  ylim(0,10)
p2

p3<-ggplot(data=df, aes(x=LIMIT_BAL)) +
  geom_histogram() +
  xlim(-1000,0)
p3


## Remove negative limits                                               ----
df$LIMIT_BAL[df$LIMIT_BAL < 0] <- 0
summary(df$LIMIT_BAL)


## SEX - look at values                                                 ----
p<-ggplot(data=df, aes(x=SEX)) +
  geom_histogram(stat="count")
p

# Histogram output shows several invalid values
# How to clean? - replace invalid values with NULL
## Clean invalid SEX values and convert to number instead of character  ----
df$SEX[df$SEX != 1 & df$SEX != 2] <- 3

table(df_raw['SEX'])
table(df['SEX'])
# Confirm that NULL replacement worked
sum(is.na(df$SEX))

## Marriage                                                             ----
summary(df$MARRIAGE)
p<-ggplot(data=df, aes(x=MARRIAGE)) +
  geom_histogram()
p

# 0 value for marriage is invalid - replace with 3 (others)
df$MARRIAGE[df$MARRIAGE == 0] <- 3
table(df['MARRIAGE'])

## Education                                                            ----
summary(df$EDUCATION)
p<-ggplot(data=df, aes(x=EDUCATION)) +
  geom_histogram()
p

# Group invalid and two unknown values together (0,5 and 6)
df$EDUCATION[df$EDUCATION == 0] <- 5
df$EDUCATION[df$EDUCATION == 6] <- 5

## AGE                                                                  ----
summary(df$AGE)

p<-ggplot(data=df, aes(x=AGE)) +
  geom_histogram()
p
p1<-ggplot(data=df, aes(x=AGE)) +
  geom_histogram() +
  ylim(0,2000)
p1

# Band ages into groups (21-30 = 1, 31-40 = 2, 41-50 = 3, 51+ = 4)      ----
df$AGE[df$AGE <= 30] <- 1
df$AGE[df$AGE > 30 & df$AGE <= 40] <- 2
df$AGE[df$AGE > 40 & df$AGE <= 50] <- 3
df$AGE[df$AGE > 50 & df$AGE <= 60] <- 4
df$AGE[df$AGE > 60 & df$AGE <= 70] <- 5
df$AGE[df$AGE > 70] <- 6
p<-ggplot(data=df, aes(x=AGE)) +
  geom_histogram()
p

## PAY_X                                                                ----
summary(df$PAY_6)
p<-ggplot(data=df, aes(x=PAY_6)) +
  geom_histogram()
p



## Default value (convert to factor)                                    ----
df$default  <- as.factor(df$default)
summary(df$default)

## Create new data -> number of payment details last 6 months
df$PAY_DELAYS <- case_when(df$PAY_0 > 0 | 
                            df$PAY_2 > 0 | 
                             df$PAY_3 > 0 |
                             df$PAY_4 > 0 |
                             df$PAY_5 > 0 |
                             df$PAY_6 > 0 ~ 1,
                           TRUE ~ 0)

# df$DELAYED_SCORE <- df$PAY_0 + df$PAY_2 + df$PAY_3 + df$PAY_4 + df$PAY_5 + df$PAY_6
# df$PAY_AVG <- rowMeans(subset(df, select = c(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)), na.rm = TRUE)
# df$MAX_DELAY <- pmax(df$PAY_0,df$PAY_2, df$PAY_3 , df$PAY_4 , df$PAY_5 , df$PAY_6)
# df$BILL_PAY <- (df$PAY_AMT1 + df$PAY_AMT2 + df$PAY_AMT3 +df$PAY_AMT4 + df$PAY_AMT5 + df$PAY_AMT6 )-(df$BILL_AMT1 + df$BILL_AMT2 + df$BILL_AMT3 + df$BILL_AMT4 + df$BILL_AMT5 + df$BILL_AMT6)

## Looking at SVM                                                       ----
set.seed(1234)


# Train/Test Split
trainset_size <- floor(0.80 * nrow(df))

trainset_indices <- sample(seq_len(nrow(df)), size = trainset_size)


trainset <- df[trainset_indices, ]
testset <- df[-trainset_indices, ]

head(df)


## Build and test SVM model

svm_model<- 
  svm(default ~ MARRIAGE + PAY_DELAYS + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6
      , data=trainset, type="C-classification", kernel="linear", scale = TRUE, probability=TRUE)

pred_train <- predict(svm_model,newdata = trainset)
mean(pred_train==trainset$default)

pred_test <- predict(svm_model,testset)
table(pred_test)
mean(pred_test==testset$default)

# PAY_X only = 0.7613071
# PAY_X + LIMIT BAL =             0.7610907
# PAY_X + LIMIT BAL + AGE =       0.7600087
# PAY_X + LIMIT BAL + SEX =       0.7608743
# PAY_X + LIMIT BAL + EDUCATION = 0.7608743
# PAY_X + LIMIT BAL + MARRIAGE =  0.7634711
# PAY_X + LIMIT_BAL + MARRIAGE + PAY_DELAYS = 0.7658515
# PAY_X + LIMIT_BAL + MARRIAGE + PAY_DELAYS + DELAY_SCORE = 0.7645531
# PAY_X + LIMIT_BAL + MARRIAGE + PAY_DELAYS + MAX_DELAY = 0.7507033
# LIMIT_BAL + MARRIAGE + PAY_DELAYS + MAX_DELAY + DELAY_SCORE = 0.7539494
#  MARRIAGE + PAY_DELAYS + PAY_X = 0.7719108
#  MARRIAGE + AGE + PAY_X = 0.7636875

# LINEAR = 0.7719108
# 




predict_probability <- predict(svm_model,newdata = testset, na.action = na.pass, probability=TRUE)


testset$probabilties <- attr(predict_probability, "probabilities")[,"Y"]

## Testing on the credit_sample file
validset <- read.csv("AT2_credit_test.csv")

# Cleaning data
summary(validset$LIMIT_BAL) ## no need to clean
table(validset['SEX'])  ## no need to clean
summary(validset$MARRIAGE)
validset$MARRIAGE[validset$MARRIAGE == 0] <- 3 # removed 0 from marriage values
table(validset['MARRIAGE']) 
summary(validset$AGE)
validset$AGE[validset$AGE <= 30] <- 1                       ##grouped ages into bands
validset$AGE[validset$AGE > 30 & validset$AGE <= 40] <- 2
validset$AGE[validset$AGE > 40 & validset$AGE <= 50] <- 3
validset$AGE[validset$AGE > 50] <- 4

# run SVM model against the validation set
predict_probability <- predict(svm_model,newdata = validset, na.action = na.pass, probability=TRUE)

validset$probabilties <- attr(predict_probability, "probabilities")[,"Y"]

output <- validset %>%
  select(ID, probabilties)

output <- rename(output, default = probabilties)

write.csv(output, "predictions.csv", row.names = FALSE)

## AUC

