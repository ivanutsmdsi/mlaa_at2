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


## Testing on the credit_sample file
validset <- read.csv("AT2_credit_test.csv")

# Cleaning data
summary(validset$LIMIT_BAL) ## no need to clean
table(validset['SEX'])  ## no need to clean
summary(validset$MARRIAGE)
validset$MARRIAGE[validset$MARRIAGE == 0] <- 3 # removed 0 from marriage values
table(validset['MARRIAGE']) 
summary(validset$AGE)

## remove entries with bad limit balance
summary(df$LIMIT_BAL)
df <- subset(df, df$LIMIT_BAL > 0)
summary(df$MARRIAGE)
df <- subset(df, df$MARRIAGE == 1 | df$MARRIAGE == 2 | df$MARRIAGE == 3)
summary(df$AGE)
df <- subset(df, df$AGE < 100)

## create PAY_DELAYS category
df$PAY_DELAYS <- case_when(df$PAY_0 > 0 | 
                             df$PAY_2 > 0 | 
                             df$PAY_3 > 0 |
                             df$PAY_4 > 0 |
                             df$PAY_5 > 0 |
                             df$PAY_6 > 0 ~ 1,
                           TRUE ~ 0)

validset$PAY_DELAYS <- case_when(validset$PAY_0 > 0 | 
                                   validset$PAY_2 > 0 | 
                                   validset$PAY_3 > 0 |
                                   validset$PAY_4 > 0 |
                                   validset$PAY_5 > 0 |
                                   validset$PAY_6 > 0 ~ 1,
                           TRUE ~ 0)


df$PAY_AVG <- rowMeans(subset(df, select = c(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)), na.rm = TRUE)

## Default value (convert to factor)                                    ----
df$default  <- as.factor(df$default)
summary(df$default)

## Looking at SVM                                                       ----
set.seed(1234)


# Train/Test Split
trainset_size <- floor(0.80 * nrow(df))

trainset_indices <- sample(seq_len(nrow(df)), size = trainset_size)


trainset <- df[trainset_indices, ]
testset <- df[-trainset_indices, ]



## Build and test SVM model

svm_model<- 
  svm(default ~ LIMIT_BAL + MARRIAGE + AGE + PAY_DELAYS + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6
      , data=trainset, type="C-classification", kernel="linear", scale = TRUE, probability=TRUE)

pred_train <- predict(svm_model,newdata = trainset)
mean(pred_train==trainset$default)

pred_test <- predict(svm_model,testset)
table(pred_test)
mean(pred_test==testset$default)

#  MARRIAGE + PAY_DELAYS + PAY_X = 0.7730035
# LIMIT_BAL + MARRIAGE + AGE + PAY_DELAYS + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 = 0.7727865

## MARRIAGE
p <- ggplot(df, aes(fill=default, x=MARRIAGE)) +
  geom_bar(position = "dodge")
p

##LIMIT BAL
p <- ggplot(df, aes(fill=default, x=LIMIT_BAL)) +
  geom_histogram(position = "dodge")
p

##AGE
p <- ggplot(df, aes(fill=default, x=AGE)) +
  geom_histogram(position = "dodge")
p

## PAY DELAY
p <- ggplot(df, aes(fill=default, x=PAY_DELAYS)) +
  geom_histogram(position = "dodge")
p
p <- ggplot(df, aes(fill=default, x=PAY_AVG)) +
  geom_histogram(position = "dodge")
p

## Train model on whole dataframe
svm_model<- 
  svm(default ~ LIMIT_BAL + MARRIAGE + AGE + PAY_DELAYS + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6
      , data=df, type="C-classification", kernel="linear", scale = TRUE, probability=TRUE)

# run SVM model against the validation set
predict_probability <- predict(svm_model,newdata = validset, na.action = na.pass, probability=TRUE)

validset$probabilties <- attr(predict_probability, "probabilities")[,"Y"]

output <- validset %>%
  select(ID, probabilties)

output <- rename(output, default = probabilties)

write.csv(output, "predictions.csv", row.names = FALSE)
