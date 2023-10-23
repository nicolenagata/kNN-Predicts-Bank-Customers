##  Installing Packages (KNN)
library(caret)

## 1. Loading the Data --------------------------------------------------------
bank <- read.csv("UniversalBank.csv", header = TRUE)

## 1a. Data exploration (before cleaning) -------------------------------------
head(bank, 10)  
str(bank)       # ID and Zipcode are not relevant        
names(bank)     
nrow(bank)  

## 1b. Cleaning Variables -----------------------------------------------------
bank <- bank[ , -c(1, 5)]   # Drop ID and zip code
names(bank)

bank <- bank[ , c(1:7, 9:12, 8)]  # Reorder variables, personal loan last (y)
head(bank)

# Set categorical variables as factor.
bank$Education <- as.factor(bank$Education)
bank$Securities.Account <- as.factor(bank$Securities.Account)
bank$CD.Account <- as.factor(bank$CD.Account) 
bank$Online <- as.factor(bank$Online) 
bank$CreditCard <- as.factor(bank$CreditCard) 
str(bank)

#Renaming Outcome variable (Y) - Personal Loan
bank$Personal.Loan <- factor(bank$Personal.Loan,
                             levels = c("0", "1"),
                             labels = c("No", "Yes"))

table(bank$Personal.Loan)

# Training and validation sets -------------------------------------

set.seed(666)

train_index <- sample(1:nrow(bank), 0.6 * nrow(bank))
valid_index <- setdiff(1:nrow(bank), train_index)

train <- bank[train_index, ]
valid <- bank[valid_index, ]

nrow(train)
nrow(valid)

# Normalize, only numerical variables
norm_values <- preProcess(train[, -c(6, 8:12)],
                          method = c("center",
                                     "scale"))
# Normalize training set
train_norm <- train
valid_norm <- valid

train_norm[, -c(12)] <- predict(norm_values,
                               train[, -c(12)])

head(train_norm)

#Normalize validation set
valid_norm[, -c(12)] <- predict(norm_values,
                               valid[, -c(12)])

head(valid_norm)

# 2. The kNN model (k=3) -----------------------------------------------------------

# Training the kNN model
knn_model_k3 <- caret::knn3(Personal.Loan ~ ., data = train_norm, k = 3)
knn_model_k3

# Predicting the training set
knn_pred_k3_train <- predict(knn_model_k3, newdata = train_norm[, -c(12)],
                          type = "class")
head(knn_pred_k3_train)

# Confusion Matrix on the training set
confusionMatrix(knn_pred_k3_train, as.factor(train_norm[, 12]),
                positive = "Yes")

" kNN model k3 Evaluation:
In training, the k3 kNN model works well due to:
  - high accuracy (97.53%)
  - high tpr (77.27%) and high tnr (9.85%)
      - relatively similar tpr and tnr
  - high percision (98.35%) and high percision of neg class (97.46%)
      - very close percision "

# The kNN model (k=5) -----------------------------------------------------------

# Training the kNN model
knn_model_k5 <- caret::knn3(Personal.Loan ~ ., data = train_norm, k = 5)
knn_model_k5

# Predicting the training set
knn_pred_k5_train <- predict(knn_model_k5, newdata = train_norm[, -c(12)],
                             type = "class")
head(knn_pred_k5_train)

# Confusion Matrix on the training set
confusionMatrix(knn_pred_k5_train, as.factor(train_norm[, 12]),
                positive = "Yes")

" kNN model k5 Evaluation:
In training, the k5 kNN model works well due to:
  - high accuracy (96.67%)
  - high tpr (69.16%) and high tnr (99.81%)
      - tpr and tnr are slightly varied
  - high percision (97.71%) and high percision of neg class (96.59%)
      - very close percision "

# The kNN model (k=7) -----------------------------------------------------------

# Training the kNN model
knn_model_k7 <- caret::knn3(Personal.Loan ~ ., data = train_norm, k = 7)
knn_model_k7

# Predicting the training set
knn_pred_k7_train <- predict(knn_model_k7, newdata = train_norm[, -c(12)],
                             type = "class")
head(knn_pred_k7_train)

# Confusion Matrix on the training set
confusionMatrix(knn_pred_k7_train, as.factor(train_norm[, 12]),
                positive = "Yes")

" kNN model k7 Evaluation:
In training, the k7 kNN model works well due to:
  - high accuracy (95.87%)
  - high tpr (60.06%) and high tnr (99.96%)
      - tpr and tnr are slightly varied
  - high percision (99.46%) and high percision of neg class (95.63%)
      - pretty close percision "

# 2. What is the best choice for k ---------------------------------------------
" k = 3 is the best choice because of the high accuracy (97.53%), high and 
similar tpr (77.27%) /tnr (99.85%), and high and similiar percision (89.73%)/ 
percision of the negative class (97.46%). Other values of k had similar results 
but somtimes had variation of tnp and tnr, therefore k = 3 is the best 
overall choice to perdict the validation set."

# 3 Predictions on the Validation Set ------------------------------------------
# Predicting the validation set
knn_pred_k3_valid <- predict(knn_model_k3, newdata = valid_norm[, -c(12)],
                             type = "class")
head(knn_pred_k3_valid)

# Confusion Matrix on the validation set 
confusionMatrix(knn_pred_k3_valid, as.factor(valid_norm[, 12]),
                positive = "Yes")

#ROC Curve Evaluation
library(ROSE)

ROSE::roc.curve(valid_norm$Personal.Loan, 
                knn_pred_k3_valid)

" The area under the ROC curve is 0.787, which means this model is pretty good 
at distinguishing whether a customer will accept or not accpet the loan."


# 3b. How good is the model? ---------------------------------------------------
" The model is pretty good at predicting if customers will not accept a personal
loan. However, it is important to consider that most customers at Universal 
Bank will not accept a personal loan. The model has a 57.56% rating for how well
the model predicts customers will accept a personal loan. This is not the
best model becuase it does a better job at predicting when customers wont accept
the loan over when customers will accept the loan, but it is a better chance 
than simply guessing."

# 4. New customer --------------------------------------------------------------
new_cust <- data.frame(Age = 40,
                       Experience = 10,
                       Income = 84,
                       Family = 2,
                       CCAvg = 2,
                       Education = 2,
                       Mortgage = 0,
                       Securities.Account = 0,
                       CD.Account = 0,
                       Online = 1,
                       CreditCard = 1)

# Set categorical variables as factor.
new_cust$Education <- as.factor(new_cust$Education)
new_cust$Securities.Account <- as.factor(new_cust$Securities.Account)
new_cust$CD.Account <- as.factor(new_cust$CD.Account) 
new_cust$Online <- as.factor(new_cust$Online) 
new_cust$CreditCard <- as.factor(new_cust$CreditCard) 
new_cust

#predict normalized values for new customer
new_cust_norm <- predict(norm_values, new_cust)
new_cust_norm

# prediction for new customer
new_cust_predict <- predict(knn_model_k3, newdata = new_cust_norm,
                           type = "prob")
new_cust_predict

" The new customer will not accept the personal loan."