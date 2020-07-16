if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gbm)) install.packages("gbm")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(xgboost)) install.packages("xgboost")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(PRROC)) install.packages("PRROC")
if(!require(reshape2)) install.packages("reshape2")
if(!require(caTools)) install.package("caTools")
if(!require(smotefamily)) install.package("smotefamily")
if(!require(data.table))
  install.packages("data.table", repos = "http://cran.us.r-project.org")
if (!require(ROSE)) install.package("ROSE")

library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(ggplot2)
library(gbm)
library(caret)
library(xgboost)
library(e1071)
library(class)
library(ROCR)
library(randomForest)
library(PRROC)
library(reshape2)
library(rpart.plot)
library(caTools)
library(smotefamily)
library(ROSE)
#-----------------------------------------------------------------
#Loading dataset and preliminary analysis

credit_card <- read.csv('creditcard.csv')

str(credit_card)

head(credit_card)
credit_card$Class <- factor(credit_card$Class, levels = c(0,1))

summary(credit_card)

sum(is.na(credit_card))
#--------------------------------------------------------------

#get the distribution of fraud and legit transactions in the dataset
table(credit_card$Class)

#get the percentage of fraud and legit transactions in the dataset
prop.table(table(credit_card$Class))

#Pie chart of credit card transactions
labels <- c("legit", "fraud")
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)),2))
labels <- paste0(labels, "%")

pie (table(credit_card$Class), labels, col = c("blue","red"),
     main = "Pie Chart of Credit Card Transactions")
#-------------------------------------------------------------

#No model Predictions

predictions <- rep.int(0,nrow(credit_card))
predictions <- factor(predictions, levels = c(0,1))


confusionMatrix(data = predictions, reference = credit_card$Class)

#----------------------------------------------------------------------
#Use a subset for faster processing and hardware limitations


set.seed(1)
#sub_credit_card <- credit_card %>% sample_frac(0.1) #10% of the sample data set will be used
credit_sample = sample.split(credit_card$Class,SplitRatio=0.90) #Take 10% of the dataset

final_credit_sample = subset(credit_card,credit_sample == TRUE)
sub_credit_card = subset(credit_card,credit_sample == FALSE) #sub_credit_card will be 10% of the dataset

#sub_credit_card
table(sub_credit_card$Class)


ggplot(data = sub_credit_card, aes(x = V1, y = V2, col = Class)) +
  geom_point() + 
  ggtitle("Kalbo") +
  theme_bw() +
  scale_color_manual(values = c('blue', 'red'))
#--------------------------------------------------------------
#Creating training and test sets for Fraud Detection Model

set.seed(123)

data_sample = sample.split(sub_credit_card$Class,SplitRatio=0.80)

train_data = subset(sub_credit_card,data_sample == TRUE)
test_data = subset(sub_credit_card,data_sample == FALSE)

dim(train_data)
dim(test_data)

table(train_data$Class)
table(test_data$Class)

ggplot(data = train_data, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  ggtitle("Training Data Distribution") +
  theme_bw() +
  scale_color_manual(values = c('blue', 'red'))

summary(test_data)
table(test_data$Class)
ggplot(data = test_data, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  ggtitle("Test Data Distribution") +
  theme_bw() +
  scale_color_manual(values = c('blue', 'red'))
#--------------------------------------------------------------
#Random Over-Sampling (ROS)

n_legit <- 22745
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit

oversampling_result <- ovun.sample(Class ~., data = train_data, method = "over", N = new_n_total, seed = 2019)

oversample_credit <- oversampling_result$data

table(oversample_credit$Class)

ggplot(data = oversample_credit, aes(x = V1, y = V2, col = Class)) + 
  geom_point(position = position_jitter(width = 0.1)) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

#-------------------------------------------------------------------

#Random Under-Sampling(RUS)

n_fraud <- 39
new_frac_fraud <- 0.50
new_n_total <- n_fraud/new_frac_fraud

undersampling_result <- ovun.sample(Class ~ .,
                                    data = train_data,
                                    method = "under",
                                    N = new_n_total,
                                    seed = 2019)

undersampled_credit <- undersampling_result$data

table(undersampled_credit$Class)

ggplot(data = undersampled_credit,aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('blue','red'))

#---------------------------------------------------------------------------
#ROS and RUS

n_new <- nrow(train_data)
fraction_fraud_new <- 0.50

sampling_result <- ovun.sample(Class ~.,
                               data = train_data,
                               method = "both",
                               N = n_new,
                               p = fraction_fraud_new,
                               seed = 2019)

sampled_credit <- sampling_result$data

table(sampled_credit$Class)

ggplot(data = sampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.2)) +
  theme_bw() +
  scale_color_manual(values = c('blue', 'red'))
#--------------------------------------------------------------
#Using SMOTE to balance the dataset

table(train_data$Class)

# Set the number of fraud and legitimate cases and the desired percentage of legimate cases

n0 <- 22745
n1 <- 39
r0 <- 0.6

#Calculate the value for the dup_size parameter of SMOTE
ntimes <- ((1 - r0) / r0) * (n0 / n1) - 1

smote_output = SMOTE(X = train_data[, -c(1,31)],
                     target = train_data$Class,
                     K = 5,
                     dup_size = ntimes)

credit_smote <- smote_output$data	

colnames(credit_smote)[30] <- "Class"

prop.table(table(credit_smote$Class))

#Class distribution for original dataset
ggplot(credit_smote, aes(x = V1, y = V2, color = Class)) +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2','red'))


#---------------------------------------------------------
#Decision Tree with ROS

CART_ROS_model <- rpart(Class ~. , oversample_credit, model=TRUE)

rpart.plot(CART_ROS_model, extra = 0, type = 5, tweak = 1.2)

#Predict fraud classes
predicted_val <- predict(CART_ROS_model, test_data, type = 'class')

#build confusion matrix
confusionMatrix(predicted_val, test_data$Class)

#---------------------------------------------------------
#Decision Tree with RUS

CART_RUS_model <- rpart(Class ~. , undersampled_credit, model=TRUE)

rpart.plot(CART_RUS_model, extra = 0, type = 5, tweak = 1.2)

#Predict fraud classes
predicted_val <- predict(CART_RUS_model, test_data, type = 'class')

#build confusion matrix
confusionMatrix(predicted_val, test_data$Class)

#---------------------------------------------------------
#Decision Tree with ROS and RUS

CART_ROS_AND_RUS_model <- rpart(Class ~. ,sampled_credit, model=TRUE)

rpart.plot(CART_ROS_AND_RUS_model, extra = 0, type = 5, tweak = 1.2)

#Predict fraud classes
predicted_val <- predict(CART_ROS_AND_RUS_model, test_data, type = 'class')

#build confusion matrix
confusionMatrix(predicted_val, test_data$Class)


#---------------------------------------------------------
#Decision Tree with SMOTE

CART_SMOTE_model <- rpart(Class ~. , credit_smote, model=TRUE)

rpart.plot(CART_SMOTE_model, extra = 0, type = 5, tweak = 1.2)

#Predict fraud classes
predicted_val <- predict(CART_SMOTE_model, test_data, type = 'class')

#build confusion matrix
confusionMatrix(predicted_val, test_data$Class)

#---------------------------------------------------------------

#Decision Tree without SMOTE and without anything

CART_model <- rpart(Class ~ ., train_data[,-1], model=TRUE)

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

#predict fraud classes
predicted_val <- predict(CART_model, test_data[-1], type = 'class')

confusionMatrix(predicted_val, test_data$Class)

#-----------------------------------------------------------------------
### Approach 1 final assessments ####

#Final Assessments

#Vanilla
predicted_val <- predict(CART_model, final_credit_sample, type = 'class')
confusionMatrix(predicted_val, final_credit_sample$Class)

#ROS
predicted_val <- predict(CART_ROS_model, final_credit_sample, type = 'class')
confusionMatrix(predicted_val, final_credit_sample$Class)

#RUS
predicted_val <- predict(CART_RUS_model, final_credit_sample, type = 'class')
confusionMatrix(predicted_val, final_credit_sample$Class)

#ROS + RUS
predicted_val <- predict(CART_ROS_AND_RUS_model, final_credit_sample, type = 'class')
confusionMatrix(predicted_val, final_credit_sample$Class)


#SMOTE
predicted_val <- predict(CART_SMOTE_model, final_credit_sample, type = 'class')
confusionMatrix(predicted_val, final_credit_sample$Class)


######################################################
## At this point, we have multiple datasets from Analysis 1:
# credit_card = 100% of the dataset
# final_credit_sample = 90% of the dataset (Used for final analysis)
# sampled_credit = 10% of the dataset used for training
#
#
# train_data = 80% of sampled_credit
# test_data = 20% of sampled_credit
#
# oversample_credit = train_data balanced with ROS
# undersampled_credit = train_data balanced with RUS
# sampled_credit = train_data balanced with ROS and RUS
# credit_smote = train_data balanced with SMOTE
##
#####################################################

#------------------------------------------------------------------------------
#Approach #2, let's try to make use of our complete dataset this time around, instead of subsetting. Let's see how 
#Balancing the Dataset will work
#NOTE: This could take time

credit_card_copy <- credit_card #let's make a copy

#################################################VANILLA TECHNIQUE####################################################

set.seed(1)

#Split to 80 - 20 training and test again
credit_sample_vanilla = sample.split(credit_card_copy$Class,SplitRatio=0.80)

training_credit_vanilla = subset(credit_card_copy,credit_sample_vanilla == TRUE)
testing_credit_vanilla = subset(credit_card_copy,credit_sample_vanilla == FALSE) #sub_credit_card will be 10% of the dataset

table(training_credit_vanilla$Class)
table(testing_credit_vanilla$Class)

#Now let's see how the decision tree will go

CART_vanilla_full_model <- rpart(Class ~ ., training_credit_vanilla, model=TRUE)

rpart.plot(CART_vanilla_full_model, extra = 0, type = 5, tweak = 1.2)

#predict fraud classes
predicted_val <- predict(CART_vanilla_full_model, testing_credit_vanilla, type = 'class')

confusionMatrix(predicted_val, as.factor(testing_credit_vanilla$Class))

#################################################ROS TECHNIQUE####################################################
# Using ROS to balance the dataset
n_legit <- 284315
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit

oversampling_result <- ovun.sample(Class ~., data = credit_card_copy, method = "over", N = new_n_total, seed = 2019)

oversample_credit_complete <- oversampling_result$data

table(oversample_credit_complete$Class)


set.seed(1)

#Split to 80 - 20 training and test again
credit_sample_ros = sample.split(oversample_credit_complete$Class,SplitRatio=0.80)

training_credit_ros = subset(oversample_credit_complete,credit_sample_ros == TRUE)
testing_credit_ros = subset(oversample_credit_complete,credit_sample_ros == FALSE) #sub_credit_card will be 20% of the dataset

table(training_credit_ros$Class)
table(testing_credit_ros$Class)

#Now, let's create the model
CART_ros_full_model <- rpart(Class ~ ., training_credit_ros, model=TRUE)

rpart.plot(CART_ros_full_model, extra = 0, type = 5, tweak = 1.2)

#predict fraud classes
predicted_val <- predict(CART_ros_full_model, testing_credit_ros, type = 'class')

confusionMatrix(predicted_val, as.factor(testing_credit_ros$Class))


#################################################RUS TECHNIQUE######################################################


n_fraud <- 492
new_frac_fraud <- 0.50
new_n_total <- n_fraud/new_frac_fraud

undersampling_result <- ovun.sample(Class ~ .,
                                    data = credit_card_copy,
                                    method = "under",
                                    N = new_n_total,
                                    seed = 2019)

undersampled_credit_complete <- undersampling_result$data

table(undersampled_credit_complete$Class)

set.seed(1)

#Split to 80 - 20 training and test again
credit_sample_rus = sample.split(undersampled_credit_complete$Class,SplitRatio=0.80)

training_credit_rus = subset(undersampled_credit_complete,credit_sample_rus == TRUE)
testing_credit_rus = subset(undersampled_credit_complete,credit_sample_rus == FALSE) #sub_credit_card will be 20% of the dataset

table(training_credit_rus$Class)
table(testing_credit_rus$Class)

### Now to make the model ###

CART_RUS_model <- rpart(Class ~. , training_credit_rus, model=TRUE)

rpart.plot(CART_RUS_model, extra = 0, type = 5, tweak = 1.2)

#Predict fraud classes
predicted_val <- predict(CART_RUS_model, testing_credit_rus, type = 'class')

#build confusion matrix
confusionMatrix(predicted_val, testing_credit_rus$Class)

#################################################ROS + RUS TECHNIQUE################################################
n_new <- nrow(credit_card_copy)
fraction_fraud_new <- 0.50

sampling_result <- ovun.sample(Class ~.,
                               data = credit_card_copy,
                               method = "both",
                               N = n_new,
                               p = fraction_fraud_new,
                               seed = 2019)

sampled_credit_complete <- sampling_result$data

table(sampled_credit_complete$Class)

set.seed(1)

#Split to 80 - 20 training and test again
credit_sample_both = sample.split(sampled_credit_complete$Class,SplitRatio=0.80)

training_credit_both = subset(sampled_credit_complete,credit_sample_both == TRUE)
testing_credit_both = subset(sampled_credit_complete,credit_sample_both == FALSE)

table(training_credit_both$Class)
table(testing_credit_both$Class)

### Now to make the Model ###

CART_ROS_AND_RUS_model <- rpart(Class ~. ,training_credit_both, model=TRUE)

rpart.plot(CART_ROS_AND_RUS_model, extra = 0, type = 5, tweak = 1.2)

#Predict fraud classes
predicted_val <- predict(CART_ROS_AND_RUS_model, testing_credit_both, type = 'class')

#build confusion matrix
confusionMatrix(predicted_val, testing_credit_both$Class)

#################################################SMOTE TECHNIQUE####################################################

#--------------------------------------------------------------
#Using SMOTE to balance the dataset

# Set the number of fraud and legitimate cases and the desired percentage of legimate cases

n0 <- 284315
n1 <- 492
r0 <- 0.6

#Calculate the value for the dup_size parameter of SMOTE
ntimes <- ((1 - r0) / r0) * (n0 / n1) - 1

smote_complete_output = SMOTE(X = credit_card_copy[, -c(1,31)],
                     target = credit_card_copy$Class,
                     K = 5,
                     dup_size = ntimes)

credit_card_copy_smote <- smote_complete_output$data	

colnames(credit_card_copy_smote)[30] <- "Class"

table(credit_card_copy_smote$Class) #Our data should be more balanced now.
#------------------------------------------------------------
set.seed(1)

#Split to 80 - 20 training and test again
credit_sample_smote = sample.split(credit_card_copy_smote$Class,SplitRatio=0.80)

training_credit_smote = subset(credit_card_copy_smote,credit_sample_smote == TRUE)
testing_credit_smote = subset(credit_card_copy_smote,credit_sample_smote == FALSE) #sub_credit_card will be 10% of the dataset

table(training_credit_smote$Class)
table(testing_credit_smote$Class)

#Now let's see how the decision tree will go

CART_smote_full_model <- rpart(Class ~ ., training_credit_smote, model=TRUE)

rpart.plot(CART_smote_full_model, extra = 0, type = 5, tweak = 1.2)

#predict fraud classes
predicted_val <- predict(CART_smote_full_model, testing_credit_smote, type = 'class')

confusionMatrix(predicted_val, as.factor(testing_credit_smote$Class))

#--------------------------------------------------------------
#####################################
#
#
# Printing the table. Refer to the document
# and confusion matrixes (particularly the Sensitivy and Specifity)
# for values obtained
#
# P:S 
# True Negative Rate = Specificity
# True Positive Rate = Sensitivity / Recall
#####################################

subds_results <- tibble(Model = "Sub DS No Balancing", TNR = .6275, TPR = .9999, Kappa = 0.7133)
subds_results <- bind_rows(subds_results,                      
                           tibble(Model = "Sub DS ROS", TNR = .8375, TPR = .9809, Kappa = 0.1276))
subds_results <- bind_rows(subds_results,                      
                           tibble(Model = "Sub DS RUS", TNR = .8646, TPR = .9681, Kappa = 0.082))
subds_results <- bind_rows(subds_results,                      
                           tibble(Model = "Sub DS ROS+RUS", TNR = .8375, TPR = .9808, Kappa = 0.1268))
subds_results <- bind_rows(subds_results,                      
                           tibble(Model = "Sub DS SMOTE", TNR = .8578, TPR = .9902, Kappa = 0.2258))

fullds_results <- tibble(Model = "Full DS No Balancing", TNR = .7347, TPR = .9999, Kappa = 0.8266)
fullds_results <- bind_rows(fullds_results,                      
                           tibble(Model = "Full DS ROS", TNR = .9253, TPR = .9439, Kappa = 0.8692))
fullds_results <- bind_rows(fullds_results,                      
                           tibble(Model = "Full DS RUS", TNR = .8878, TPR = .8980, Kappa = 0.7857))
fullds_results <- bind_rows(fullds_results,                      
                           tibble(Model = "Full DS ROS+RUS", TNR = .9218, TPR = .9525, Kappa = 0.8743))
fullds_results <- bind_rows(fullds_results,                      
                           tibble(Model = "Full DS SMOTE", TNR = .9086, TPR = .9760, Kappa = 0.8929))


subds_results
fullds_results
