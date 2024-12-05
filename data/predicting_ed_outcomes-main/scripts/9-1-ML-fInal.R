#ML Final
#a note for education administrators, Analyses like this are cool, but they are only possible if data is collected. My strongest recommendation is "mandatory reporting in order to attend". In other words, 
#students cannot attend the first day of class until a parent or guardian submit important information like the variables found within this dataset. However, should this not be possible, 
#a lot of this information is accessible via the student and a few different administrative offices. 
library(coefplot)
library(caret)
library(dplyr)
library(glmnet)
library(ggplot2)
library(rvest)
library(tidyverse)
library(stringr)
library(tibble)




################################################################################
# DOWNLOAD AND CLEAN DATA

#url = Higher ED = https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

# Step 1: get working directory
getwd()
setwd("/Users/kevinmcgowan/projects")
current_dir<- getwd()

# Step 2: Set the file path relative to the working directory
file_path <- file.path(current_dir, "data 2.csv")

# Step 3: Load the dataset using the file path
data2 <- read.csv(file_path, sep = ";", header = TRUE)

# Verify the data is loaded correctly
str(data2)
View(data2)

# Convert Target to a factor and mutate the original data_2 data frame
data2 <- data2 %>%
  mutate(Target = as.factor(Target))

# Check the structure to confirm that Target is now a factor
str(data2$Target)
################################################################################
# REDO ABOVE SO ANYONE WHO DOWNLOADS THE DATASET CAN RUN THIS CODE
################################################################################
# URL of the webpage
url <- "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"

##################################################################################
#ALL THE ABOVE SHOULD BE DONE IN EXCEL
################################################################################
# PARTITION DATA
#set up test and train sets
# for reproducibility
set.seed(123)  
trainIndex <- createDataPartition(data2$Target, p = .8, 
                                  list = FALSE, 
                                  times = 1)
trainData <- data2[trainIndex,]
testData  <- data2[-trainIndex,]

################################################################################
# EXPLORITORY DATA ANALYSIS

# Inspect data for number of rows to understand how many students will be part of the analysis
nrow(trainData)

#demographic analysis
unique(trainData$Nacionality)

#evaluate Target AKA graduated, enrolled, or dropout
table(trainData$Target)
# Create a value formula 
# Extract the variable names
variables_train <- colnames(trainData)

# Remove the 'Target' variable from the list of predictors
variables_train <- variables_train[variables_train != "Target"]

# Create the formula as a string
formula_string_train <- paste("Target ~", paste(variables_train, collapse = " + "))#, "- 1")  don't include or do?

# Convert the string to a formula object
formula_train <- as.formula(formula_string_train)
print(formula_train)

#linear regression
value1<- lm(formula_train, data = trainData)
value1

#visualize
coefplot(value1, sort='magnitude', conf.int = TRUE)



################################################################################
# FEATURE DEVELOPMENT








################################################################################
# LINEAR REGRESSION
#DO THIS AGAIN AFTER FEATURE DEVELOPMENT

#create value formula FOR TRAINING
# Extract the variable names
variables_train <- colnames(trainData)

# Remove the 'Target' variable from the list of predictors
variables_train <- variables_train[variables_train != "Target"]

# Create the formula as a string
formula_string_train <- paste("Target ~", paste(variables_train, collapse = " + "))#, "- 1")  don't include or do?

# Convert the string to a formula object
formula_train <- as.formula(formula_string_train)
print(formula_train)



################################################################################
colnames(data2)
################################################################################









################################################################################
#linear regression
value1<- lm(formula_train, data = trainData)
value1

#visualize
coefplot(value1, sort='magnitude', conf.int = TRUE)


#evaluate Target
table(trainData$Target)

################################################################################









############################################################################################
############################################################################################
#trying glm multinomial algorethm:
#goal of shrinkage is to find a linear regression where the coefficicents have been shrunk towards zero (bayseians will call this shrinkage) LAnders call this an elastic net formula also called a cost function
#this is constrained optimization
valueX <- build.X(formula_train, data = trainData,
                       contrasts = FALSE, sparse = TRUE)

valueY <- build.y(formula_train, data = trainData)





###########above are replaced with the following###############################
# For creating the design matrix (equivalent to build.x)
valueX <- model.matrix(formula_train, data = trainData)

# For creating the response vector (equivalent to build.y)
valueY <- model.response(model.frame(formula_train, data = trainData))



head(as.matrix(valueX)) #just to see
############################################################################################









############################################################################################
# Run the glmnet model
value2 <- glmnet(x = valueX, y = valueY, family = 'multinomial')

# Print summary of the model
print(value2) #multinomial because Target has 3 levels
#this above just fit about a hundred models. #glemnet will automatically scale x's for you

plot(value2, xvar='lambda')#WOULD BE COOL TO SEE THE VARIABLE NAMES ON THIS PLOT


#each line is a different coeficicent over it's life time
#x-axis is lamda on log scale
#if lamda is small all coefificents in model
#as lamda is bigger fewer in model, then all in zero
##top tells you how many variables are included at that point of lambda

coefpath(value2) #doesn't work  ##should see interactive of above with proper labels
################################################################################################









#################################################################################################
#################################################################################################









##################################################################################################
#CROSS VALIDATION TIME LASSO
#now we need to build cross validation to optimmize the lambda
#caret package deos this and is awesome, but we don't even need it
value3 <- cv.glmnet(x = valueX, y=valueY,
                    family='multinomial',
                    nfolds=5) #cv.

plot(value3) #x-axis shows small lamda to big lambda. Y-axis is error or cross validated mean square error. One line is the minimum error you get (which is good)
#second line is for the group that says minimum error isnt the best.They say take the minimum error that is at the max of the confidence interval of the minimum error point
#if you have 5 highly correlated variables it will keep the best 1
#all the above is the lasso. Lasso is good for variable selection
coefpath(value3) ##doesn't work. Should show live animation of above

coefplot(value3, sort='magnitude', lambda='lambda.1se') #doesn't work with 3 level outcome
##################################################################################################
##################################################################################################
#CROSS VALIDATION Eslastic Net (Ridge and Lasso)
value4 <- cv.glmnet(x = valueX, y=valueY,#calling it value 4 
                             family='multinomial',
                             alpha=0.6,#by default alpha is set to 1 which means lasso. setting to zero it becomes ridge. by setting to .6 is does 60% lasso, 40% ridge AKA THE ELASTIC NET
                             nfolds=5)
plot(value4)
coefpath(value4) #this shows shrinkage and variable selection








#################################################################################################
#OPTIMIZE LAMBDA MIN
# Fit the final model using the optimal lambda
optimal_lambda <- value3$lambda.min  # or use lambda.1se for a more regularized model
final_model <- glmnet(x = valueX, y = valueY, family = 'multinomial', lambda = optimal_lambda)
#################################################################################################
#################################################################################################
#OPTIMIZE LAMBDA 1SE
# Fit the final model using a more regularized model of lambda
optimal_lambda_se <- value3$lambda.1se  # or use lambda.1se for a more regularized model
final_model_se <- glmnet(x = valueX, y = valueY, family = 'multinomial', lambda = optimal_lambda_se)
#################################################################################################
#################################################################################################
#OPTIMIZE LAMBDA MIN
# Fit the final model using the optimal lambda
optimal_lambda4 <- value4$lambda.min  # or use lambda.1se for a more regularized model
final_model4 <- glmnet(x = valueX, y = valueY, family = 'multinomial', lambda = optimal_lambda4)
#################################################################################################
#################################################################################################
#OPTIMIZE LAMBDA 1SE
# Fit the final model using a more regularized model of lambda
optimal_lambda_se4 <- value4$lambda.1se  # or use lambda.1se for a more regularized model
final_model_se4 <- glmnet(x = valueX, y = valueY, family = 'multinomial', lambda = optimal_lambda_se4)








#################################################################################################
#get Ready to test glmnet multinomial algorithm
testvalueX <- model.matrix(formula_train, data = testData)

testvalueY <- model.response(model.frame(formula_train, data = testData))
#################################################################################################










##################################################################################################
# Predict probabilities for each class LASSO min lambda
predicted_probabilities <- predict(final_model, newx = testvalueX, type = "response")

# If you also want class predictions
predicted_classes <- predict(final_model, newx = testvalueX, type = "class")
##################################################################################################
##################################################################################################
# Predict probabilities for each class LASSO 1se lambda
predicted_probabilities_se <- predict(final_model_se, newx = testvalueX, type = "response")

# If you also want class predictions
predicted_classes_se <- predict(final_model_se, newx = testvalueX, type = "class")
##################################################################################################
##################################################################################################
# Predict probabilities for each class (LASSO & RIDGE Min Error)
predicted_probabilities4 <- predict(final_model4, newx = testvalueX, type = "response")

# If you also want class predictions
predicted_classes4 <- predict(final_model4, newx = testvalueX, type = "class")
##################################################################################################
##################################################################################################
# Predict probabilities for each class (LASSO & RIDGE 1SE)
predicted_probabilities_se4 <- predict(final_model_se4, newx = testvalueX, type = "response")

# If you also want class predictions
predicted_classes_se4 <- predict(final_model_se4, newx = testvalueX, type = "class")
##################################################################################################









##################################################################################################
#LAMBDA MIN
# Convert predicted probabilities to factor if necessary
predicted_classes <- as.factor(predicted_classes)

# Assuming your test outcomes are in a variable called `valueY_test`
confusionMatrix(data = predicted_classes, reference = testvalueY)


#Class: Dropout Class: Enrolled Class: Graduate
#Sensitivity                   0.824          0.2658           0.927
#Specificity                   0.878          0.9421           0.812
#Pos Pred Value                0.762          0.5000           0.831
#Neg Pred Value                0.913          0.8548           0.918
#Prevalence                    0.322          0.1789           0.499
#Detection Rate                0.265          0.0476           0.463
#Detection Prevalence          0.348          0.0951           0.557
#Balanced Accuracy             0.851          0.6039           0.870









#####VISUALIZE###########
# Generate the confusion matrix as a table
conf_matrix <- confusionMatrix(data = predicted_classes, reference = testvalueY)

# Convert confusion matrix to a data frame for plotting
conf_matrix_df <- as.data.frame(conf_matrix$table)

# Create a heatmap plot
ggplot(conf_matrix_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()



####
# Select only numeric columns for PCA
numeric_columns <- testData %>%
  select_if(is.numeric)

# Run PCA on the numeric columns
pca <- prcomp(numeric_columns, scale. = TRUE)

# Create a data frame of the first two principal components
pca_data <- as.data.frame(pca$x[, 1:2])
colnames(pca_data) <- c("PC1", "PC2")

# Add actual, predicted, and misclassified information back to the PCA data frame
pca_data <- pca_data %>%
  mutate(actual = testData$actual,
         predicted = testData$predicted)

# Plot PCA results and highlight misclassified students
ggplot(pca_data, aes(x = PC1, y = PC2, color = actual)) +
  geom_point() +
  # Highlight the misclassified students in red with a different shape
  geom_point(data = pca_data %>% filter(predicted == "Graduate" & actual == "Dropout"), 
             aes(x = PC1, y = PC2), color = "red", shape = 4, size = 3) +
  labs(title = "PCA of Test Data with Misclassified Students Highlighted",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal() +
  theme(legend.position = "bottom")




 ##################################################################################################
##################################################################################################
#LAMBDA.1SE
# Convert predicted probabilities to factor if necessary
predicted_classes_se <- as.factor(predicted_classes_se)

# Assuming your test outcomes are in a variable called `valueY_test`
confusionMatrix(data = predicted_classes_se, reference = testvalueY)


#                     Class: Dropout Class: Enrolled Class: Graduate
#Sensitivity                   0.831          0.2468           0.939
#Specificity                   0.876          0.9628           0.790
#Pos Pred Value                0.761          0.5909           0.817
#Neg Pred Value                0.916          0.8543           0.928
#Prevalence                    0.322          0.1789           0.499
#Detection Rate                0.267          0.0442           0.469
#Detection Prevalence          0.351          0.0747           0.574
#Balanced Accuracy             0.854          0.6048           0.864
#####################################################################################









#####################################################################################
##################################################################################################
#LAMBDA MIN (LASSO & RDIGE)
# Convert predicted probabilities to factor if necessary
predicted_classes4 <- as.factor(predicted_classes4)

# Assuming your test outcomes are in a variable called `valueY_test`
confusionMatrix(data = predicted_classes4, reference = testvalueY)

#Class:                   Dropout Class: Enrolled Class:      Graduate
#Sensitivity                   0.831          0.2658           0.927
#Specificity                   0.878          0.9448           0.812
#Pos Pred Value                0.764          0.5122           0.831
#Neg Pred Value                0.916          0.8552           0.918
#Prevalence                    0.322          0.1789           0.499
#Detection Rate                0.267          0.0476           0.463
#Detection Prevalence          0.350          0.0929           0.557
#Balanced Accuracy             0.855          0.6053           0.870

#####################################################################################
##################################################################################################
#LAMBDA.1SE (LASSO & RIDGE)
# Convert predicted probabilities to factor if necessary
predicted_classes_se4 <- as.factor(predicted_classes_se4)

# Assuming your test outcomes are in a variable called `valueY_test`
confusionMatrix(data = predicted_classes_se4, reference = testvalueY)


#Class:                     Dropout Class: Enrolled Class:    Graduate
#Sensitivity                   0.831          0.1962           0.948
#Specificity                   0.878          0.9655           0.774
#Pos Pred Value                0.764          0.5536           0.807
#Neg Pred Value                0.916          0.8464           0.937
#Prevalence                    0.322          0.1789           0.499
#Detection Rate                0.267          0.0351           0.473
#Detection Prevalence          0.350          0.0634           0.587
#Balanced Accuracy             0.855          0.5809           0.861

#####################################################################################









#####################################################################################
#DO I NEED TO DO BOOTSTRAPPING?
#NEED TO REFRESH ON BOOTSTRAPPING AND HOW WE CAN REPORT FINDINGS ABOVE AND USE THEM TO MAKE DECISIONS
#access variability in classification via bootstrapping (doesn't work)
library(boot)
boot_model <- function(data, indices) {
  d <- data[indices,]  # allows bootstrapping from the dataset
  fit <- glmnet(x = d$x, y = d$y, family = 'multinomial', lambda = optimal_lambda)
  return(coef(fit))  # return coefficients
}

# Set up your data for bootstrapping
boot_data <- list(x = valueX, y = valueY)
boot_results <- boot(data = boot_data, statistic = boot_model, R = 1000)  # R is the number of bootstrap replicates

# Analyze results
boot_results
#####################################################################################





#####################################################################################
#trying again:(doesn't work)
# Define the bootstrapping function
boot_model <- function(data, indices) {
  # Extract subsets based on bootstrap indices
  x_sub <- data$x[indices, , drop = FALSE]  # Ensure matrix format is maintained
  y_sub <- data$y[indices]                 # Extract corresponding response variable
  
  # Fit the model using subset data
  fit <- glmnet(x = x_sub, y = y_sub, family = 'multinomial', lambda = optimal_lambda)
  
  # Return coefficients (or another statistic of interest)
  return(coef(fit))  # Coefficients for each bootstrap replicate
}

# Prepare data for bootstrapping
boot_data <- list(x = valueX, y = valueY)  # Ensure this matches the expected input for your model
boot_results <- boot(data = boot_data, statistic = boot_model, R = 1000)  # R is the number of bootstrap replicates

# Analyze results
boot_results

###################################################################################
#trying stratified bootstrap: (works!)

# Modified bootstrapping function with stratified sampling
boot_model_strat <- function(data, indices) {
  # Stratify sampling to ensure each class is represented
  strat_samples <- lapply(unique(data$y), function(cls) {
    idx_cls <- which(data$y == cls)
    sample(idx_cls, size = max(2, length(idx_cls)), replace = TRUE)  # Ensuring at least two samples per class if possible
  })
  strat_indices <- unlist(strat_samples)
  
  # Subsetting the data
  x_sub <- data$x[strat_indices, , drop = FALSE]
  y_sub <- data$y[strat_indices]
  
  # Fitting the model
  fit <- glmnet(x = x_sub, y = y_sub, family = 'multinomial', lambda = optimal_lambda)
  return(coef(fit))  # return coefficients
}

# Prepare data for bootstrapping
boot_data <- list(x = valueX, y = valueY)

# Perform the bootstrap with stratified sampling
boot_results <- boot(data = boot_data, statistic = boot_model_strat, R = 1000)
####################################################################################################









####################################################################################################
boot_coefs <- boot_results$t
coef_summary <- apply(boot_coefs, 2, function(x) {
  c(Mean = mean(x), SD = sd(x), LowerCI = quantile(x, 0.025), UpperCI = quantile(x, 0.975))
})
print(coef_summary)
###################################################################################################









##################################################################################################
# Assuming `final_model` and `valueX_test` are available
predicted_classes <- predict(final_model, newx = valueX_test, s = "lambda.min", type = "class")
confusionMatrix(predicted_classes, valueY_test)  # valueY_test needs to be defined

###################################################################################################














##################################################################################################
#visualize
# Example of plotting the bootstrap distribution of a specific coefficient
ggplot(data = data.frame(Coef = boot_coefs[,1]), aes(x = Coef)) +
  geom_histogram(bins = 30) +
  labs(title = "Distribution of Bootstrap Coefficients for Predictor 1")




#Reference for Dataset:
  #link: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
#Citation
  #Realinho,Valentim, Vieira Martins,Mónica, Machado,Jorge, and Baptista,Luís. (2021). Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.
 