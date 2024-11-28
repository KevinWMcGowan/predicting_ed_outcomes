# Predicting College Graduation Analysis
# This script processes the “Predict Students’ Dropout and Academic Success” dataset, 
# developed by Realinho and colleagues (2021). It includes feature engineering and 
# utilizes ________ models to predict student outcomes: college graduation, dropout, or enrollment.
# College is not only an expensive edevour, it's also predicting of long-term helth outcomes and income. As such
# being able to predict dropout before it happens could allow for early intervention programs
# to support students towards success. 
# Knowing universities have thousands of students, and only a few counselors in some cases, a predictive model
# like the one develoed below, would be immensly helpful in giving counselors a place to start.
# With this desired end in mind, the following reports aims to support
# counselors, administrators, and teachers and ultimately students by informing who might needs the most help.

# Requirements:
# - Ensure necessary libraries are installed (see library loading section)(should be automated).
# - If you've downloaded the entire repository then the code should run fine. 


################################################################################
# Load Libraries
################################################################################
# List of required packages
packages <- c("readr", "tinytex", "dplyr", "caret", "stringr", "tidyverse", "tidyr", "broom", "glmnet", "Matrix","coefplot", "here")

# Check and install missing packages
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Load the libraries
lapply(packages, library, character.only = TRUE)

################################################################################
# Download data & variable table
################################################################################
# Set up the file paths for the data files
data_path <- here("data", "data.csv")
variable_table_path <- here("data", "variable_table.csv")

# Check if the file exists in the expected location
if (!file.exists(data_path)) {
  stop("The file 'data.csv' is not found in the 'data' folder. 
       Please ensure the file is placed in the 'data' folder inside the 'predicting_ed_outcomes' repository.")
}

# Load the datasets
data <- read.csv(data_path, header = TRUE, sep = ";")
variable_table <- read_csv(variable_table_path)
#view(variable_table)

################################################################################
# As seen below, the column names include spaces and parenthesis. Both of which
# will cause issues with analysis later on.
colnames(data)
print(variable_table$`Variable Name`)

# In order to remove these problematic characters, the following regex code removes them.
clean_column_names <- function(colnames) {
  colnames %>%
    # Replace spaces with underscores
    gsub(" ", "_", .) %>%
    # Remove parentheses
    gsub("\\(", "", .) %>%
    gsub("\\)", "", .) %>%
    # Replace periods with underscores
    gsub("\\.", "_", .) %>%
    # Replace multiple underscores with a single one
    gsub("_+", "_", .) %>%
    # Trim trailing underscores
    gsub("_$", "", .)
}

# Apply cleaning function to `data` columns
colnames(data) <- clean_column_names(colnames(data))

# Apply cleaning function to `variable_table`s "Variable Name" column
variable_table <- variable_table %>%
  mutate(`Variable Name` = clean_column_names(`Variable Name`))

# Print cleaned column names
print(colnames(data))
print(variable_table$`Variable Name`)

# Now that the problematic characters have been removed, for simplicity and future referencing,
# ensure all variable names are the same:
matching_variables <- intersect(colnames(data), variable_table$`Variable Name`)
non_matching_in_data <- setdiff(colnames(data), variable_table$`Variable Name`)
non_matching_in_variable_table <- setdiff(variable_table$`Variable Name`, colnames(data))

# Print results
print(matching_variables)
print(non_matching_in_data)
print(non_matching_in_variable_table)
# Print out above shows 6 variables have different names. 
# The read out shows these differences include 
# Regex code will help fix these difference include apostrophes, slashes, and dots with underscores.

clean_names <- function(names) {
  names %>%
    tolower() %>%
    gsub("['/\\-\\.]", "_", .) %>%  # Replace apostrophes, slashes, and dots with underscores
    gsub(" +", "_", .) %>%          # Replace spaces with underscores
    trimws()                        # Remove leading/trailing whitespace
}

# apply clean_names function to `data`
colnames(data) <- clean_names(colnames(data))

# apply clean_names function to `Variable Name` column in `variable_table`
variable_table <- variable_table %>%
  mutate(`Variable Name` = clean_names(`Variable Name`))

# Recheck matching variables
matching_variables <- intersect(colnames(data), variable_table$`Variable Name`)
non_matching_in_data <- setdiff(colnames(data), variable_table$`Variable Name`)
non_matching_in_variable_table <- setdiff(variable_table$`Variable Name`, colnames(data))

# Print results
cat("Matching Variables:\n")
print(matching_variables)
cat("\nVariables in `data` but not in `variable_table`:\n")
print(non_matching_in_data)
cat("\nVariables in `variable_table` but not in `data`:\n")
print(non_matching_in_variable_table)

#The above shows that all variable names are now matching.

# The next cleaning step is to set variable types. Luckily, the variable_table tells us
# how to code each variable type for analysis. The following code shows expected type (variable table) vs
# the type currently in the data.

# Extract the actual types from the data
data_types <- data.frame(
  Variable_Name = colnames(data),
  Data_Type = sapply(data, class),
  stringsAsFactors = FALSE
)

# Extract expected types from the variable_table
# Adjust column names as per the actual structure of `variable_table`
lookup_types <- variable_table %>%
  select(Variable_Name = `Variable Name`, Expected_Type = Type)

# Join the actual types with the expected types
comparison <- data_types %>%
  left_join(lookup_types, by = "Variable_Name")

# View the comparison
print(comparison)

# Since R's equivalent of a "continuous" variable type is "numeric", the following changes are made:
# - `curricular_units_1st_sem_grade` is changed from numeric to integer.
# - The `target` variable is replaced with numeric values for regression analysis.

# Change `curricular_units_1st_sem_grade` from numeric to integer
data$curricular_units_1st_sem_grade <- as.integer(data$curricular_units_1st_sem_grade)

# Verify the change for `curricular_units_1st_sem_grade`
str(data[c("curricular_units_1st_sem_grade")])
# This ensures that the column is now stored as integers, aligning with the expected variable type in the lookup table.

# Encode `target` as numeric
# Convert `target` into a factor if it isn't already
data$target <- as.factor(data$target)

# Replace the `target` variable with its numeric representation
data$target <- as.numeric(data$target)

# Verify the change for `target`
unique(data$target)
# The `target` variable is now numeric:
# For example:
# "Dropout" -> 1
# "Enrolled" -> 2
# "Graduate" -> 3
# Count the occurrences of each value in the target column
target_counts <- table(data$target)

# Print the counts
print(target_counts)

# If you want a more descriptive output
cat("Counts for each target value:\n")
for (value in names(target_counts)) {
  cat(value, ":", target_counts[value], "\n")
}
################################################################################
## Split the dataset
# Now before any analysis or exploraiton is done, the dataset must be split in order
# to avoid overfitting
# PARTITION DATA
#set up test and train sets
# for reproducibility
set.seed(123)  
trainindex <- createDataPartition(data$target, p = .8, 
                                  list = FALSE, 
                                  times = 1)
traindata <- data[trainindex,]
testdata  <- data[-trainindex,]


################################################################################
# Exploratory Data Analysis
################################################################################
# Now that the data has been cleaned, some exploration can be done to better understand
# the students in the dataset by the variables that describe them. Any findings can contribute
# to the modeling done later.

# see number of students in dataset
nrow(traindata)
#4424 students
#see the number of variables in the dataset
ncol(traindata)
# 36 variables to help predict target
# See the unique values of the target variable.
unique(traindata$target)
table(traindata$target)
# 3 levels of outcomes to predict, 1137 have dropped out, 636 are currently enrolled, and 1768 have graduated.
# the relativly small number of currently enrolled students might lead to difficulty. 
# Look at the top 6 values of each column
head(traindata)
#all values are coded with numbers, except the target variable.

################################################################################
# Create lookup Table

# As was seen in the top 6 rows of each column, all values are coded with numbers, except the target variable.
# As a result, a lookup table from the variable_table is needed to make sense of these values.

# Create a lookup table with variable names and descriptions
lookup_table <- variable_table %>%
  select(`Variable Name`, Description) %>%
  rename(Column = `Variable Name`)

# Display the lookup table
lookup_table

################################################################################
# Perform Linear regression

# Before diving into deep analysis of all 36 variables, a quick linear regression 
# will help identify the most important predictors of dropout, enrolled, and graduated.

# Extract variable names for the formula
variables_train <- colnames(traindata)

# Remove the 'target' variable from the list of predictors
variables_train <- variables_train[variables_train != "target"]

# Create the formula as a string
formula_string_train <- paste("target ~", paste(variables_train, collapse = " + "), - 1)

# Convert the string to a formula object
formula_train <- as.formula(formula_string_train)
print(formula_train)

# Perform linear regression
value1 <- lm(formula_train, data = traindata)
summary(value1)

# Visualize the coefficients
coefplot(value1, sort = 'magnitude', conf.int = TRUE)

# The coefficient plot above highlights variables that are strongly predictive 
# of the target outcome, as well as those with wide error margins crossing zero, 
# indicating they may have little to no effect.

# The plot suggests focusing on variables such as:
# `tuition_fees_up_to_date`, `international`, 
# `curricular_units_2nd_sem_approved`, `scholarship_holder`, 
# `daytime_evening_attendance`, and `curricular_units_1st_sem_enrolled`, 
# which show positive predictive values ranging from approximately 0.1 to 0.5.
# These variables may contribute positively to predicting student outcomes like 
# graduation or enrollment.

# Conversely, certain factors show negative correlations with the target, 
# as indicated by their negative coefficients. These include:
# `gender`, `curricular_units_1st_sem_credited`, 
# `curricular_units_2nd_sem_credited`, `educational_special_needs`, 
# `debtor`, and `curricular_units_2nd_sem_enrolled`.
# These insights suggest these variables may be associated with less favorable 
# outcomes, such as dropout.

# Together, these findings provide a foundation for deeper exploration into how 
# these variables influence the target outcomes and how they can be utilized 
# effectively in predictive modeling.

################################################################################
## Explore Predictive Variables (all the following needs to update with lookup help)(read pliots fodler for visuals)

# Ensure the 'plots/' directory exists
if (!dir.exists("plots")) {
  dir.create("plots")
}

# Variables of interest
variables_of_interest <- c(
  "tuition_fees_up_to_date", "international", 
  "curricular_units_2nd_sem_approved", "scholarship_holder", 
  "daytime_evening_attendance", "curricular_units_1st_sem_enrolled",
  "gender", "curricular_units_1st_sem_credited", 
  "curricular_units_2nd_sem_credited", "educational_special_needs", 
  "debtor", "curricular_units_2nd_sem_enrolled"
)

# Analyze distributions for binary variables
binary_vars <- c("tuition_fees_up_to_date", "international", "scholarship_holder", 
                 "daytime_evening_attendance", "gender", "educational_special_needs", "debtor")

for (var in binary_vars) {
  cat(paste("\n--- Distribution of", var, "---\n"))
  print(table(data[[var]]))
  
  plot <- ggplot(data, aes_string(x = var)) +
    geom_bar(fill = "steelblue", color = "black") +
    labs(title = paste("Distribution of", var), x = var, y = "Count") +
    theme_minimal()
  
  ggsave(filename = paste0("plots/", var, "_distribution.png"), plot = plot)
}

# Analyze distributions for continuous variables
continuous_vars <- c("curricular_units_2nd_sem_approved", 
                     "curricular_units_1st_sem_enrolled", 
                     "curricular_units_1st_sem_credited", 
                     "curricular_units_2nd_sem_credited", 
                     "curricular_units_2nd_sem_enrolled")

for (var in continuous_vars) {
  cat(paste("\n--- Summary of", var, "---\n"))
  print(summary(data[[var]]))
  
  plot <- ggplot(data, aes_string(x = var)) +
    geom_histogram(binwidth = 1, fill = "steelblue", color = "black", alpha = 0.7) +
    labs(title = paste("Distribution of", var), x = var, y = "Count") +
    theme_minimal()
  
  ggsave(filename = paste0("plots/", var, "_distribution.png"), plot = plot)
}

# Explore relationships between variables and the target
cat("\n--- Exploring relationships with Target ---\n")

# Binary variables vs. target
for (var in binary_vars) {
  cat(paste("\n---", var, "vs. Target ---\n"))
  print(table(data[[var]], data$target))
  
  plot <- ggplot(data, aes_string(x = var, fill = "target")) +
    geom_bar(position = "fill") +
    labs(title = paste(var, "vs. Target"), x = var, y = "Proportion", fill = "Target") +
    theme_minimal()
  
  ggsave(filename = paste0("plots/", var, "_vs_target.png"), plot = plot)
}

# Continuous variables vs. target
for (var in continuous_vars) {
  plot <- ggplot(data, aes_string(x = "target", y = var, fill = "target")) +
    geom_boxplot() +
    labs(title = paste(var, "vs. Target"), x = "Target", y = var) +
    theme_minimal()
  
  ggsave(filename = paste0("plots/", var, "_vs_target.png"), plot = plot)
}

# Summarize findings
cat("\n--- Summary of Exploratory Analysis ---\n")
cat("Bar plots for binary variables and histograms for continuous variables have been saved in the 'plots' folder.\n")
cat("Relationships with the target variable are visualized with bar plots for binary variables and boxplots for continuous variables.\n")


################################################################################
#LOOK INTO NATIONALITY OF STUDENTS WITH LOOKUP TABLE FOR BIAS READING IN EXPORITORY ANALYSIS























































################################################################################
# Optional

# Create a function for getting descriptions
# Function to get description for a column
get_description <- function(column_name, lookup_table) {
  description <- lookup_table %>%
    filter(Column == column_name) %>%
    pull(Description)
  
  if (length(description) == 0) {
    return(paste("No description found for", column_name))
  }
  return(description)
}

# Example usage: Get description for "Marital.status"
get_description("Marital_status", lookup_table)

