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

#Since R's equivelent of a "continuous" variable type is "numeric", there are only 2 changes to make:
#- curricular_units_1st_sem_grade from numeric to integer
#- target from character to factor (the categorical equivalent for ML classification tasks)
# The following code makes these changes:
data$curricular_units_1st_sem_grade <- as.integer(data$curricular_units_1st_sem_grade)

# Change `target` from character to factor
data$target <- as.factor(data$target)
# Verify the changes
str(data[c("curricular_units_1st_sem_grade", "target")])

#Above shows the three levels of outcomes for the target variable as a factor and
# the integer values for curricular units 1st semester.


################################################################################
################################################################################
# Exploratory Data Analysis
################################################################################
# Now that the data has been cleaned, some exploration can be done to better understand
# the students in the dataset by the variables that describe them. Any findings can contribute
# to the modeling done later.

# see number of students in dataset
nrow(data)
#4424 students
ncol(data)

# 36 variables to help predict target
unique(data$target)
# Knowing our task is to predict Dropout, Graduate, Enrolled


# Create a lookup table with variable names and descriptions
lookup_table <- variable_table %>%
  select(`Variable Name`, Description) %>%
  rename(Column = `Variable Name`)

# Display the lookup table
lookup_table

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

