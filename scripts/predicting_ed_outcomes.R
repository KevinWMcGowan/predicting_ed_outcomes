# Predicting College Graduation Analysis
# Introduction
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

## The data set
  # describe dataset, it's origins, and size
#"different undergraduate degrees, such as agronomy, design, education, nursing, journalism, management, social service, and technologies. The dataset includes information known at the time of student enrollment (academic path, demographics, and social-economic factors) and the students' academic performance at the end of the first and second semesters. The data is used to build classification models to predict students' dropout and academic sucess. The problem is formulated as a three category classification task, in which there is a strong imbalance towards one of the classes." - plagerism rn... cite
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

head(data)
head(variable_table)

################################################################################
# Clean the datasets
################################################################################

# As seen below, the column names include spaces and parenthesis. Both of which
# will cause issues with analysis later on.
cat("Column names in `data` before cleaning:\n")
colnames(data)

cat("\nVariable names in `variable_table` before cleaning:\n")
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

# Apply cleaning function to `variable_table` columns
colnames(variable_table) <- clean_column_names(colnames(variable_table))

# Apply cleaning function to the "Variable Name" column in `variable_table`
variable_table <- variable_table %>%
  mutate(Variable_Name = clean_column_names(Variable_Name))

# Print cleaned column names
cat("\nColumn names in `data` after cleaning:\n")
colnames(data)

cat("\nVariable names in `variable_table` after cleaning:\n")
print(variable_table$Variable_Name)

# Now that the problematic characters have been removed, for simplicity and future referencing,
# ensure all variable names are the same:
matching_variables <- intersect(colnames(data), variable_table$Variable_Name)
non_matching_in_data <- setdiff(colnames(data), variable_table$Variable_Name)
non_matching_in_variable_table <- setdiff(variable_table$Variable_Name, colnames(data))

# Print results
cat("\nMatching Variables:\n")
print(matching_variables)

cat("\nVariables in `data` but not in `variable_table`:\n")
print(non_matching_in_data)

cat("\nVariables in `variable_table` but not in `data`:\n")
print(non_matching_in_variable_table)

# Clean any remaining differences in names using an additional cleaning function
clean_names <- function(names) {
  names %>%
    tolower() %>%
    gsub("['/\\-\\.]", "_", .) %>%  # Replace apostrophes, slashes, and dots with underscores
    gsub(" +", "_", .) %>%          # Replace spaces with underscores
    trimws()                        # Remove leading/trailing whitespace
}

# Apply the `clean_names` function to `data`
colnames(data) <- clean_names(colnames(data))

# Apply the `clean_names` function to the `Variable Name` column in `variable_table`
variable_table <- variable_table %>%
  mutate(Variable_Name = clean_names(Variable_Name))

# Recheck matching variables
matching_variables <- intersect(colnames(data), variable_table$Variable_Name)
non_matching_in_data <- setdiff(colnames(data), variable_table$Variable_Name)
non_matching_in_variable_table <- setdiff(variable_table$Variable_Name, colnames(data))

# Print results after final cleaning
cat("\nMatching Variables after final cleaning:\n")
print(matching_variables)

cat("\nVariables in `data` but not in `variable_table` after final cleaning:\n")
print(non_matching_in_data)

cat("\nVariables in `variable_table` but not in `data` after final cleaning:\n")
print(non_matching_in_variable_table)

#The above shows that all variable names are now matching.


################################################################################
## Set Variable Types

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
  select(Variable_Name = `Variable_Name`, Expected_Type = Type)

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
# Exploratory Data Analysis (ENSURE HENCE FORTH NO DATA IS USED!!!!! ONLY TRAIN DATA)
################################################################################
# Now that the data has been cleaned, some exploration can be done to better understand
# the students in the dataset by the variables that describe them. Any findings can contribute
# to the modeling done later.

# see number of students in dataset
nrow(traindata)
#3540 students
#see the number of variables in the dataset
ncol(traindata)
# 36 variables to help predict target (37th)
# See the unique values of the target variable.
unique(traindata$target)
table(traindata$target)
# 3 levels of outcomes to predict, 1137 have dropped out, 636 are currently enrolled, and 1768 have graduated.
# the relativly small number of currently enrolled students might lead to difficulty. 
# Look at the top 6 values of each column
head(traindata)
#all values are coded with numbers. In order to have a functional and targeted exploration, the following section 
# performs a linear regression to get an idea of what variables are predictive of dropout, enrolled, and graduated. Afterwards, these predictive variables
# will be explored more thoroughly.


################################################################################
# Perform Linear regression


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
# Investigate predictive variables

# Group variables for analysis
binary_vars <- c("tuition_fees_up_to_date", "gender", "scholarship_holder", 
                 "debtor", "international", "educational_special_needs")
categorical_vars <- c("marital_status", "application_mode", "daytime_evening_attendance", "nacionality")
continuous_vars <- c("admission_grade", "curricular_units_1st_sem_grade", "gdp")


######################################
# Binary variables
view(variable_table)
# View descriptions for binary variables

# Select specific rows and print only the Variable_Name and Description columns
variable_table %>%
  slice(c(14:19, 21)) %>%  # Select rows 14 to 19 and 21
  select(Variable_Name, Description) %>%  # Select specific columns
  print()
# 1 = yes & 0 = no for binary values


# Prepare the data for plotting
binary_data <- traindata %>%
  select(all_of(binary_vars)) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  count(Variable, Value)

# Map human-readable names for binary variable values
value_labels <- c("0" = "No/Not Applicable/Male", "1" = "Yes/Applicable/Female")

# Create the bar chart
ggplot(binary_data, aes(x = Variable, y = n, fill = as.factor(Value))) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  scale_fill_manual(
    values = c("0" = "steelblue", "1" = "lightcoral"),
    labels = value_labels
  ) +
  geom_text(
    aes(label = n), 
    position = position_dodge(width = 0.9), 
    vjust = -0.5, 
    size = 3
  ) +
  labs(
    title = "Distribution of Binary Variables",
    x = "Variables",
    y = "Count",
    fill = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

# The chart above shows that the vast majority of students are not debtor (don't owe money to school), nor special needs, most are male, aren't international, or scholarship holders
# This finding warrent more digging, since the regression above found all but gender to be predictive of our target variable.
# Below, compares the same variables against the 33 levels of target ( 0 = dropout, 1 = enrolled, 2 = graduated)
# Prepare the data for plotting
binary_target_data <- traindata %>%
  select(all_of(binary_vars), target) %>%
  pivot_longer(cols = -target, names_to = "Variable", values_to = "Value") %>%
  count(Variable, Value, target)

# Map human-readable names for binary variable values
value_labels <- c("0" = "No/Not Applicable/Male", "1" = "Yes/Applicable/Female")

# Create the grouped bar chart
ggplot(binary_target_data, aes(x = Variable, y = n, fill = as.factor(target))) +
  geom_bar(stat = "identity", position = position_dodge(), color = "black") +
  facet_wrap(~Value, labeller = labeller(Value = value_labels)) +
  labs(
    title = "Distribution of Binary Variables Across Target Levels",
    x = "Variables",
    y = "Count",
    fill = "Target (1 = Dropout, 2 = Enrolled, 3 = Graduated)"
  ) +
  scale_fill_manual(
    values = c("1" = "steelblue", "2" = "gold", "3" = "darkgreen"),
    labels = c("1" = "Dropout", "2" = "Enrolled", "3" = "Graduated")
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5),
    strip.text = element_text(size = 10)
  )
# summary table of above: 
# Table of binary variables against target
binary_target_table <- traindata %>%
  select(all_of(binary_vars), target) %>%
  pivot_longer(cols = -target, names_to = "Variable", values_to = "Value") %>%
  count(Variable, Value, target) %>%
  pivot_wider(
    names_from = target,
    values_from = n,
    values_fill = 0, # Fill missing combinations with 0
    names_prefix = "Target_"
  )
print(binary_target_table)
# The chart above reflects the regression above that these variables do have a high number of 
# of graduated students, which is typical with the National Center for Education Statistics reporting average 6year gradaution of college
# at 64% in 2020.

# A few powerful takeaways include: 
#- A significant number of students who dropped out (312) were debtors compared to those who graduated (101) or are still enrolled (90).
#- A similar number of students without scholarships dropped out (1287) as those who graduated (1374). However, The rate of scholarship holders dropping out (12%)
  # is much loter tahn that of nonscholarship holdrs 38%. Suggesting financial support is an incentive to not dropout
scholarship_holder_rate <- binary_target_table %>%
  filter(Variable == "scholarship_holder", Value == 1) %>%
  summarize(
    total_scholarship_dropouts = sum(Target_1),
    total_scholarship_students = sum(Target_1 + Target_2 + Target_3),
    scholarship_dropout_rate = total_scholarship_dropouts / total_scholarship_students
  )
scholarship_holder_rate
# same a above for non_scholarship holders
non_scholarship_holder_rate <- binary_target_table %>%
  filter(Variable == "scholarship_holder", Value == 0) %>%
  summarize(
    total_non_scholarship_dropouts = sum(Target_1),
    total_non_scholarship_students = sum(Target_1 + Target_2 + Target_3),
    non_scholarship_dropout_rate = total_non_scholarship_dropouts / total_non_scholarship_students
  )
non_scholarship_holder_rate
#- Although there are significantly more men (2278) than women (1262), in the dataset, they have roughly the same number of drop outs (men = 556 & women = 569).
  # As a result, the rate of female_drop out is very high 45% for women vs 24% for men.
  # This could be sampling error and possibly unique to the dataset which is taken from the following diverse degree programs:
  #"agronomy, design, education, nursing, journalism, management, social service, and technologies" (cite).
gender_male_rate <- binary_target_table %>%
  filter(Variable == "gender", Value == 0) %>%
  summarize(
    total_male_dropouts = sum(Target_1),
    total_male_students = sum(Target_1 + Target_2 + Target_3),
    male_dropout_rate = total_male_dropouts / total_male_students
  )
gender_male_rate

gender_female_rate <- binary_target_table %>%
  filter(Variable == "gender", Value == 1) %>%
  summarize(
    total_female_dropouts = sum(Target_1),
    total_female_students = sum(Target_1 + Target_2 + Target_3),
    female_dropout_rate = total_female_dropouts / total_female_students
  )
gender_female_rate

# Almost all of students who owe tuition fees (87%) dropout. Shedding light again on the importance of financial support and reflecting 
  # the finding in the regression chart as the most predictive with a correlation coefficient of .5
owe_fees_drop_rate <- binary_target_table %>%
  filter(Variable == "tuition_fees_up_to_date", Value == 0) %>%
  summarize(
    total_owe_fee_dropouts = sum(Target_1),
    total_owe_fee_students = sum(Target_1 + Target_2 + Target_3),
    owe_fee_dropout_rate = total_owe_fee_dropouts / total_owe_fee_students
  )
owe_fees_drop_rate

# International students have a drop out rate of 30%, which is similar to the total dropout rate of teh data set ~32%.
# this is interesting when internation status was the second most predictive variable after tuition and fees up to date.
international_rate <- binary_target_table %>%
  filter(Variable == "international", Value == 1) %>%
  summarize(
    total_international_dropouts = sum(Target_1),
    total_international_students = sum(Target_1 + Target_2 + Target_3),
    international_dropout_rate = total_international_dropouts / total_international_students
  )
international_rate
# Calculate overall dropout rate
overall_dropout_rate <- traindata %>%
  summarize(
    total_dropouts = sum(target == 1),
    total_students = n(),
    dropout_rate = total_dropouts / total_students
  )
overall_dropout_rate

# Althought there were not many students with special needs in the dataset, the 
# the perseverence of these students, and possibly their support is clear when the dropout rate of 35% is not much
# high than that of the dataset as a whole (~32%)
educational_special_needs_rate <- binary_target_table %>%
  filter(Variable == "educational_special_needs", Value == 1) %>%
  summarize(
    total_special_needs_dropouts = sum(Target_1),
    total_special_needs_students = sum(Target_1 + Target_2 + Target_3),
    special_needs_dropout_rate = total_special_needs_dropouts / total_special_needs_students
  )
educational_special_needs_rate


###############################################################################
#Continuous Variables, specifically, the cirriculular unit variables

variable_table %>%
  slice(22:33) %>%  # Select rows 22 to 33
  select(Variable_Name, Description) %>%  # Select specific columns
  print()
# As seen above the curricular unit variables give us 6 looks at how the students stand each semester.
# - The number of credits earned
# - The number of units enrolled in
# - The number of evaluations/ tests taken
# - The number of units they were approved to take
# - Their Grade avg 
# - The number of units that didn't have tests/evaluations (conceivably easier courses.)




#LOOK INTO NATIONALITY OF STUDENTS WITH LOOKUP TABLE FOR BIAS READING IN EXPORITORY ANALYSIS







#featues:
#- create a owe money feature of students who both are debtor = 1 and tuition feeds up to date = 0 



































# Limitations
 # This data doesnt include year over year data resulting in limited conclusion ability based
  # on annual distribution of enrolled, graduate, and drop out.











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

# Citations
SOURCE: U.S. Department of Education, National Center for Education Statistics, Integrated Postsecondary Education Data System (IPEDS), Winter 2020–21, Graduation Rates component. See Digest of Education Statistics 2021, table 326.20. Retreieved on 11/25/24 from https://nces.ed.gov/fastfacts/display.asp?id=40