# Predicting College Graduation Analysis

# This script processes the “Predict Students’ Dropout and Academic Success” dataset, 
# developed by Martins and colleagues (2021). It includes feature engineering and 
# utilizes decision tree and gradient boosted trees with cross validation models 
# to predict 3 levels of student outcomes: college graduation, dropout, or enrollment.

# Requirements:
# - Ensure necessary libraries are installed (see library loading section)(should be automated).
# - If you've downloaded the entire repository then the code should run fine. 



################################################################################
# Load Libraries
################################################################################

# List of required packages
packages <- c("readr", "gbm", "tinytex", "rpart.plot", "dplyr", "caret", "rpart", "stringr", "tidyverse", "tidyr", "broom", "glmnet", "Matrix","coefplot", "here")

# Check and install missing packages
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Load the libraries
invisible(lapply(packages, library, character.only = TRUE))

################################################################################
# Download and unzip Repository
################################################################################
##########Test###########

################################################################################
# Download and Unzip Repository
################################################################################

# Define the GitHub repository URL and destination path
repo_url <- "https://github.com/KevinWMcGowan/predicting_ed_outcomes/archive/refs/heads/main.zip"
zip_file <- here("data", "predicting_ed_outcomes-main.zip")
unzip_dir <- here("data", "predicting_ed_outcomes-main")

# Check if the repository is already downloaded
if (!file.exists(zip_file)) {
  cat("Downloading the repository...\n")
  download.file(repo_url, zip_file, mode = "wb")
}

# Check if the repository is already unzipped
if (!dir.exists(unzip_dir)) {
  cat("Unzipping the repository...\n")
  unzip(zip_file, exdir = here("data"))
}

################################################################################
# Load Data and Variable Table
################################################################################

# Define the paths to the data folder within the unzipped repository
data_folder <- file.path(unzip_dir, "data")  # Path to the "data" folder
data_path <- file.path(data_folder, "data.csv")
variable_table_path <- file.path(data_folder, "variable_table.csv")

# Check if the "data" folder exists
if (!dir.exists(data_folder)) {
  stop("The 'data' folder is not found in the unzipped repository. 
       Please check the repository structure.")
}

# Check if the required files exist in the "data" folder
if (!file.exists(data_path)) {
  stop("The file 'data.csv' is not found in the 'data' folder. 
       Please ensure it exists in the repository structure.")
}

if (!file.exists(variable_table_path)) {
  stop("The file 'variable_table.csv' is not found in the 'data' folder. 
       Please ensure it exists in the repository structure.")
}

# Load the datasets
data <- read.csv(data_path, header = TRUE, sep = ";")
variable_table <- read_csv(variable_table_path)

cat("Data and variable table loaded successfully from the 'data' folder.\n")


################################################################################
# Clean the datasets
################################################################################

# As seen below, the column names in data table and the variable and the variable_name column
  #include spaces and parenthesis. All of which will cause issues with analysis later on.
cat("Column names in `data` before cleaning:\n")
colnames(data)
nrow(data)
cat("\nColumn names in `variable_table` before cleaning:\n")
colnames(variable_table)
cat("\nVariable names in `variable_table` before cleaning:\n")
print(variable_table$`Variable Name`)


# In order to remove these problematic characters, the following function and regex code removes them.
clean_names <- function(names) {
  names %>%
    # Convert to lowercase
    tolower() %>%
    # Replace spaces and special characters with underscores
    gsub("[ .\\(\\)/'\\-]", "_", .) %>%
    # Replace multiple underscores with a single one
    gsub("_+", "_", .) %>%
    # Remove leading and trailing underscores
    gsub("^_|_$", "", .) %>%
    # Trim any remaining whitespace
    trimws()
}


# Apply the cleaning function to `data` and `variable_table` names
colnames(data) <- clean_names(colnames(data))
colnames(variable_table) <- clean_names(colnames(variable_table))

# Print cleaned column names
cat("\nColumn names in `data` after cleaning:\n")
print(colnames(data))
cat("\nColumn names in `variable_table` after cleaning:\n")
print(colnames(variable_table))

# Now, clean the entries within the 'variable_name' column in `variable_table`
variable_table$variable_name <- clean_names(variable_table$variable_name)

# Now that the problematic characters have been removed, ensure all variable names are consistent
matching_variables <- intersect(colnames(data), variable_table$variable_name)
non_matching_in_data <- setdiff(colnames(data), variable_table$variable_name)
non_matching_in_variable_table <- setdiff(variable_table$variable_name, colnames(data))

# Display the matching and non-matching variables
cat("\nMatching Variables:\n")
print(matching_variables)

#The above shows that all 37 variable names are now matching.


################################################################################
## Set Variable Types

# The next cleaning step is to set variable types. Luckily, the variable_table tells the author
# how to code each variable type for analysis. The following code shows expected type (variable table) vs
# the type currently in the data.

# Extract the actual types from the data
#data_types <- data.frame(
 # variable_name = colnames(data),   
#  data_type = sapply(data, class), 
#  stringsAsFactors = FALSE
#)

# Extract expected types from the variable_table
#lookup_types <- variable_table %>%
#  select(variable_name, expected_type = type)

# Join the actual types with the expected types
#comparison <- data_types %>%
#  left_join(lookup_types, by = "variable_name") 

# Display the comparison
#cat("\nComparison of Actual and Expected Types:\n")
#print(comparison)

# Since R's equivalent of a "continuous" variable type is "numeric", the following changes are made:(or only 1 change is needed (target as factor)
# - `curricular_units_1st_sem_grade` is changed from numeric to integer.
# - The `target` variable is replaced with numeric values for regression analysis.
#data$curricular_units_1st_sem_grade <- as.integer(data$curricular_units_1st_sem_grade)`
#str(data[c("curricular_units_1st_sem_grade")])
# Above shows column is now stored as integers, aligning with the expected variable type in the lookup table.

# Since target is the only variable not represented with a number value, 
  # the following encodes 'target' variable with a numeric representation
data$target <- ifelse(data$target == "Dropout", 1,
                      ifelse(data$target == "Enrolled", 2,
                             ifelse(data$target == "Graduate", 3, NA)))
sort(unique(data$target))
# The target variable is now numeric:
# "Dropout" -> 1
# "Enrolled" -> 2
# "Graduate" -> 3


################################################################################
## Split The Dataset and Justify The Decision

# Now before any analysis or exploration can be done, as best practice the dataset is split in order
# to avoid over training. 

# A quick inspection of the dataset shows a relatively small sample size for model training, and
# a strong class imbalance:
nrow(data)
table(data$target)

# Below an 80/20 split is used to maintain as much of the dataset for training as possible.
# Once modeling begins, resampling techniques will be needed to address class imbalance.
set.seed(123)  
trainindex <- createDataPartition(data$target, p = .8, 
                                  list = FALSE, 
                                  times = 1)
traindata <- data[trainindex,]
final_holdout_set  <- data[-trainindex,]


# For sanity, below the number of nas in both the train and final_holdout set is counted.
  # This is the only time the final_holdout_set is inspected before testing algorithms.

# Count total NAs in traindata
total_nas_traindata <- sum(is.na(traindata))
cat("Total NAs in traindata:", total_nas_traindata, "\n")

# Count total NAs in testdata
total_nas_final_holdout_set <- sum(is.na(final_holdout_set))
cat("Total NAs in final_holdout_set data:", total_nas_final_holdout_set, "\n")


################################################################################
# Exploratory Data Analysis
################################################################################
# Now that the data has been cleaned, some exploration can be done to better understand
# the students in the traindata by inspecting the variables that describe them. 
# Any findings can contribute to the modeling done later.

# See number of students in dataset
nrow(traindata)
ncol(traindata)
unique(traindata$target)
table(traindata$target)
head(traindata)

# The code above shows the data contains 3540 students, 36 variables to help predict target (37th),
#3 levels of outcomes to predict, 1137 have dropped out, 636 are currently enrolled, and 1768 have graduated.
# The relatively small number of currently enrolled students will likely lead to difficulty in modeling due to class imbalance
# Lastly, all values are coded with numbers. In order to have a functional and targeted exploration, the following section 
# performs a linear regression to get an idea of what variables are predictive of dropout, enrolled, and graduated. Afterwards, these predictive variables
# will be explored more thoroughly.


################################################################################
## Linear Regression 
## To better understand the predictive variables in the dataset, this regression will visualize
  # how certain variables might be predictive of target as a whole.

# Extract variable names for the formula
variables_train <- colnames(traindata)

# Remove the 'target' variable from the list of predictors
variables_train <- variables_train[variables_train != "target"]

# Create the formula as a string
formula_string_train <- paste("target ~", paste(variables_train, collapse = " + "), - 1)

# Convert the string to a formula object
formula_train <- as.formula(formula_string_train)

# Perform linear regression
value1 <- lm(formula_train, data = traindata)

# Visualize the coefficients
regression_plot<-coefplot(value1, sort = 'magnitude', conf.int = TRUE)
regression_plot
# The coefficient plot above provides insights into the variables' relationships 
# with the target outcome and their relative strengths. It highlights variables 
# with significant predictive power, as well as those with wide error margins 
# crossing zero, suggesting limited or no effect.

# Key Findings:
# Due to the positive coefficients, the plot suggests focusing on variables such as:
# `tuition_fees_up_to_date`, `international`, 
# `curricular_units_2nd_sem_approved`, `scholarship_holder`, 
# `daytime_evening_attendance`, and `curricular_units_1st_sem_approved` 
# These range approximately from 0.1 to 0.5, indicating a positive contribution 
# toward favorable outcomes like graduation or enrollment.

# Conversely, certain factors show negative correlations with the target, 
# as indicated by their negative coefficients. These include:
# `gender`, `curricular_units_1st_sem_credited`, 
# `curricular_units_2nd_sem_credited`, `educational_special_needs`, 
# `debtor`, and `curricular_units_2nd_sem_enrolled`.
# These insights suggest these variables may be associated with lower values of target, 
# such as dropout and enrolled.

# Together, these findings provide a foundation for deeper exploration into how 
# these variables influence the target outcomes and how they can be utilized 
# effectively in predictive modeling.


################################################################################
# Investigate predictive variables
################################################################################

# Group variables for analysis
binary_vars <- c("tuition_fees_up_to_date", "gender", "scholarship_holder", 
                 "debtor", "international", "educational_special_needs")
categorical_vars <- c("marital_status", "application_mode", "daytime_evening_attendance", "nacionality")
continuous_vars <- c("admission_grade", "curricular_units_1st_sem_grade", "gdp")


######################################
# Binary variables

# Since all variables are encoded numerically, the following code will tell us what the binary values mean.

# Print Binary variable definitions
variable_table %>%
  slice(c(14:19, 21)) %>%  # Select rows 14 to 19 and 21
  select(variable_name, description) %>%  # Select specific columns
  print()
# 1 = yes & 0 = no for binary values

# Plot distribution of binary data
binary_data <- traindata %>%
  select(all_of(binary_vars)) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  count(Variable, Value)
# Map human-readable names for binary variable values
value_labels <- c("0" = "No/Not Applicable/Male", "1" = "Yes/Applicable/Female")
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
# The chart above shows that the vast majority of students are not debtor (don't owe money to school), 
# nor special needs, most are male, aren't international, or scholarship holders.
# This finding warrants more digging, since the regression above found all but gender to be predictive of our target variable.
# Below, the chart compares the same variables against the 3 levels of target ( 0 = dropout, 1 = enrolled, 2 = graduated)
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
#- A similar number of students without scholarships dropped out (1287) as those who graduated (1374). 

#Below, one can see the rate of scholarship holders dropping out (12%)
  # is much lower than that of non_scholarship holders (38%). 


### Scholarship Comparison to Dropout
# Compare scholarship holder drop out vs non_scholarship holder dropout
scholarship_holder_rate <- binary_target_table %>%
  filter(Variable == "scholarship_holder", Value == 1) %>%
  summarize(
    total_scholarship_dropouts = sum(Target_1),
    total_scholarship_students = sum(Target_1 + Target_2 + Target_3),
    scholarship_dropout_rate = total_scholarship_dropouts / total_scholarship_students
  )
scholarship_holder_rate
# same as above for non_scholarship holders
non_scholarship_holder_rate <- binary_target_table %>%
  filter(Variable == "scholarship_holder", Value == 0) %>%
  summarize(
    total_non_scholarship_dropouts = sum(Target_1),
    total_non_scholarship_students = sum(Target_1 + Target_2 + Target_3),
    non_scholarship_dropout_rate = total_non_scholarship_dropouts / total_non_scholarship_students
  )
non_scholarship_holder_rate
# This finding suggests financial support is a strong incentive to not dropout.


### Gender Comparison to Dropout
  #Although there are significantly more men (2278) than women (1262), in the dataset, they have roughly the same number of drop outs (men = 556 & women = 569).
  # As a result, the rate of female_drop out is very high 45% for women vs 24% for men.
  # This could be sampling error and possibly unique to the dataset which is taken from the following diverse degree programs:
  #"agronomy, design, education, nursing, journalism, management, social service, and technologies" (Martins et al., 2021).
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

### International Students & Dropout
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

### Students with Special Needs & Dropout
# Although there were not many students with special needs in the dataset, the 
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
## Exploring Continuous Variables

semester_variables <- variable_table %>%
  slice(22:33) %>%
  mutate(
    Semester = if_else(str_detect(variable_name, "1st_sem"), "1st Semester", "2nd Semester")
  )
print(semester_variables %>% select(variable_name, Semester))
# As seen above the curricular unit variables give us 6 looks at how the students stand each semester.
# - The number of credits earned
# - The number of units enrolled in
# - The number of evaluations/ tests taken
# - The number of units they were approved to take
# - Their Grade avg 
# - The number of units that didn't have tests/evaluations (conceivably easier courses.)

# Identify trends
# Group the continuous variables by semester and summarize their statistics
semester_summary <- traindata %>%
  select(all_of(semester_variables$variable_name)) %>%
  pivot_longer(everything(), names_to = "variable_name", values_to = "value") %>%
  left_join(semester_variables, by = "variable_name") %>%
  group_by(Semester, variable_name) %>%
  summarize(
    Mean = mean(value, na.rm = TRUE),
    Median = median(value, na.rm = TRUE),
    SD = sd(value, na.rm = TRUE),
    Min = min(value, na.rm = TRUE),
    Max = max(value, na.rm = TRUE),
    .groups = "drop"
  )
print(semester_summary)
# The summary above shows teh following
# On average students aer approved for 4.7 to 4.4 units each semester, but enroll 6.25 to 6.22, and are only credited .6 to .5 in the first and second semester, respectively.
  # this finding suggests many students are not succeeding in passing their course.
# Which is futher eemplified in an average greade of 10.3/20 in sem 1 and sem2. which is only .3 on average above the minimum passing grade.
# most course offer atleast a few evalauations (evident by .14 -.15 avg units without an eval each semesters)

# The following code visualizes the distribution of each of these variables.
# Boxplot of variables grouped by semester
units_distributed<- traindata %>%
  select(all_of(semester_variables$variable_name)) %>%
  pivot_longer(everything(), names_to = "variable_name", values_to = "value") %>%
  left_join(semester_variables, by = c("variable_name" = "variable_name")) %>%
  ggplot(aes(x = variable_name, y = value, fill = Semester)) + # Match exact case of "Semester"
  geom_boxplot() +
  facet_wrap(~ Semester, scales = "free") + # Ensure "Semester" column exists and matches case
  labs(
    title = "Distribution of Continuous Variables by Semester",
    x = "Variables",
    y = "Values",
    fill = "Semester"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
units_distributed
# most clearly, students on average are earning 0 credits, which may be more descriptive of graduates
# who are no longer earning credits, but also telling of students who might have dropped out, more than students enrolled (which is the smallest group)
# without more context into what units credited means, it's hard to draw conclusions.

# Again, average grades are only just above 10/20 in semester 1, with some more variablility in semester 2, but a similar average.


################################################################################
# Categorical variables, specifically nationality[8], Previous education, previous education grade, degree programs, and courses

### Nationality
# Summarize nationality distribution in the dataset
nationality_summary <- traindata %>%
  group_by(nacionality) %>%
  summarize(
    count = n(),
    percentage = (n() / nrow(traindata)) * 100
  ) %>%
  arrange(desc(count))
print(nationality_summary)

# Pull description for nationality (row 8) in the variable table
variable_description <- variable_table %>%
  slice(8) %>% 
  select(variable_name, description) %>%
  mutate(description = str_wrap(description, width = 80)) 
description_text <- variable_description$description
cat(description_text)
# Above shows over 97% of the dataset is Portuguese. An appropriate finding for a Portugal based university.


### Previous Educational Achievement
# Summarize previous_qualification (index 6)
previous_education_summary <- traindata %>%
  group_by(previous_qualification) %>%
  summarize(
    Count = n(),
    Percentage = (n() / nrow(traindata)) * 100
  ) %>%
  arrange(desc(Count))
print(previous_education_summary)

# Pull description for nationality (row 6) in the variable table
variable_description <- variable_table %>%
  slice(6) %>% 
  select(variable_name, description) %>%
  mutate(description = str_wrap(description, width = 80)) 
description_text <- variable_description$description
cat(description_text)
# Result above show that 84% of students in the sample have recently come from secondary education (1) AKA High Schools. 
# suggesting the sample describes largely Freshman and Sophomore students who had not had other more recent achievements, like the 5% with Technological specialization course (39).
# notably,~4%, the third largest group, is coming with 9,10,11th grade completed, but not a secondary level degree. Suggesting these students are taking 
# college credits. This group may perform better or worse than others and could be a useful feature. Below confirms the age of the sample is aligns with first and second year univeristy students:
# Summarize age_at_enrollment
age_summary <- traindata %>%
  group_by(age_at_enrollment) %>%
  summarize(
    Count = n(),
    Percentage = (n() / nrow(traindata)) * 100
  ) %>%
  arrange(desc(Percentage))
print(age_summary)
# about half the sample is 18 or 19, with 80% falling between 18 and 27.

### Previous Education Grade
# Summarize previous_qualification_grade (index 7)
grade_summary <- traindata %>%
  summarize(
    Min_Grade = min(previous_qualification_grade, na.rm = TRUE),
    Max_Grade = max(previous_qualification_grade, na.rm = TRUE),
    Mean_Grade = mean(previous_qualification_grade, na.rm = TRUE),
    Median_Grade = median(previous_qualification_grade, na.rm = TRUE),
    SD_Grade = sd(previous_qualification_grade, na.rm = TRUE)
  )
print(grade_summary)
# Pull description for previous education grade (row 7) in the variable table
variable_description <- variable_table %>%
  slice(7) %>% 
  select(variable_name, description) %>%
  mutate(description = str_wrap(description, width = 80)) 
description_text <- variable_description$description
cat(description_text)
# The average grade coming into university was about 133/200, with a standard deviation of 13.

### Courses
# Investigating the distribution of courses [row 4] taken in the sample will contribute to understanding the study and where possible student
# support is need
# Summarize the course variable
course_summary <- traindata %>%
  group_by(course) %>%
  summarize(
    Count = n(),
    Percentage = (n() / nrow(traindata)) * 100
  ) %>%
  arrange(desc(Percentage)) # Sort by Percentage in descending order
print(course_summary)
# Sum the Count column
total_students <- course_summary %>%
  summarize(Total_Count = sum(Count))
print(total_students)
# Pull description for course (row 4) in the variable table
variable_description <- variable_table %>%
  slice(4) %>% 
  select(variable_name, description) %>%
  mutate(description = str_wrap(description, width = 80)) # Adjust width as needed
# Convert to data frame and print the description using cat
description_text <- variable_description$description
cat(description_text)
# Above shows that out of 17 courses and 3540 students, the majority (17%) are in the nursing program
# 8.5% in Management, 8% in social services, and ~8%  in both veterinary nursing and Journalism and communication.
# The rst of the course have 6% - .2% each. These results suggest a good distribution across fields of study.


################################################################################
### Exploratory Data Analysis Summary:

# - Examined binary variables (e.g., debtor status, scholarship holder, gender) and their distributions.
# - Identified that scholarship holders have a significantly lower dropout rate (12%) compared to non-scholarship holders (38%).
# - Noted that international students have a dropout rate similar to the overall rate (~30%), despite being a predictive variable.
# - Analyzed semester-wise academic performance, observing that dropouts have lower average grades compared to graduates.
# - Visualized data using bar charts and boxplots to highlight key differences across groups.

# Based on the insights from the exploratory analysis, the next section engineers new features.
# These features will capture important factors such as financial status, academic performance, and course difficulty,
# enhancing the predictive power of the models to be built.


################################################################################
## Feature Development
################################################################################

# Feature development is the practice of getting the most out of a dataset by introducing new
# variables. The feature engineering function below creates 26 new features in 10 categories. 
# The description of each category of features is found at the end of each chunk of code.

feature_engineering <- function(data) {
  
  # 1. Grading Scale Mapping (Numeric Categories)
  map_grades_to_category <- function(grade) {
    case_when(
      grade >= 18 & grade <= 20 ~ 5,  # Very Good with Distinction
      grade >= 16 & grade < 18 ~ 4,   # Very Good
      grade >= 14 & grade < 16 ~ 3,   # Good
      grade >= 10 & grade < 14 ~ 2,   # Sufficient
      grade >= 7 & grade < 10 ~ 1,    # Poor
      grade < 7 ~ 0,                  # Very Poor
      TRUE ~ NA_real_                 # Handle missing values
    ) 
  }
  data <- data %>%
    mutate(
      grade_category_1st_sem = map_grades_to_category(curricular_units_1st_sem_grade),
      grade_category_2nd_sem = map_grades_to_category(curricular_units_2nd_sem_grade)
    ) # The grade_categories features use the Portuguese grading standards to define 6 ordinal academic performance categories.
  
  # 2. Owe Money Feature
  data <- data %>%
    mutate(
      owe_money = ifelse(debtor == 1 & tuition_fees_up_to_date == 0, 1, 0)
    ) # The owe_money feature attempts to create meaning out of students who are financially strained vs. those who are not.
  
  # 3. Failing Features
  data <- data %>%
    mutate(
      failing_first_sem = ifelse(grade_category_1st_sem <= 1, 1, 0),
      failing_both_semesters = ifelse(grade_category_1st_sem <= 1 & grade_category_2nd_sem <= 1, 1, 0)
    ) # These failing semester features define class failure vs. pass by using the grade_category in feature #1.
  
  # 4. Under-Enrolled
  data <- data %>%
    mutate(
      under_enrolled_1st_sem = ifelse(curricular_units_1st_sem_enrolled < 10, 1, 0),
      under_enrolled_2nd_sem = ifelse(curricular_units_2nd_sem_enrolled < 10, 1, 0),
      under_enrolled_both_sem = ifelse(under_enrolled_1st_sem == 1 & under_enrolled_2nd_sem == 1, 1, 0)
    ) # The under-enrolled feature identifies students who are off track for graduating on time by 
  # using a threshold of 50% of the required pace. To graduate in 4 years with 180 credits, 
  # students need 22.5 credits per semester. This feature flags students enrolled in fewer than 
  # half that amount (10 credits in the 1st semester and 2nd semester).
  
  # 5. Course Difficulty
  data <- data %>%
    group_by(course) %>%
    mutate(
      avg_grade_1st_sem = mean(curricular_units_1st_sem_grade, na.rm = TRUE),
      std_grade_1st_sem = sd(curricular_units_1st_sem_grade, na.rm = TRUE),
      avg_grade_2nd_sem = mean(curricular_units_2nd_sem_grade, na.rm = TRUE),
      std_grade_2nd_sem = sd(curricular_units_2nd_sem_grade, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    mutate(
      overall_avg_grade = (avg_grade_1st_sem + avg_grade_2nd_sem) / 2,
      overall_std_grade = (std_grade_1st_sem + std_grade_2nd_sem) / 2
    ) # The course difficulty feature quantifies how challenging a course is based on all student grades in the sample. 
  
  # 6. Hard Courses
  data <- data %>%
    mutate(
      hard_courses = ifelse(curricular_units_1st_sem_evaluations > 10, 1, 0)
    ) # Hard courses assume that many evaluations/tests make a course harder.
  
  # 7. Discouraging Courses
  data <- data %>%
    group_by(course) %>%
    mutate(
      avg_course_grade = mean(curricular_units_1st_sem_grade, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    mutate(
      discouraging_courses = ifelse(avg_course_grade < 10, 1, 0)
    )# Courses with low average grades (below the passing threshold of 10) are flagged as discouraging.
  
  # 8. Historical Success
  data <- data %>%
    mutate(
      normalized_previous_grade = previous_qualification_grade / 200,
      strong_historical_success = ifelse(normalized_previous_grade >= 0.663, 1, 0),
      weak_historical_success = ifelse(normalized_previous_grade < 0.5966, 1, 0)
    ) # Uses normalized thresholds to categorize students with strong or weak prior success before coming to university.
  
  # 9. Grade-Based Summaries
  data <- data %>%
    mutate(
      average_grade = (curricular_units_1st_sem_grade + curricular_units_2nd_sem_grade) / 2,
      semester_grade_gap = curricular_units_2nd_sem_grade - curricular_units_1st_sem_grade
    ) # Summarizes overall grade performance and tracks grade changes between semesters.
  
  # 10. Prior Education Group Distributions
  data <- data %>%
    group_by(previous_qualification) %>%
    mutate(
      prior_avg_grade_1st_sem = mean(curricular_units_1st_sem_grade, na.rm = TRUE),
      prior_std_grade_1st_sem = sd(curricular_units_1st_sem_grade, na.rm = TRUE),
      prior_avg_grade_2nd_sem = mean(curricular_units_2nd_sem_grade, na.rm = TRUE),
      prior_std_grade_2nd_sem = sd(curricular_units_2nd_sem_grade, na.rm = TRUE)
    ) %>%
    ungroup() 
  # Tracks grade patterns for students by prior education level.
  
  return(data)
}


# Apply to training data
traindata <- feature_engineering(traindata)


################################################################################
## Validate Feature Development on Training Data 
################################################################################
# This section validates that the feature engineering function above was effective and
# and didn't introduce NA's into the dataset. 

# The following code prints the new size of the datasets:
ncol(traindata)
colnames(traindata)


# The following function does 4 key things to evaluate the newly engineered features:
#- 1. Count total NAs in the dataset
#- 2. Count total NAs in each feature
#- 3. Count if any features are 100% 0 or 1 to indicate a lack of predictive power
#- 4. Plots 8 of a few of the 26 new features

validate_features <- function(data) {
  # 1. Count Total NAs in the Dataset
  total_na <- sum(is.na(data))
  cat("Total NA Count in Dataset:", total_na, "\n")
  
  # 2. Count NAs per Feature
  na_counts <- data %>%
    summarise(across(everything(), ~ sum(is.na(.)))) %>%
    pivot_longer(everything(), names_to = "Feature", values_to = "NA_Count") %>%
    arrange(desc(NA_Count))
  
  print("NA Counts for Each Feature:")
  print(na_counts)
  
  # 3. Identify Features with 100% 0s or 1s
  constant_features <- data %>%
    summarise(across(everything(), ~ all(. == 0, na.rm = TRUE) | all(. == 1, na.rm = TRUE))) %>%
    pivot_longer(everything(), names_to = "Feature", values_to = "Is_Constant") %>%
    filter(Is_Constant == TRUE)
  
  if (nrow(constant_features) > 0) {
    cat("Features with 100% 0s or 1s:\n")
    print(constant_features$Feature)
  } else {
    cat("No features have 100% 0s or 1s.\n")
  }
  
  # 4. Visualize Key Features
  key_features <- c(
    "owe_money", 
    "failing_first_sem", 
    "under_enrolled_2nd_sem", "under_enrolled_both_sem", 
    "discouraging_courses", "weak_historical_success", 
    "average_grade", "semester_grade_gap"
  )
  
  feature_plots <- list()
  for (feature in key_features) {
    if (feature %in% names(data)) {
      p <- ggplot(data, aes_string(x = feature)) +
        geom_bar(fill = "steelblue", color = "black") +
        labs(
          title = paste("Distribution of", feature),
          x = feature,
          y = "Count"
        ) +
        theme_minimal()
      feature_plots[[feature]] <- p
    }
  }
  
  # Return validation results
  return(list(
    na_counts = na_counts,
    total_na = total_na,
    constant_features = constant_features,
    feature_plots = feature_plots
  ))
}
# Apply the validation function
validation_train_results <- validate_features(traindata)

# Total NAs in the dataset
cat("Total NA Count:", validation_train_results$total_na, "\n")

# NA counts per feature
print(validation_train_results$na_counts)
# The dataset only has 2 NAs. These two instances can be removed for simplicity.

# Features with 100% 0s or 1s
if (nrow(validation_train_results$constant_features) > 0) {
  cat("Features with constant values (100% 0s or 1s):\n")
  print(validation_train_results$constant_features$Feature)
} else {
  cat("No features have 100% 0s or 1s.\n")
}
# no features appear to have no predictive value by being entirely 0 or 1

# View the plots for key features
for (feature_name in names(validation_train_results$feature_plots)) {
  print(validation_train_results$feature_plots[[feature_name]])
}
# Distribution of Semester gap: shows most students have no little difference in grades with a normal distribution around 0
  # with some outliars at -10 and + 10, which may point to dropouts or students who return from a break.
# Distribution of Average grade: shows largest cluster between 10/20 and 15/20, with a very large concentration at 0, and a smaller concentration between 5 and 10
  # suggesting some students are in trouble of failing or simply not taking courses.
# Distribution of weak historical success shows majority of students performed well in past, but about 500 did not
# Discouraging courses show about 100 students have taken classes that led to them receiving a failing grade (below 10).
# Strangely, most students fall into the category of being under enrolled in both semesters (less than 10 credits). 
  # This may be due to missing domain knowledge on part of the author.
# Distribution of failing first semester shows about 500 students are getting grades below 10/20.
# Owe money distribution is not capturing many students, but since tuition fees up to date was most predictive of target
# in the regression before, it will be retained.

# With the feature development successful at generating new features, a final step includes
# removing the 2 NAS that were generated"
traindata <- traindata %>% drop_na()
# Validate the dataset no longer contain NAs
cat("Total NA Count in Training Dataset after removal:", sum(is.na(traindata)), "\n")

# This feature validation section shows that the number of features has been nearly doubled. This could
# increase predictive power, but could also increase the likelihood of over-fitting. To avoid over fitting
# steps will be taken in modeling, like cross validation.


################################################################################
## Choosing Classification 
################################################################################
# This section justifies classification as the task for predicting student outcoems.

# Now that the dataset has been loaded, cleaned, and split into training and test sets, 
# followed by exploratory analysis and the development of features to maximize the information within the dataset, 
# it’s time to start modeling an algorithm to predict college outcomes. 
# The developed features have been validated and applied consistently to both the training and test sets.

# Since the college outcomes in this dataset are categorized into three distinct groups—dropout, 
# enrolled, and graduated—a classification approach is the most suitable modeling strategy.
# Classification enables the algorithm to learn from the dataset and classify each student 
# into one of these three outcome groups based on the provided features. 
# This approach aligns well with the structure of the target variable and 
# ensures predictions are interpretable and actionable.

################################################################################
### Splitting 'traindata' for Model Training and Evaluation

# As before, the dataset in split in 80% for training and 20% for training to preseve as much of the trainingset as possible,
# while training 20% for validation.

# Partition traindata
set.seed(456)
train_model_index <- createDataPartition(traindata$target, p = 0.8, list = FALSE)
training_set <- traindata[train_model_index, ]   # Model training set
validation_set <- traindata[-train_model_index, ] # Validation set for model eval


# Convert target to a factor in the training and evaluation sets
training_set$target <- as.factor(training_set$target)
validation_set$target <- as.factor(validation_set$target)


################################################################################
## Model #1 Decision Tree Classification
################################################################################
### Train the Decision Tree Model

# Training
decision_tree_model <- rpart(
  target ~ ., 
  data = training_set, 
  method = "class"
)

## Predict on the validation set
dt_predictions <- predict(decision_tree_model, validation_set, type = "class")

## Evaluate the model
dt_conf_matrix <- confusionMatrix(dt_predictions, validation_set$target)
print(dt_conf_matrix)

#The model performs well in predicting students who will drop out and graduate:
  #- **81% of actual dropouts were correctly identified as dropouts (Sensitivity = 0.8122).**
  #- **73% of predicted graduates were correct (Positive Predictive Value = 0.7304).**
  
#However, there are notable challenges:
  #- The model entirely fails to predict the "enrolled" class (Sensitivity = 0, Precision = NaN). 
  #This suggests a lack of information or patterns in the data for this group or is the impact of class imbalance (as seen below)
  #- A concerning misclassification occurs where **43 students who dropped out were classified as graduates. 
  #This is problematic because misclassifying a dropout as a graduate could lead to overlooking at-risk students needing intervention.

## Visualize the Decision Tree
rpart.plot(decision_tree_model, type = 2, extra = 104)
# As seen in the decision tree, the model used two key variables to make it's classifications: 
#- curricular_units_2nd_sem_approved
#- tuition_fees_up_to_date.
# These splits effectively separated dropouts (Class 1) and graduates (Class 3), 
# but struggled to distinctly classify enrolled (Class 2) students, 
# as evidenced by low proportions of Class 2 in terminal nodes. 
# Unfortunately, this means the model decided the rest of the features were not helpful. 
# This is called overfitting when the model assumes the other variables are just noise and not worth listening to.
# Another key takeaway of this model, as seen in the decision tree, is that it tells counselors to focus on students who
# are not approved for at least 4 credits in second semester, and who do not have their
# tuition fees up to date as these students are most likely to be dropout. 


################################################################################
## Model #2: Gradient Boosted Trees with Cross-Validation
################################################################################
# In order to account for the class imbalance, boosted trees increase the number of 
# iterations and samplings for enrolled AKA Target =2. Additionally, to account for the many features,
# this model uses cross validation as the method.

## Define training control for cross-validation
train_control_gbt <- trainControl(
  method = "cv",          # k-fold cross-validation
  number = 5,             # folds
  verboseIter = TRUE
)

## Define the grid of hyperparameters to tune
gbt_grid <- expand.grid(
  n.trees = seq(100, 700, by = 100),  # Number of boosting iterations
  interaction.depth = c(1, 2, 3),     # Max depth of each tree
  shrinkage = c(0.05, 0.075, 0.1),    # Learning rate
  n.minobsinnode = c(5, 10)           # Minimum number of observations in a node
)

## Train the Gradient Boosted Trees model (This will take a few minutes)
set.seed(123)
gbt_model <- train(
  target ~ ., 
  data = training_set, 
  method = "gbm", 
  trControl = train_control_gbt, 
  tuneGrid = gbt_grid,
  verbose = FALSE
)
# Thanks to cross validations 5 folds, the model is trained on 4 folds and validated on the 5th.
# This ensures that all data is used for both training and validation.
# Boosting reweights categories that are misclassified within the folds to handle class imbalance (e.g., Target = 2)

# Print the best model parameters from cross-validation
cat("Best Model Parameters from Cross-Validation:\n")
print(gbt_model$bestTune)

## Predict on the validation set
gbt_predictions <- predict(gbt_model, validation_set)

## Evaluate the model
gbt_conf_matrix <- confusionMatrix(gbt_predictions, validation_set$target)
print(gbt_conf_matrix)

# Dropout (Class 1)
# The GBT model achieves a sensitivity of 75.55% for predicting dropouts, 
# indicating strong performance in identifying students who are likely to drop out.
# Precision (Positive Predictive Value) for this class is 82.38%, showing good reliability 
# in the model's dropout predictions. These results suggest that the GBT model balances 
# sensitivity and precision effectively for the critical dropout category.

# Enrolled (Class 2)
# For the enrolled class, sensitivity improves to 41.60% compared to 0% in the Decision Tree model.
# While this is an improvement, it still reflects challenges in accurately identifying enrolled students.
# Specificity remains high at 92.44%, indicating the model's ability to correctly exclude non-enrolled students.
# Precision (Positive Predictive Value) for this class is 54.17%, showing moderate reliability 
# in enrolled predictions. The results suggest the GBT model partially addresses class imbalance, 
# though further improvements are needed for this minority class.

# Graduate (Class 3)
# The GBT model shows strong sensitivity for graduates at 92.07%, effectively identifying students likely to graduate.
# Specificity for this class is 78.53%, with a precision of 81.05%, indicating good reliability in graduate predictions.
# These results maintain strong performance for the graduate class, with slight improvements in specificity 
# over the Decision Tree model, albeit with a small trade-off in sensitivity.

# Visualize the performance of the Gradient Boosted Trees model
plot(gbt_model)
# The plot shows that boosting iterations (x-axis) did greatly impact accuracy (as seen on y axis)
# fora ll three outcomes (dropout=1, enrolled =2, graduated =3).
# Cross validation automatically picked the best tune that optimizes for the best outcoems across all 2.
# The chart shows that the three values align best around 200 trees, a depth of 3, shrinkage of .05
# and 10 observations per node.

# Overall, the GBT model outperformed the Decision Tree model, 
# particularly in its ability to predict the enrolled class. 



################################################################################
# Results
################################################################################
colnames(final_holdout_set)

# Results Section

# With two models developed and tested on the training sets' validation_set,
# it's now time to test each model on the final_holdout_set

# Step 1: Set Target as factor (consider moving above of feature engineering)
# Ensure the target variable in the final_holdout_set is a factor
final_holdout_set$target <- as.factor(final_holdout_set$target)

# Step 2: Feature Engineering
# Apply feature engineering function to final_holdout_set
final_holdout_set <- feature_engineering(final_holdout_set)

# Apply the feature engineering validation function
validation_holdout_results <- validate_features(final_holdout_set)


#There are 8 NAs. A negligable amount. So they are dropped
cat("Total NA Count in final_holdout_set  before removal:", sum(is.na(final_holdout_set)), "\n")
final_holdout_set <- final_holdout_set %>% drop_na()
cat("Total NA Count in final_holdout_set after removal:", sum(is.na(final_holdout_set)), "\n")



# Step 3: Predict with Decision Tree Model
dt_holdout_predictions <- predict(decision_tree_model, final_holdout_set, type = "class")
dt_holdout_conf_matrix <- confusionMatrix(dt_holdout_predictions, final_holdout_set$target)
cat("Decision Tree Results on Final Holdout Set:\n")
print(dt_holdout_conf_matrix)

# Step 4: Predict with Gradient Boosted Trees Model
gbt_holdout_predictions <- predict(gbt_model, final_holdout_set)
gbt_holdout_conf_matrix <- confusionMatrix(gbt_holdout_predictions, final_holdout_set$target)
cat("Gradient Boosted Trees Results on Final Holdout Set:\n")
print(gbt_holdout_conf_matrix)
# The results above show, in terms of balanced accuracy, the boosted tree model was able to maintain similar performance to the DT model
# in predicting dropout (GBT = .81 vs DT = .81) and enrolled (GBT = .85 vs DT = .80)  while making up 15% in enrolled (GBT = .65 vs DT = .50).
# This difference is certainly attributed to cross validations and the boosted resampling. 



## Step 5: Visualize GBT & Decision Tree Performance on Final Holdout

# Function to plot confusion matrix heatmap
plot_confusion_matrix <- function(conf_matrix, model_name) {
  conf_matrix_df <- as.data.frame(conf_matrix$table)
  ggplot(conf_matrix_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "white", fontface = "bold") +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    labs(
      title = paste("Confusion Matrix Heatmap for", model_name, "Model"),
      x = "Actual Class",
      y = "Predicted Class"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    )
}

# Visualize Decision Tree Model
dt_conf_matrix_plot <- plot_confusion_matrix(dt_holdout_conf_matrix, "Decision Tree")
print(dt_conf_matrix_plot)
# Represented by the darker colors in the top right and bottom left, the model greatly favored enrollment and dropout, and completely missed enrolled.
  # This performance was expected and negligably different from the DT performance on the validation set:
    #- validation set results =  balanced accuracy of  0.8423   0.5000   0.8008 in Dropout, enrolled, and graduate, respectively.
    # Final_holdout_set results = 0.8133    0.500   0.8011 

# Visualize Gradient Boosted Trees Model
gbt_conf_matrix_plot <- plot_confusion_matrix(gbt_holdout_conf_matrix, "GBT")
print(gbt_conf_matrix_plot)
# The GBT heat map shows slightly lower correct predictions for enrolled and dropout compared to DT, 
  # but actually made predictions for enrolled, making it the stronger of the two models.


################################################################################
# Conclusion
################################################################################
# Sumamary

# In conclusion, this report has cleaned the dataset, explored the variables, built new features, and developed two mmachine learning models
# to predict student outcomes at university (dropout, enrolled, graduate). The two models developed were decision trees & gradiaent boosted trees.
# The performance of these two models are mediocre at best. Due to class imbalance, precision/pos predictive value is one of the best measures of success. 
# As shown above, GBT performs much better than decision trees in this regaard (DT = 0.7267      NaN   0.7329) vs (GPT = 0.7695  0.50000   0.8148).
# This difference is largely due to cross validation and the use of boosted sampling for mis-classified target in it's training. 
# However, with GBT only identified 38% of enrolled students (sensitivity), as a result, the model is left wanting. 
# On the positive side, enrolled is the least important group for prediction since it is the most easily identified in the present. Notably, 
# accurately predicting 74% of dropouts and 90% (sensitivity) of graduates are strong outcomes and suggests that with further refinement
# as discussed in the following section greater outcomes can be achieved.

## Limitations
 # This data doesn't include year over year data resulting in limited conclusion ability based
  # on annual distribution of enrolled, graduate, and drop out.
  # don't know the university? maybe we do? If we don't then we don't know what financial support looks like (maybe future direction)
  # Don't know for certain without further digging which courses belong to which degree program. Making this connection would contribute to knowing the unique suport needs of each degree path.
  # There are too many features and not enough data. 
ncol(traindata)
nrow(traindata)
table(traindata$target)
# With 62 features interacting to predict educational outcomes on a training dataset with 3539 students, 
# only 647 currently enrolled, vs 1124 dropout and 1768 graduate both models were over-fitting due to this low prevelance.

# Most students on average were earning 0 credits, which may reflect graduates who are no longer earning credits or students who have dropped out. 
# This pattern may not accurately represent currently enrolled students, who are the smallest group. Without more context about what "units credited" means, 
# it is difficult to draw clear conclusions.

## Future Directions
  # There were many variables that were not further explored because their predictability in their current form wasn't strong enough to warrant immediate exploration and feature development.
  # For instance, all the variables that fell along 0 in the regression plot below were largely left alone, leading to opportunity to develop new feature not yet explored.
  # Specially, creating demographic features that describe parent occupation and previous education achievement (qualification) could increase or decrease likelihood of success in University.

  # In order to address class imbalance, additional strategies can be employed like SMOTE or weighted loss functions) could further improve performance on the minority class.
  # Neither model explicitly used techniques like oversampling, undersampling, or class-weighted learning to fully address class imbalance (is this statement true of the gradient boosted method employed above?)
  # Additionally, research should test an elastic net model or lasso to more strategically select which features to include. Another future direction is 
  # to add 

  # it would be interesting to change this to a 2 classification problem only seeking the end of college outcome of 
  # of graduate or dropout. 

  # Lastly, a probabilistic approach like multinomial linear regression would be a logical next step to inform
  # counselors and stakeholders of the probability each student has of falling into the dropout category.
  # This would give counselors information they could act on.


################################################################################
# Citations
# Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.
# U.S. Department of Education, National Center for Education Statistics, Integrated Postsecondary Education Data System (IPEDS), Winter 2020–21, Graduation Rates component. See Digest of Education Statistics 2021, table 326.20. Retreieved on 11/25/24 from https://nces.ed.gov/fastfacts/display.asp?id=40
# US Department of Health and Human Services. (2023). Enrollment in higher education. Enrollment in Higher Education - Healthy People 2030. https://odphp.health.gov/healthypeople/priority-areas/social-determinants-health/literature-summaries/enrollment-higher-education 