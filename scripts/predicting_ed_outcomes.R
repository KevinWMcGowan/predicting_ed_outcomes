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
packages <- c("readr", "gbm", "tinytex", "dplyr", "caret", "rpart", "stringr", "tidyverse", "tidyr", "broom", "glmnet", "Matrix","coefplot", "here")

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

################################################################################
# Clean the datasets
################################################################################

# As seen below, the column names in data table and the variable and the variable_name column
  #include spaces and parenthesis. All of which will cause issues with analysis later on.
cat("Column names in `data` before cleaning:\n")
colnames(data)
cat("\nColumn names in `variable_table` before cleaning:\n")
colnames(variable_table)
cat("\nVariable names in `variable_table` before cleaning:\n")
print(variable_table$`Variable Name`)


# In order to remove these problematic characters, the following regex code removes them.
# Define a function to clean column names by removing problematic characters
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


# Apply the cleaning function to `data` column names
colnames(data) <- clean_names(colnames(data))

# Apply the cleaning function to `variable_table` column names
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

# The next cleaning step is to set variable types. Luckily, the variable_table tells us
# how to code each variable type for analysis. The following code shows expected type (variable table) vs
# the type currently in the data.

# Extract the actual types from the data
data_types <- data.frame(
  variable_name = colnames(data),   
  data_type = sapply(data, class), 
  stringsAsFactors = FALSE
)

# Extract expected types from the variable_table
lookup_types <- variable_table %>%
  select(variable_name, expected_type = type)

# Join the actual types with the expected types
comparison <- data_types %>%
  left_join(lookup_types, by = "variable_name") 

# Display the comparison
cat("\nComparison of Actual and Expected Types:\n")
print(comparison)

# Since R's equivalent of a "continuous" variable type is "numeric", the following changes are made:
# - `curricular_units_1st_sem_grade` is changed from numeric to integer.
# - The `target` variable is replaced with numeric values for regression analysis.
data$curricular_units_1st_sem_grade <- as.integer(data$curricular_units_1st_sem_grade)`
str(data[c("curricular_units_1st_sem_grade")])
# Above shows column is now stored as integers, aligning with the expected variable type in the lookup table.

# Since target is the only variable not represented with a number value, 
  # the following encodes 'target' variable with a numeric representation
data$target <- ifelse(data$target == "Dropout", 1,
                      ifelse(data$target == "Enrolled", 2,
                             ifelse(data$target == "Graduate", 3, NA)))
sort(unique(data$target))
# The `target` variable is now numeric:
# "Dropout" -> 1
# "Enrolled" -> 2
# "Graduate" -> 3


################################################################################
## Split the dataset

# Now before any analysis or exploration can be done, as best practice the dataset is split in order
# to avoid over training.
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
summary(value1)#prolly don't need this summary... chatgpt help me make sense of this summary

# Visualize the coefficients
regression_plot<-coefplot(value1, sort = 'magnitude', conf.int = TRUE)
regression_plot
# The coefficient plot above provides insights into the variables' relationships 
# with the target outcome and their relative strengths. It highlights variables 
# with significant predictive power, as well as those with wide error margins 
# crossing zero, suggesting limited or no effect.

# Key Findings:
# - **Strong Predictors (Positive Association):**

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


# Select specific rows and print only the Variable_Name and Description columns
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



#compare above to dropout and graduate

# Filter and summarize for dropout students
dropout_summary <- traindata %>%
  filter(target == 1) %>%
  select(all_of(continuous_vars)) %>%
  pivot_longer(cols = everything(), names_to = "Variable_Name", values_to = "Value") %>%
  mutate(Semester = if_else(str_detect(Variable_Name, "1st"), "1st Semester", "2nd Semester")) %>%
  group_by(Semester, Variable_Name) %>%
  summarize(
    Mean_Dropout = mean(Value, na.rm = TRUE),
    .groups = "drop"
  )

# Filter and summarize for graduated students
graduated_summary <- traindata %>%
  filter(target == 3) %>%
  select(all_of(continuous_vars)) %>%
  pivot_longer(cols = everything(), names_to = "Variable_Name", values_to = "Value") %>%
  mutate(Semester = if_else(str_detect(Variable_Name, "1st"), "1st Semester", "2nd Semester")) %>%
  group_by(Semester, Variable_Name) %>%
  summarize(
    Mean_Graduated = mean(Value, na.rm = TRUE),
    .groups = "drop"
  )

# Merge the summaries and calculate differences
dropout_with_diff_summary <- dropout_summary %>%
  left_join(graduated_summary, by = c("Semester", "Variable_Name")) %>%
  mutate(Difference_Graduated_vs_Dropout = Mean_Graduated - Mean_Dropout) %>%
  select(Semester, Variable_Name, Mean_Dropout, Mean_Graduated, Difference_Graduated_vs_Dropout)

# Print the final table
print(dropout_with_diff_summary)


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



# Exploratory Data Analysis Summary:

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

#- create a owe money feature of students who both are debtor = 1 and tuition feeds up to date = 0 
# create students who failed first semester with 1st Semester curricular_units_1st_sem_grade  =>10 (50% of 20) 
# students who failed both first and second semester 2nd Semester curricular_units_2nd_sem_grade  =>10 & 1st Semester curricular_units_1st_sem_grade  =>10
# under enrolled
# create course ID to semester grade, and semester units credited to get idea of harder courses. maybe even discouraging courses when taken first semester and then lead to student enrollment of 0 the next semester.
# taking a break students who 2nd semester unit enrollment enroll = 0
#hard courses could also be a feature where user had high number of unit 1st semeseter evalautions
# graduates could be earning 0 credits  in first or second semester because they are graduated. Maybe this could be a feature 

# Define a feature engineering function:
combined_feature_engineering <- function(data) {
  
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
    ) # The grade_categories features use the Portuguese grading standards to define clear pass and fail, along with exemplary performance.
  
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
    ) # The course difficulty feature quantifies how challenging a course is based on student grades. 
  
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
    ) # Uses normalized thresholds to categorize students with strong or weak prior success.
  
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


### Separately apply the combined function to the training data and test data
# Apply to training data
traindata <- combined_feature_engineering(traindata)

# Apply to final_holdout_set
final_holdout_set <- combined_feature_engineering(final_holdout_set)


################################################################################
## Validate Feature Development on Training Data 
################################################################################
ncol(traindata)
colnames(traindata)

validate_combined_features <- function(data) {
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
    "grade_category_1st_sem", "grade_category_2nd_sem", "owe_money", 
    "failing_first_sem", "failing_both_semesters", "under_enrolled_1st_sem", 
    "under_enrolled_2nd_sem", "under_enrolled_both_sem", "hard_courses", 
    "discouraging_courses", "strong_historical_success", "weak_historical_success", 
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
# Apply the updated validation function
validation_train_results <- validate_combined_features(traindata)

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


# remove the 2 NAs:
# Remove rows with NA values from the training and test datasets
traindata <- traindata %>% drop_na()
# Revalidate the datasets to confirm no remaining NAs
cat("Total NA Count in Training Dataset after removal:", sum(is.na(traindata)), "\n")

# This feature validation section shows that the number of features has been nearly doubled. This could
# increase predictive power, but could also increase the likelihood of over-fitting. To avoid overfitting
# steps will be taken in modeling like cross validation and elastic net.






################################################################################
## Choosing Classification 
################################################################################
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

## Splitting 'traindata' for Model Training and Evaluation
# Step 1. Partition traindata
set.seed(456)
train_model_index <- createDataPartition(traindata$target, p = 0.8, list = FALSE)
training_set <- traindata[train_model_index, ]   # Model training set
validation_set <- traindata[-train_model_index, ] # Validation set for model evaluation


# Step 2. Convert `target` to a factor in the training and evaluation sets
training_set$target <- as.factor(training_set$target)
validation_set$target <- as.factor(validation_set$target)

################################################################################
# Model #1 Decision Tree Classification
################################################################################

## Train the Decision Tree model
# Step 3. Train Decision Tree Model
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
  # As seen in the decision tree, the model used two key variables to make it's classificaitons: curricular_units_2nd_sem_approved and tuition_fees_up_to_date.
  # These splits effectively separated dropouts (Class 1) and graduates (Class 3) but struggled to distinctly classify enrolled (Class 2) students, as evidenced by low proportions of Class 2 in terminal nodes.

################################################################################
# Model Performance
#1. Address class imbalance by oversampling the minority classes (enrolled) or try a class-weighted algorithms or resampling strategies with cross validation to balance the dataset during training. 
#See class imbalance below:
table(training_set$target)
#2. Optimize for higher sensitivity (at the cost of some precision) since it is better to err on the side of predicting a dropout when in doubt by adjusting the decision thresholds or introduce cost-sensitive learning to penalize false negatives for the "dropout" class.
#3. Examine feature importance to identify why the model struggles with the "enrolled" class and refine the features to enhance separability between classes.


################################################################################
# Model #2: Gradient Boosted Trees with Cross-Validation
################################################################################

## Define training control for cross-validation
train_control_gbt <- trainControl(
  method = "cv",          # Use k-fold cross-validation
  number = 5,             # Number of folds
  verboseIter = TRUE
)

## Define the grid of hyperparameters to tune
gbt_grid <- expand.grid(
  n.trees = seq(100, 700, by = 100),  # Number of boosting iterations
  interaction.depth = c(1, 2, 3),     # Max depth of each tree
  shrinkage = c(0.05, 0.075, 0.1),    # Learning rate
  n.minobsinnode = c(5, 10)           # Minimum number of observations in a node
)

## Train the Gradient Boosted Trees model
set.seed(123)
gbt_model <- train(
  target ~ ., 
  data = training_set, 
  method = "gbm", 
  trControl = train_control_gbt, 
  tuneGrid = gbt_grid,
  verbose = FALSE
)

## Predict on the validation set
gbt_predictions <- predict(gbt_model, validation_set)

## Evaluate the model
gbt_conf_matrix <- confusionMatrix(gbt_predictions, validation_set$target)
print(gbt_conf_matrix)

# Optional: Visualize the performance of the Gradient Boosted Trees model
plot(gbt_model)

################################################################################
# GBT Model Performance
print(gbt_conf_matrix)
# While the GBT model shows improved performance in predicting the “enrolled” class (Class 2) compared to the Decision Tree, it still struggled, as evidenced by its sensitivity for Class 2 being only 38.4%.
# This suggests that the GBT model indirectly handles some aspects of class imbalance through its iterative learning process, although not directly.

#The GBT model maintained strong sensitivity for Class 1 (dropout) at 75.98% and improved precision for this class (Positive Predictive Value: 84.88%) compared to the Decision Tree. As a result, the GBT model achieves a good balance of sensitivity and precision for the critical drop out class.
# GBT implementation already demonstrates reasonable performance for sensitivity.

# For  Class 2 (Enrolled), the GPT modele achieving a sensitivity of 38.40%. While this is still relatively low, it is a significant improvement over the Decision Tree model, which failed to predict any enrolled students. The specificity was 92.44%, and the positive predictive value was 52.17%.

# For Class 3 (Graduate), the sensitivity was 92.07%, with a specificity of 75.99% and a positive predictive value of 79.27%. These results are comparable to those of the Decision Tree model, with a slight decrease in sensitivity but improved specificity and precision.

# Overall, the GBT model outperformed the Decision Tree model, particularly in its ability to predict the enrolled class. The improved sensitivity and positive predictive value for enrolled students suggest that the GBT model is better at capturing the nuances in the data that distinguish this class. This enhancement may be attributed to the GBT model’s ability to model complex interactions and its robustness against overfitting due to cross-validation.


################################################################################
# Results
################################################################################

# Results Section

# With two models developed and tested on the trainingsets' partitioned evaluation set,
# it's now time to test each model on the final_holdout_set

# Step 1: Feature Engineering
#apply feature enginnering function to final_holdout_set
combined_feature_engineering(final_holdout_set)
# Apply the updated validation function
validation_holdout_results <- validate_combined_features(final_holdout_set)
cat("Total NA Count:", validation_holdout_results$total_na, "\n")

# no features appear to have no predictive value by being entirely 0 or 1
cat("Total NA Count in final_holdout_set  before removal:", sum(is.na(final_holdout_set)), "\n")
final_holdout_set <- final_holdout_set %>% drop_na()
cat("Total NA Count in final_holdout_set after removal:", sum(is.na(final_holdout_set)), "\n")


# Step 2: Set Target as factor
# Ensure the target variable in the final_holdout_set is a factor
final_holdout_set$target <- as.factor(final_holdout_set$target)

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



################################################################################
# Conclusion
################################################################################


## Limitations
 # This data doesn't include year over year data resulting in limited conclusion ability based
  # on annual distribution of enrolled, graduate, and drop out.
  # don't know the university? maybe we do? If we don't then we don't know what fianncial support looks like (maybe future direction)
  # Don't know for certain without further digging which courses belong to which degree program. Making this connection would contribute to knowing the unique suport needs of each degree path.

## Future Directions
  # There were many variables that were not further explored because their predictability in their current form wasn't strong enough to warent immediate feature development.
  # For instance, all the variables that feel along 0 in the plot below were largely left alone, leading to opportunity to develop new feature not yet explored.
  # Specially, creating demographic features that describe parent occupation and previsou education acheivement (qualification) could increase or decrease liklihood of success in UNiversity.

  #Additional strategies (e.g., SMOTE or weighted loss functions) could further improve performance on the minority class.
  #Neither model explicitly used techniques like oversampling, undersampling, or class-weighted learning to fully address class imbalance.

units_distributed
# most clearly, students on average are earning 0 credits, which may be more descriptive of graduates
# who are no longer earning credits, but also telling of students who might have dropped out, more than students enrolled (which is the smallest group)
# without more context into what units credited means, it's hard to draw conclusions.


regression_plot











################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################################################################################################
################################################################################
################################################################################


# Citations
# SOURCE: U.S. Department of Education, National Center for Education Statistics, Integrated Postsecondary Education Data System (IPEDS), Winter 2020–21, Graduation Rates component. See Digest of Education Statistics 2021, table 326.20. Retreieved on 11/25/24 from https://nces.ed.gov/fastfacts/display.asp?id=40