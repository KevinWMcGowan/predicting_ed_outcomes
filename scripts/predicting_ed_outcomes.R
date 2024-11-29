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
summary(value1)#prolly don't need this summary... chatgpt help me make sense of this summary

# Visualize the coefficients
regression_plot<-coefplot(value1, sort = 'magnitude', conf.int = TRUE)
regression_plot
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
#view(variable_table)
# View descriptions for binary variables

# Select specific rows and print only the Variable_Name and Description columns
variable_table %>%
  slice(c(14:19, 21)) %>%  # Select rows 14 to 19 and 21
  select(Variable_Name, Description) %>%  # Select specific columns
  print()
# 1 = yes & 0 = no for binary values # USE CAT FOR FULL PRINT


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
# Below, compares the same variables against the 3 levels of target ( 0 = dropout, 1 = enrolled, 2 = graduated)
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

semester_variables <- variable_table %>%
  slice(22:33) %>%
  mutate(
    Semester = if_else(str_detect(Variable_Name, "1st_sem"), "1st Semester", "2nd Semester")
  )
print(semester_variables %>% select(Variable_Name, Semester))
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
  select(all_of(semester_variables$Variable_Name)) %>%
  pivot_longer(everything(), names_to = "Variable_Name", values_to = "Value") %>%
  left_join(semester_variables, by = "Variable_Name") %>%
  group_by(Semester, Variable_Name) %>%
  summarize(
    Mean = mean(Value, na.rm = TRUE),
    Median = median(Value, na.rm = TRUE),
    SD = sd(Value, na.rm = TRUE),
    Min = min(Value, na.rm = TRUE),
    Max = max(Value, na.rm = TRUE),
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
traindata %>%
  select(all_of(semester_variables$Variable_Name)) %>%
  pivot_longer(everything(), names_to = "Variable_Name", values_to = "Value") %>%
  left_join(semester_variables, by = "Variable_Name") %>%
  ggplot(aes(x = Variable_Name, y = Value, fill = Semester)) +
  geom_boxplot() +
  facet_wrap(~ Semester, scales = "free") +
  labs(
    title = "Distribution of Continuous Variables by Semester",
    x = "Variables",
    y = "Values",
    fill = "Semester"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#most clearly, students on average are earning 0 credits, which may be more descriptive of graduates
# who are no longer earning credits, but also telling of students who might have dropped out, more than students enrolled (which is the smallest group)

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

#COMPLETE DESCRIPTION/SUMAMRY OF FINDINGS ABOVE AND HOW THEY SUGGEST FEATURE DEVELOPMENT
# The table above shows clearly the difference in 



################################################################################
# Categorical variables, specifically nationality[8], Prvious education, previous education grade, degree programs, and courses

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
  select(Variable_Name, Description) %>%
  mutate(Description = str_wrap(Description, width = 80)) 
description_text <- variable_description$Description
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
  select(Variable_Name, Description) %>%
  mutate(Description = str_wrap(Description, width = 80)) 
description_text <- variable_description$Description
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
  select(Variable_Name, Description) %>%
  mutate(Description = str_wrap(Description, width = 80)) 
description_text <- variable_description$Description
cat(description_text)
# The average grade coming into university was about 133/200, with a standard deviation of 13.


### Degree Program
# Looking into the distribution of study in the sample could give insight to where students perform high or low. However, 
# a degree feature does not exist. Instead, course is provided which will supply some insight.


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
  select(Variable_Name, Description) %>%
  mutate(Description = str_wrap(Description, width = 80)) # Adjust width as needed
# Convert to data frame and print the description using cat
description_text <- variable_description$Description
cat(description_text)
# Above shows that out of 17 courses and 3540 students, the majority (17%) are in the nursing program
# 8.5% in Management, 8% in social services, and ~8%  in both veterinary nursing and Journalism and communication.
# The rst of the course have 6% - .2% each. These results suggest a good distribution across fields of study.

################################################################################
#Feature Development
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
  course_difficulty <- data %>%
    group_by(course) %>%
    summarize(
      avg_grade_1st_sem = mean(curricular_units_1st_sem_grade, na.rm = TRUE),
      std_grade_1st_sem = sd(curricular_units_1st_sem_grade, na.rm = TRUE),
      avg_grade_2nd_sem = mean(curricular_units_2nd_sem_grade, na.rm = TRUE),
      std_grade_2nd_sem = sd(curricular_units_2nd_sem_grade, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      overall_avg_grade = (avg_grade_1st_sem + avg_grade_2nd_sem) / 2,
      overall_std_grade = (std_grade_1st_sem + std_grade_2nd_sem) / 2
    )
  data <- data %>%
    left_join(course_difficulty, by = "course") # The course difficulty feature quantifies how challenging a course is based on student grades. 
  
  # 6. Hard Courses
  data <- data %>%
    mutate(
      hard_courses = ifelse(curricular_units_1st_sem_evaluations > 10, 1, 0)
    ) # Hard courses assume that many evaluations/tests make a course harder.
  
  # 7. Break Between Semesters
  data <- data %>%
    mutate(
      break_between_semesters = ifelse(curricular_units_1st_sem_enrolled > 0 & 
                                         curricular_units_2nd_sem_enrolled == 0, 1, 0)
    ) # Students enrolled in units in the first semester but not the next might be taking a break or dropping out.
  
  # 8. Discouraging Courses
  course_grade_summary <- data %>%
    group_by(course) %>%
    summarize(
      avg_course_grade = mean(curricular_units_1st_sem_grade, na.rm = TRUE),
      .groups = "drop"
    )
  data <- data %>%
    left_join(course_grade_summary, by = "course") %>%
    mutate(
      discouraging_courses = ifelse(avg_course_grade < 10, 1, 0)
    ) # Courses with low average grades (below the passing threshold of 10) are flagged as discouraging.
  
  # 9. Historical Success
  data <- data %>%
    mutate(
      normalized_previous_grade = previous_qualification_grade / 200,
      strong_historical_success = ifelse(normalized_previous_grade >= 0.663, 1, 0),
      weak_historical_success = ifelse(normalized_previous_grade < 0.5966, 1, 0)
    ) # Uses normalized thresholds to categorize students with strong or weak prior success.
  
  # 10. Grade-Based Summaries
  data <- data %>%
    mutate(
      average_grade = (curricular_units_1st_sem_grade + curricular_units_2nd_sem_grade) / 2,
      semester_grade_gap = curricular_units_2nd_sem_grade - curricular_units_1st_sem_grade
    ) # Summarizes overall grade performance and tracks grade changes between semesters.
  
  # 11. Prior Education Group Distributions
  prior_education_summary <- data %>%
    group_by(previous_qualification) %>%
    summarize(
      prior_avg_grade_1st_sem = mean(curricular_units_1st_sem_grade, na.rm = TRUE),
      prior_std_grade_1st_sem = sd(curricular_units_1st_sem_grade, na.rm = TRUE),
      prior_avg_grade_2nd_sem = mean(curricular_units_2nd_sem_grade, na.rm = TRUE),
      prior_std_grade_2nd_sem = sd(curricular_units_2nd_sem_grade, na.rm = TRUE),
      .groups = "drop"
    )
  data <- data %>%
    left_join(prior_education_summary, by = "previous_qualification") 
  # Tracks grade patterns for students by prior education level.
  
  return(data)
}


### Separately apply the combined function to the training data and test data
# Apply to training data
traindata <- combined_feature_engineering(traindata, is_training = TRUE)

# Apply to test data
testdata <- combined_feature_engineering(testdata, is_training = FALSE)


################################################################################
# Validate Feature Development
################################################################################

ncol(traindata)
head(traindata)





################################################################################
# Explain Feature Logic

#5. Course Difficulty: High evaluations might indicate a rigorous or time-consuming course, potentially contributing to student performance outcomes (e.g., dropping out or succeeding).












# Conclusion



## Limitations
 # This data doesn't include year over year data resulting in limited conclusion ability based
  # on annual distribution of enrolled, graduate, and drop out.
  # don't know the university? maybe we do? If we don't then we don't know what fianncial support looks like (maybe future direction)
  # Don't know for certain without further digging which courses belong to which degree program. Making this connection would contribute to knowing the unique suport needs of each degree path.

## Future Directions
  # There were many variables that were not further explored because their predictability in their current form wasn't strong enough to warent immediate feature development.
  # For instance, all the variables that feel along 0 in the plot below were largely left alone, leading to opportunity to develop new feature not yet explored.
  # Specially, creating demographic features that describe parent occupation and previsou education acheivement (qualification) could increase or decrease liklihood of success in UNiversity.
regression_plot











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