# 🎓 Predicting Educational Outcomes

Welcome to the Predicting Educational Outcomes Project, an analysis leveraging machine learning to classify and predict student outcomes: dropout, enrollment, or graduation. This repository includes an R Markdown report (predicting_ed_outcomes.Rmd), a PDF version knit from the R Markdown report (predicting_ed_outcomes.pdf), and a companion R script for analysis.

## 📚 Overview

This project employs machine learning techniques to predict student success using features such as academic performance, demographics, and socioeconomic factors. By analyzing these predictors, the goal is to inform educators and administrators about students at risk of dropping out, enabling targeted interventions.

### Highlights:

- Implemented Gradient Boosted Trees (GBT) with Cross-Validation and Decision Tree Classifiers to model outcomes.
- Feature engineering transformed raw data into insightful predictors like grade categories, financial strain indicators, and course difficulty measures.
- Evaluation metrics include sensitivity, specificity, precision, and balanced accuracy.

## 📂 Repository Structure
- **predicting_ed_outcomes/**
  - **reports/**: Directory for report outputs
  
    - `predicting_ed_outcomes.Rmd`: R Markdown file for the detailed report
    - `predicting_ed_outcomes.pdf`: PDF version knit from R Markdown report
  - **scripts/**: Contains the R script
    - `predicting_ed_outcomes.R`: R Script file for the analysis
  - **data/**: Contains the data for the report and script
    - `data.csv`: the numerically encoded dataset used for analysis
    - `variable_table.csv`: the definitions for the numerically encoded dataset


## 📊 Dataset Information

The dataset originates from Martins and colleagues (2021) and represents student records from a single Portuguese university. It includes:

- **4,424 student records** across various undergraduate programs such as agronomy, education, and technologies.
- **Key fields**: Academic grades, enrollment status, demographics, and socioeconomic indicators.

Since the dataset is part of the repository, it will automatically load and preprocess when running the analysis using the following steps.


## 🚀 How to Run

1. **Clone the Repository**

To ensure the analysis runs correctly, download the entire repository below as a zip file:

- [GitHub Repository URL:](https://github.com/KevinWMcGowan/predicting_ed_outcomes.git)

2. **Unzip the Zip File**

The script automatically checks for the data folder within the unzipped repository. It ensures the presence of data.csv and variable_table.csv and loads them into the environment.
- Click into the Zipfile and open either the R Script or Rmd Report in R Studio

3. Run Analysis

- R Script: Run predicting_ed_outcomes.R to execute all preprocessing, feature engineering, and modeling steps.
- R Markdown: Open predicting_ed_outcomes.Rmd and click “Knit” to generate a comprehensive report in PDF format.
- Reports: Access pre-generated results in predicting_ed_outcomes.pdf.

**Additional Notes**
If you're unable to knit the Rmd file to PDF you're likely missing the appropriate software. Alternatively, use the carret key by the knit button to "knit to HTML."

## 🧠 Methodology

**Data Preprocessing**
The dataset was cleaned and transformed for analysis. Key feature engineering steps include:

- Categorizing grades into performance levels.
- Identifying financial strain using variables like tuition payments and debt status.
- Flagging students under-enrolled or at risk based on course load.

**Modeling**
The project uses two primary machine learning models:

- Decision Tree Classifiers for initial predictive modeling.
- Gradient Boosted Trees (GBT) with cross-validation for robust predictions.

**Evaluation Metrics**
Model performance was assessed using:

- Sensitivity: Ability to correctly identify positive cases.
- Specificity: Ability to correctly identify negative cases.
- Precision: Positive Predictive Value.
- Balanced Accuracy: Average of sensitivity and specificity.

## 📋 Report Objectives

This report outlines the step-by-step methodology used to build and validate the models. The goal is to highlight actionable insights for improving student outcomes, focusing on:

- Identifying at-risk students.
- Enhancing predictive accuracy with advanced modeling techniques.
- Informing decision-making for educators and administrators.

## 🤝 Connect

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/kevin-w-mcgowan-m-s-iop/) for collaboration opportunities
