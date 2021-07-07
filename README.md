# Loan Approval Prediction
## Objective
Design a supervised ML model that can predict whether a loan will be approved or not for a given applicant.

_But Why?_
Models like this, can be used to automate the loan eligibility process for the banking industry. Pre-screening applicants based on their online application details.

## Hypotheses
**Applicants are more likely to be approved for a loan if they have:**
1. A strong credit history 
2. A high unified applicant and co-applicant income
3. A high education level
4. A property in an urban area with high growth perspectives

## Approach
1. Hypothesis Generation – brainstorm factors that can impact an applicants probability of loan approval.
2. Exploratory Data Analysis – examine data and make inferences about distributions.
3. Data Cleaning – imputing missing values in the data and checking for outliers.
4. Feature Engineering – modifying existing variables and creating new ones for analysis.
5. Model Building – make predictive models capable of predicting whether an applicant will be approved for a loan (to a high degree of accuracy)

## TODO
6. Deploy model as REST-API (Flask) on AWS
7. Complete Presentation (gaps)

## Presentation
[Google Slides](https://docs.google.com/presentation/d/1pDMQFmwXwtYOgxeC4BEFW06G6skYX9iLMknwgyfUl9c/edit?usp=sharing)

## Data
The dataset used can be found [here](https://drive.google.com/file/d/1h_jl9xqqqHflI5PsuiQd_soNYxzFfjKw/view?usp=sharing)

Information collected, per loan applicant include:
|Variable| Description|
|------------- |-------------|
|Loan_ID| Unique Loan ID|
|Gender| Male/ Female|
|Married| Applicant married (Y/N)|
|Dependents| Number of dependents|
|Education| Applicant Education (Graduate/ Under Graduate)|
|Self_Employed| Self employed (Y/N)|
|ApplicantIncome| Applicant income|
|CoapplicantIncome| Coapplicant income|
|LoanAmount| Loan amount in thousands|
|Loan_Amount_Term| Term of loan in months|
|Credit_History| credit history meets guidelines|
|Property_Area| Urban/ Semi Urban/ Rural|
|Loan_Status| Loan approved (Y/N) |
