# Titanic Machine Learning from Disaster

This repository contains my methodology for predicting passenger survival based on historical Titanic data.

## Methodology: Random Forest Architecture

The final pipeline utilizes a Random Forest classifier. This configuration achieved an accuracy score of 0.78708.

### 1. Data Engineering
Before feeding the data to the algorithm, I performed mathematical transformations:
* Age Imputation: Missing ages were filled using the median age of specific title groups extracted from passenger names.
* Missing Values: Missing embarked ports were filled with the most common port, and missing fares were filled with the median fare.
* Categorical Encoding: Text columns were converted into mathematical integers.

### 2. The Model
I trained exactly 1 Random Forest algorithm to capture non linear patterns in the passenger data without severe overfitting. The model utilizes an ensemble of 100 independent decision trees voting on the final outcome.

## How to Run the Code
Follow these terminal commands to execute the pipeline on your local machine:

```bash
pip install -r requirements.txt
cd src
python3 clean_data.py
python3 train_models.py
```

The final prediction file will be generated and saved in the submissions folder.
