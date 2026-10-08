import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("Loading raw data from data/ folder...")
train = pd.read_csv('../data/raw/train.csv')
test = pd.read_csv('../data/raw/test.csv')
full_data = [train, test]

for dataset in full_data:
    # 1. Extract Title from Name
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Standardize rare titles
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # 2. Predictive Imputation: Fill missing Age using the Median of each Title group
    dataset['Age'] = dataset['Age'].fillna(dataset.groupby('Title')['Age'].transform('median'))
    
    # 3. Mode Imputation: Fill missing Embarked port with the most common port
    freq_port = train['Embarked'].dropna().mode()[0]
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
    # 4. Median Imputation: Fill the 1 missing Fare in test set with median fare
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    
    # 5. Feature Engineering: Create a Has_Cabin column instead of dropping Cabin completely
    dataset['Has_Cabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    
    # 6. Encoding: Convert text to numbers
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# 7. Drop columns that algorithms cannot process mathematically
drop_elements = ['Name', 'Ticket', 'Cabin', 'PassengerId']
train = train.drop(drop_elements, axis=1)

# Note: We must KEEP PassengerId in the test set because we need it for the final Kaggle submission format!
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

print("Saving cleaned data...")
train.to_csv('../data/processed/cleaned_train.csv', index=False)
test.to_csv('../data/processed/cleaned_test.csv', index=False)
print("Success! Cleaned files are ready.")
