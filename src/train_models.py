import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("Loading cleaned data...")
train = pd.read_csv('../data/processed/cleaned_train.csv')
test = pd.read_csv('../data/processed/cleaned_test.csv')

# Prepare training data
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

# Prepare testing data
X_test = test.drop('PassengerId', axis=1)
passenger_ids = test['PassengerId']

print("Training Logistic Regression...")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
logreg_preds = logreg.predict(X_test)

print("Training Decision Tree...")
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
decision_tree.fit(X_train, y_train)
tree_preds = decision_tree.predict(X_test)

print("Training Random Forest (Ensemble of Trees)...")
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
random_forest.fit(X_train, y_train)
rf_preds = random_forest.predict(X_test)

# Save submissions
def save_submission(predictions, filename):
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Saved {filename}")

save_submission(logreg_preds, '../submissions/submission_logistic.csv')
save_submission(tree_preds, '../submissions/submission_tree.csv')
save_submission(rf_preds, '../submissions/submission_random_forest.csv')

print("All models trained and submission files generated!")
