from step1 import Preprocessor
from step2 import Visualizer
from step3 import ModelTrainer
from step4 import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np

# Preprocessing
preprocessor = Preprocessor("/Users/shivampratapwar/Desktop/Customer_Churn_Prediction.csv")
preprocessor.get_basic_info()
preprocessor.get_unique_values(["state", "international_plan", "voice_mail_plan", "churn"])
preprocessor.dummify_columns(["state", "area_code", "international_plan", "voice_mail_plan", "churn"])
data = preprocessor.get_data()

# Visualization
visualizer = Visualizer(data)
visualizer.scatter_plot("account_length", "total_day_minutes", hue="churn_yes")
visualizer.hist_plot("total_day_minutes")

cat_vars = ["state", "international_plan", "voice_mail_plan", "churn"]
for feature in cat_vars:
    sns.set(style='whitegrid')
    plt.figure(figsize=(20, 5))
    total = len(data)
    ax = sns.countplot(x=feature, data=data)
    plt.show()

num_vars = ["account_length", "total_day_minutes", "total_day_calls", "total_day_charge", "total_eve_minutes",
            "total_eve_calls", "total_eve_charge", "total_night_minutes", "total_night_calls", "total_night_charge",
            "total_intl_minutes", "total_intl_calls", "total_intl_charge", "number_customer_service_calls"]
for feature in num_vars:
    sns.distplot(data[feature])
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()

for feature in num_vars:
    if feature != 'churn_yes':
        sns.boxplot(x='churn_yes', y=feature, data=data)
        plt.xlabel('Churn')
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()

sns.countplot(x='churn_yes', data=data)
plt.show()

# Model Training
target_col = "churn_yes"
drop_cols = ["churn_no"]
X = data.drop(columns=drop_cols + [target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
svc = SVC(kernel='rbf', decision_function_shape='ovr', probability=True)
rfc = RandomForestClassifier()
clf = XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.7, subsample=0.8, nthread=10, learning_rate=0.01)
logreg = LogisticRegression()

# Train models
svc.fit(X_train, y_train)
rfc.fit(X_train, y_train)
clf.fit(X_train, y_train)
logreg.fit(X_train, y_train)

# Predictions
svc_train_pred = svc.predict(X_train)
svc_test_pred = svc.predict(X_test)
rfc_train_pred = rfc.predict(X_train)
rfc_test_pred = rfc.predict(X_test)
clf_train_pred = clf.predict(X_train)
clf_test_pred = clf.predict(X_test)
logreg_train_pred = logreg.predict(X_train)
logreg_test_pred = logreg.predict(X_test)

# Probabilities for ROC AUC
svc_test_prob = svc.predict_proba(X_test)[:, 1]
rfc_test_prob = rfc.predict_proba(X_test)[:, 1]
clf_test_prob = clf.predict_proba(X_test)[:, 1]
logreg_test_prob = logreg.predict_proba(X_test)[:, 1]

# Evaluation
evaluator = Evaluator()
evaluator.evaluate_model_with_auc("SVC", y_train, svc_train_pred, y_test, svc_test_pred, svc_test_prob)
evaluator.evaluate_model_with_auc("Random Forest", y_train, rfc_train_pred, y_test, rfc_test_pred, rfc_test_prob)
evaluator.evaluate_model_with_auc("XGBoost", y_train, clf_train_pred, y_test, clf_test_pred, clf_test_prob)
evaluator.evaluate_model_with_auc("Logistic Regression", y_train, logreg_train_pred, y_test, logreg_test_pred, logreg_test_prob)

# Plot ROC Curves
evaluator.plot_roc_curves()

# Print scores
print(evaluator.get_scores())
