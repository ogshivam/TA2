import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self, data, target_col, drop_cols=[]):
        self.y = data[target_col]
        self.X = data.drop(columns=drop_cols + [target_col])
        self.models = {
            'SVC': SVC(kernel='rbf', decision_function_shape='ovr'),
            'RandomForest': RandomForestClassifier(),
            'XGB': XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.7, subsample=0.8, nthread=10, learning_rate=0.01),
            'LogisticRegression': LogisticRegression()
        }
    
    def split_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
    
    def train_model(self, model_name):
        self.models[model_name].fit(self.X_train, self.y_train)
    
    def predict(self, model_name, X):
        return self.models[model_name].predict(X)
    
    def evaluate_model(self, model_name):
        predictions = self.predict(model_name, self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        return accuracy, report
    
    def get_train_test_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
