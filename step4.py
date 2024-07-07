import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

class Evaluator:
    def __init__(self):
        self.scores = pd.DataFrame(columns=['Model', 'MAE_Train', 'MSE_Train', 'RMSE_Train', 'MAPE_Train',
                                            'MAE_Test', 'MSE_Test', 'RMSE_Test', 'MAPE_Test', 'Accuracy', 'Cohen_Kappa', 'AUC'])
        self.roc_curves = {}

    def evaluate_model(self, model_name, y_train, train_pred, y_test, test_pred):
        MAE_Train = metrics.mean_absolute_error(y_train, train_pred)
        MSE_Train = metrics.mean_squared_error(y_train, train_pred)
        RMSE_Train = np.sqrt(metrics.mean_squared_error(y_train, train_pred))
        MAPE_Train = metrics.mean_absolute_percentage_error(y_train, train_pred)
        MAE_Test = metrics.mean_absolute_error(y_test, test_pred)
        MSE_Test = metrics.mean_squared_error(y_test, test_pred)
        RMSE_Test = np.sqrt(metrics.mean_squared_error(y_test, test_pred))
        MAPE_Test = metrics.mean_absolute_percentage_error(y_test, test_pred)
        Accuracy = accuracy_score(y_test, test_pred)
        Cohen_Kappa = cohen_kappa_score(y_test, test_pred)
        
        self.scores = self.scores.append(pd.Series([model_name, MAE_Train, MSE_Train, RMSE_Train, MAPE_Train, 
                                                    MAE_Test, MSE_Test, RMSE_Test, MAPE_Test, Accuracy, Cohen_Kappa, None],
                                                   index=self.scores.columns), ignore_index=True)
        
        print(f"Accuracy for {model_name}: {Accuracy}")
        print(f"Classification report for {model_name}:\n{classification_report(y_test, test_pred)}")
        print(f"Confusion Matrix for {model_name}:\n{confusion_matrix(y_test, test_pred)}")
        print(f"Cohen Kappa Score for {model_name}: {Cohen_Kappa}")

    def evaluate_model_with_auc(self, model_name, y_train, train_pred, y_test, test_pred, y_pred_prob):
        self.evaluate_model(model_name, y_train, train_pred, y_test, test_pred)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        self.roc_curves[model_name] = (fpr, tpr, roc_auc)
        self.scores.loc[self.scores['Model'] == model_name, 'AUC'] = roc_auc
        
        print(f"Area under the ROC curve for {model_name}: {roc_auc}")

    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))
        for model_name, (fpr, tpr, roc_auc) in self.roc_curves.items():
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.show()

    def plot_residuals(self, y_train, y_train_pred):
        residuals = y_train - y_train_pred
        plt.scatter(y_train_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()
    
    def get_scores(self):
        return self.scores

