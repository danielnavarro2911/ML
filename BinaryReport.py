import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BinaryReport:
    def __init__(self, ml_models):
        """
        Initializes the class with a list of ML model objects.
        
        Parameters:
        ml_models (list): List of ML class objects.
        """
        self.ml_models = ml_models

    def plot(self):
        """
        Plots the ROC curve for each model in the list of ML models.
        """
        # Get the test data from any of the models
        X_test = self.ml_models[0].X_test
        y_test = self.ml_models[0].y_test

        plt.figure(figsize=(10, 8))

        for ml_model in self.ml_models:
            # Get the predicted probabilities from the model
            y_prob = ml_model.model.predict_proba(X_test)[:, 1]

            # Calculate the ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Plot the ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{ml_model.model.__class__.__name__} (AUC = {roc_auc:.2f})')

        # Plot the diagonal line (y = x)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        # Configure the plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves of Models')
        plt.legend(loc="lower right")

        plt.show()

    def get_metrics_dataframe(self):
        """
        Generates a DataFrame containing classification metrics for each model.

        Returns:
        pd.DataFrame: DataFrame with metrics like Accuracy, Precision, Recall, and F1 score per model.
        """
        # Get the test data from any of the models
        X_test = self.ml_models[0].X_test
        y_test = self.ml_models[0].y_test
        
        metrics = []

        for ml_model in self.ml_models:
            y_pred = ml_model.predictions  # Predictions from the model
            model_name = ml_model.model.__class__.__name__

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average=None)  # Recall per class
            precision = precision_score(y_test, y_pred, average=None)  # Precision per class
            f1 = f1_score(y_test, y_pred, average=None)  # F1 score per class

            model_metrics = {
                'Model': model_name,
                'Accuracy': accuracy,
                'Recall (class 0)': recall[0] if len(recall) > 0 else None,
                'Recall (class 1)': recall[1] if len(recall) > 1 else None,
                'Precision (class 0)': precision[0] if len(precision) > 0 else None,
                'Precision (class 1)': precision[1] if len(precision) > 1 else None,
                'F1 Score (class 0)': f1[0] if len(f1) > 0 else None,
                'F1 Score (class 1)': f1[1] if len(f1) > 1 else None,
            }

            metrics.append(model_metrics)

        return pd.DataFrame(metrics)