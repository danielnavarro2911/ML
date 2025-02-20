from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class ML:
    """
    A machine learning pipeline for classification and regression tasks.
    Automates data preprocessing, model training, evaluation, and feature importance analysis.
    """
    
    def __init__(self, df, target, model, dummies=True, test_size=0.3, scalate=True, grid=False, regression=False):
        """
        Initializes the ML class.
        
        Parameters:
        df (pd.DataFrame): Dataset.
        target (str): Target variable name.
        model (estimator): Machine learning model.
        dummies (bool): Whether to convert categorical variables to dummy variables (default: True).
        test_size (float): Test dataset proportion (default: 0.3).
        scalate (bool): Whether to standardize features (default: True).
        grid (dict or bool): Grid search parameter grid (default: False).
        regression (bool): Defines if task is regression (default: False).
        """
        self.df = df
        self.target = target
        self.test_size = test_size
        self.X = df.drop(target, axis=1)
        self.y = df[target]
        self.model = model
        self.scalate = scalate
        self.regression = regression
        self.grid = grid
        if dummies:
            self.X = pd.get_dummies(self.X)
        self.X_col = self.X.columns
    
    def __split(self):
        """Splits data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)
    
    def __scalate(self):
        """Standardizes numerical features if enabled."""
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def __fit(self):
        """Fits the model, applying GridSearchCV if specified."""
        if self.grid:
            self.grid_search = GridSearchCV(self.model, param_grid=self.grid, refit=True, verbose=0)
            self.grid_search.fit(self.X_train, self.y_train)
            self.best_grid_options = self.grid_search.best_params_
            self.model = self.grid_search.best_estimator_
        else:
            self.model.fit(self.X_train, self.y_train)
    
    def predict(self):
        """
        Runs the full pipeline: splits data, scales features, trains the model, and makes predictions.
        
        Returns:
        np.array: Model predictions.
        """
        self.__split()
        self.__scalate()
        self.__fit()
        self.predictions = self.model.predict(self.X_test)
        return self.predictions
    
    def __regression_metrcis(self, show_results):
        """Computes regression performance metrics and plots predictions vs actual values."""
        plt.scatter(self.y_test, self.predictions)
        plt.plot(self.y_test, self.y_test, 'r--')
        plt.xlabel('y_test')
        plt.ylabel('predictions')
        plt.title('predictions vs y_test')
        plt.show()
        
        self.r2 = r2_score(self.y_test, self.predictions)
        self.mae = mean_absolute_error(self.y_test, self.predictions)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, self.predictions))
        
        if show_results:
            print(f'\nr2: {self.r2}\nMean Absolute Error: {self.mae}\nRoot Mean Squared Error: {self.rmse}\n')
        
        return {'r2': self.r2, 'MAE': self.mae, 'RMSE': self.rmse}
    
    def __classification_metrics(self, show_results):
        """Computes classification metrics, including confusion matrix and classification report."""
        cm = confusion_matrix(self.y_test, self.predictions)
        labels = sorted(set(self.y_test))
        self.confusion_matrix = pd.DataFrame(cm, index=[f'Actual {label}' for label in labels], 
                                             columns=[f'Predicted {label}' for label in labels])
        self.classification_report = classification_report(self.y_test, self.predictions, output_dict=True)
        
        if show_results:
            print(f'\nConfusion Matrix:\n{self.confusion_matrix}\n')
            print(f'Classification Report:\n{classification_report(self.y_test, self.predictions)}\n')
        
        return {"CM": self.confusion_matrix, 'CR': self.classification_report}
    
    def calculate_metrics(self, show_results=True):
        """
        Computes and returns model performance metrics.
        
        Parameters:
        show_results (bool): Whether to print the metrics (default: True).
        
        Returns:
        dict: Regression or classification metrics.
        """
        if self.regression:
            return self.__regression_metrcis(show_results)
        else:
            return self.__classification_metrics(show_results)
    
    def variables_importance(self, plot=True, scoring='',top=None):
        """
        Computes and optionally plots feature importance using permutation importance.
        
        Parameters:
        plot (bool): Whether to display a feature importance plot (default: True).
        scoring (str): Scoring metric for importance computation (default: 'r2' for regression, 'accuracy' for classification).
        
        Returns:
        pd.DataFrame: Feature importance scores.

        Top:
        a number to plot the top variables
        """
        if not scoring:
            scoring = 'r2' if self.regression else 'accuracy'
        
        results = permutation_importance(self.model, self.X_test, self.y_test, scoring=scoring).importances_mean
        df_results = pd.DataFrame(data=[self.X_col, results * 100], index=['Column', 'Importance']).T
        df_results = df_results.sort_values('Importance', ascending=False)
        
        if plot:
            if top==None:
                top=len(df_results)
            plt.figure(figsize=(20, 8))
            sns.barplot(data=df_results.head(top), x='Column', y='Importance')
            plt.xticks(rotation=90)
            plt.show()
        
        return df_results
