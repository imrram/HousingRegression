import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.savefig("outputs/correlation_matrix.png")

def plot_actual_vs_pred(y_true, y_pred, model_name):
    plt.figure()
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.savefig(f"outputs/{model_name.lower()}_actual_vs_pred.png")

def plot_model_comparisons(metrics_dict, metric_name="R²"):
    models = list(metrics_dict.keys())
    scores = list(metrics_dict.values())
    plt.figure()
    plt.bar(models, scores, color='green' if metric_name == "R²" else 'orange')
    plt.title(f"Model {metric_name} Comparison")
    plt.ylabel(metric_name)
    plt.savefig(f"outputs/model_{metric_name.lower()}_comparison.png")