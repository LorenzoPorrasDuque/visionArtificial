import cv2
##pip install scikit-learn pandas openpyxl xlsxwriter
import numpy as np
from glob import glob
import pandas as pd
import xlsxwriter   
from sklearn.neural_network import MLPClassifier    
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler

df=pd.read_excel('caractNums.xlsx',header=None,engine='openpyxl')
y=df.iloc[:,0].values
X=df.iloc[:,1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_mlp = MLPClassifier(hidden_layer_sizes=(10,50,20), max_iter=500, random_state=42)
model_mlp.fit(X_train, y_train)
y_pred = model_mlp.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

def test_multiple_configs(X, y):
    configs = [
        {'hidden_layer_sizes': (50,50), 'activation': 'relu', 'solver': 'adam'},
        {'hidden_layer_sizes': (100,), 'activation': 'tanh', 'solver': 'adam'},
        {'hidden_layer_sizes': (50,100,50), 'activation': 'relu', 'solver': 'sgd'},
        {'hidden_layer_sizes': (100,100), 'activation': 'logistic', 'solver': 'lbfgs'},
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    for i, cfg in enumerate(configs):
        print(f"\nConfig {i+1}: {cfg}")
        model = MLPClassifier(
            hidden_layer_sizes=cfg['hidden_layer_sizes'],
            activation=cfg['activation'],
            solver=cfg['solver'],
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

test_multiple_configs(X, y)
