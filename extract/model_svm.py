import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.svm as SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

df=pd.read_excel('/home/anime/Desktop/visionArtificial/extract/caractNums.xlsx',header=None,engine='openpyxl')
y=df.iloc[:,0].values
X=df.iloc[:,1:].values

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model_svm = SVC.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
