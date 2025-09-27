import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.svm as SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA


df=pd.read_excel('/home/anime/Desktop/visionArtificial/extract/caractNums.xlsx',header=None,engine='openpyxl')
y=df.iloc[:,0].values
X=df.iloc[:,1:].values

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

PCA_model = PCA(n_components=0.95)  # Retain 95% of variance
x_s_pca = PCA_model.fit_transform(X_scaled)

print("tamano", x_s_pca.shape)
explanation_variance = PCA_model.explained_variance_ratio_
print("Explained variance ratio by each component:", explanation_variance)
print("Total explained variance:", np.sum(explanation_variance))

## draw variance graph
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explanation_variance), marker='o')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.grid()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(x_s_pca, y, test_size=0.1, random_state=42)
model_svm = SVC.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))


##save model
import joblib
joblib.dump(model_svm, '/home/anime/Desktop/visionArtificial/extract/model_svm.pkl')
joblib.dump(scaler, '/home/anime/Desktop/visionArtificial/extract/scaler.pkl')
joblib.dump(PCA_model, '/home/anime/Desktop/visionArtificial/extract/PCA_model.pkl')        