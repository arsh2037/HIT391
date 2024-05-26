import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
path =  "kaggle_Interests_group/kaggle_Interests_group.csv"
dataset = pd.read_csv(path)
dataset.head()
data_filled = dataset.fillna(0)
# Normalize the interest columns (columns from 'interest1' to 'interest217')
interest_columns = data_filled.columns[2:]  # Excluding 'group' and 'grand_tot_interests'
scaler = StandardScaler()
data_filled[interest_columns] = scaler.fit_transform(data_filled[interest_columns])
# One-hot encode the 'group' column
encoder = OneHotEncoder(sparse_output=False, drop='first')
group_encoded = encoder.fit_transform(data_filled[['group']])
group_encoded_df = pd.DataFrame(group_encoded, columns=encoder.get_feature_names_out(['group']))
# Combine the encoded 'group' column with the rest of the data
X_encoded = pd.concat([group_encoded_df, data_filled.drop(columns=['group', 'grand_tot_interests'])], axis=1)
y = data_filled['group']
# Data splitting: 70% training, 15% evaluation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Check the shapes of the resulting datasets
print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Evaluation set: {X_eval.shape}, {y_eval.shape}')
print(f'Testing set: {X_test.shape}, {y_test.shape}')
# Check the preprocessed data
print(X_encoded.head())
# K-means Clustering on Training Set
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans_labels = kmeans.fit_predict(X_train)
X_train['KMeans_Cluster'] = kmeans_labels
# PCA for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train.drop(columns=['KMeans_Cluster']))
# Plot the K-means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_pca[:,0], y=X_train_pca[:,1], hue=kmeans_labels, palette='viridis')
plt.title('K-means Clustering on Training Set (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()
# Hierarchical Clustering on Training Set
linkage_matrix = linkage(X_train.drop(columns=['KMeans_Cluster']), method='ward')
# Plot the dendrogram
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram on Training Set')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
# Fit the Agglomerative Clustering model
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_train.drop(columns=['KMeans_Cluster']))
X_train['Hierarchical_Cluster'] = hierarchical_labels
# Plot the Hierarchical clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_pca[:,0], y=X_train_pca[:,1], hue=hierarchical_labels, palette='viridis')
plt.title('Hierarchical Clustering on Training Set (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()
# Initialize and train the LogisticRegression model
lr_model = LogisticRegression(max_iter=200, penalty='l2',C=1.0,random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate the model on the evaluation set
y_eval_pred = lr_model.predict(X_eval)
eval_accuracy = accuracy_score(y_eval, y_eval_pred)
eval_report = classification_report(y_eval, y_eval_pred, zero_division=1)

print(f"Evaluation Set Accuracy: {eval_accuracy}")
print("Evaluation Set Classification Report:")
print(eval_report)
"""
Result: 
Evaluation Set Accuracy: 0.9989484752891693
Evaluation Set Classification Report:
              precision    recall  f1-score   support

           C       1.00      1.00      1.00       277
           I       1.00      1.00      1.00       266
           P       1.00      1.00      1.00       246
           R       1.00      1.00      1.00       162

    accuracy                           1.00       951
   macro avg       1.00      1.00      1.00       951
weighted avg       1.00      1.00      1.00       951

"""

# Perform cross-validation
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {cv_scores.mean()}")
# results:
# Cross-Validation Scores: [1.         0.99887387 0.9954955  0.9988726  1.        ]
# Mean Cross-Validation Score: 0.9986483947306948
print(f"Logistic Regression Evaluation Set Accuracy: {eval_accuracy}")
print("Logistic Regression Evaluation Set Classification Report:")
print(eval_report)

"""Result: 

ogistic Regression Evaluation Set Accuracy: 0.9989484752891693
Logistic Regression Evaluation Set Classification Report:
              precision    recall  f1-score   support

           C       1.00      1.00      1.00       277
           I       1.00      1.00      1.00       266
           P       1.00      1.00      1.00       246
           R       1.00      1.00      1.00       162

    accuracy                           1.00       951
   macro avg       1.00      1.00      1.00       951
weighted avg       1.00      1.00      1.00       951"""


# now Random forest

# Model Implementation
# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=10,max_depth=50,random_state=0)
model.fit(X_train, y_train)
#model.fit(X_train., y_train)
# Evaluate the model on the evaluation set
y_eval_pred = model.predict(X_eval)
eval_accuracy = accuracy_score(y_eval, y_eval_pred)
eval_report = classification_report(y_eval, y_eval_pred, zero_division=1)
print(f"Evaluation Set Accuracy: {eval_accuracy}")
print("Evaluation Set Classification Report:")
print(eval_report)

"""
Result:
Evaluation Set Accuracy: 0.9852786540483701
Evaluation Set Classification Report:
              precision    recall  f1-score   support

           C       0.97      0.98      0.97       277
           I       0.98      0.98      0.98       266
           P       1.00      1.00      1.00       246
           R       1.00      0.98      0.99       162

    accuracy                           0.99       951
   macro avg       0.99      0.99      0.99       951
weighted avg       0.99      0.99      0.99       951
"""
# Test the model on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred, zero_division=1)

print(f"Test Set Accuracy: {test_accuracy}")
print("Test Set Classification Report:")
print(test_report)
"""Result:

Test Set Accuracy: 0.9800210304942166
Test Set Classification Report:
              precision    recall  f1-score   support

           C       0.98      0.95      0.96       273
           I       0.97      0.99      0.98       256
           P       0.98      1.00      0.99       248
           R       0.98      0.98      0.98       174

    accuracy                           0.98       951
   macro avg       0.98      0.98      0.98       951
weighted avg       0.98      0.98      0.98       951

"""