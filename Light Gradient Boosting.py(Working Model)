#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Load dataset
df = pd.read_csv("C:/Users/Charvitha Reddy/Downloads/HEALTH_DATASET.csv")

# Preprocess categorical columns
df.sex.replace(['female', 'male'], [0, 1], inplace=True)
df.claim_outcome.replace(['Approval', 'Rejected'], [1, 0], inplace=True)

# Encode other categorical features
le = LabelEncoder()
df.city = le.fit_transform(df.city)
df.job_title = le.fit_transform(df.job_title)
df.hereditary_diseases = le.fit_transform(df.hereditary_diseases)

# Plot class distribution
temp = df['claim_outcome'].value_counts()
plt.pie(temp.values, labels=temp.index.values, autopct='%1.1f%%')
plt.title("Class Distribution")
plt.show()

# Plot correlation heatmap
sb.heatmap(df.corr() > 0.7, cbar=False, annot=True)
plt.show()

# Plot count plot
sb.countplot(data=df, x='smoker', hue='claim_outcome')
plt.show()

# Plot distributions of first 9 features
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns[:9]):
    if col in ['age', 'claim_outcome']:
        continue
    plt.subplot(3, 3, i + 1)
    sb.histplot(df[col], kde=True)  # Use histplot instead of distplot
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Split data
features = df.drop('claim_outcome', axis=1)
target = df['claim_outcome']
X_train, X_val, Y_train, Y_val = train_test_split(features, target, random_state=2023, test_size=0.20)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=Y_train)
test_data = lgb.Dataset(X_val, label=Y_val, reference=train_data)

# Define number of boosting rounds
num_round = 100

# Train LightGBM model
model = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# Predict and evaluate
y_train_pred_prob = model.predict(X_train)
y_val_pred_prob = model.predict(X_val)

# Convert probabilities to class labels
y_train_pred_labels = (y_train_pred_prob > 0.5).astype(int)
y_val_pred_labels = (y_val_pred_prob > 0.5).astype(int)

# Calculate metrics
train_accuracy = accuracy_score(Y_train, y_train_pred_labels)
val_accuracy = accuracy_score(Y_val, y_val_pred_labels)
train_roc_auc = roc_auc_score(Y_train, y_train_pred_prob)
val_roc_auc = roc_auc_score(Y_val, y_val_pred_prob)

# Calculate overall accuracy (using the entire dataset)
# This is useful if you want to evaluate the model's performance on the whole dataset
df_pred_prob = model.predict(scaler.transform(features))
df_pred_labels = (df_pred_prob > 0.5).astype(int)
overall_accuracy = accuracy_score(target, df_pred_labels)

# Print metrics
print("Training Accuracy: ", train_accuracy)
print("Validation Accuracy: ", val_accuracy)
print("Training ROC-AUC: ", train_roc_auc)
print("Validation ROC-AUC: ", val_roc_auc)
print("Overall Accuracy: ", overall_accuracy)

