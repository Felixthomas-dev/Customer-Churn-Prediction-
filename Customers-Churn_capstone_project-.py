#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


# importing the libraries
import numpy as np
import pandas as pd
import sklearn as sk

# for Visualization
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

# Data preprocessing
from sklearn.preprocessing import LabelEncoder

# Machine learning Models
get_ipython().system('pip install xgboost')
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV

import os
os.environ["omp_NUM_THREADS"] = '1'
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px


# ### Load dataset

# In[2]:


# load dataset into notebook

dt = pd.read_csv("Customer-Churn.csv")


# ### EDA

# In[3]:


# display the number of rows and columns respectively

dt.shape


# In[4]:


# checking for the first ten records in the dataset table 

dt.head(10)


# In[5]:


# checking for the last 5 records in the dataset table 

dt.tail(10)


# In[6]:


dt.info()


# In[7]:


# statistical summary
dt.describe()


# ## Data Cleaning
# 
# ### Check for missing Values

# In[8]:


# checking if there are any missing number or values in the data


dt.isnull().sum()


# In[9]:


# Check missing values using msno

msno.bar(dt, color='grey')


# In[10]:


# Display columns with missing values

sns.heatmap(dt.isnull())


# In[11]:


# Create Tenure category for customers

def tenure_category(months):
    if months <= 12:
        return '0-12 months'
    elif months <= 24:
        return '13-24 months'
    elif months <= 36:
        return '25-36 months'
    else:
        return '>36 months'
    
dt['tenure_category'] = dt['tenure'].apply(tenure_category)
dt.head(2)


# In[12]:


# Identify numerical and non-numerical columns

numerical_cols = dt.select_dtypes(include=['int64', 'float64']).columns.tolist()
non_numerical_cols = dt.select_dtypes(exclude=['int64', 'float64']).columns.tolist()


# In[13]:


# Print the lists of numerical columns

print("Numerical columns:", numerical_cols)


# In[14]:


# Print the lists of non-numerical columns

print("Non-numerical columns:", non_numerical_cols)


# In[15]:


dt['tenure_category'].isnull().sum()


# In[16]:


# Impute missing values (replace with median) on TotalCharges column

dt.fillna(dt[numerical_cols].median(), inplace=True)


# In[17]:


# check for missing values after filling with median value
dt.isnull().sum()


# ### Check For Duplicates

# In[18]:


# checking for duplicates in the dataset

dt.duplicated().sum()


# In[19]:


# checking the number of unique variable in the non numerical columns
dt[non_numerical_cols].nunique()


# ### Handling Outlier

# In[20]:


# plotting boxplot to visualise outliers

# Univariate analysis (Numerical columns) - use boxplot to view all numerical columns with outliers. 

plt.figure(figsize=(18, 10))

numerical_cols_without_seniorctz = [col for col in numerical_cols if col != 'SeniorCitizen']

for i, column_name in enumerate(numerical_cols_without_seniorctz, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(dt[column_name])
    plt.xlabel(column_name)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# In[21]:


# Calculate IQR for each column
q1 = dt[numerical_cols].quantile(0.25)
q3 = dt[numerical_cols].quantile(0.75)

iqr = q3 - q1

# Identify outliers using IQR, excluding records where 'seniorcitizen' is 1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

# Exclude records where 'seniorcitizen' is 1
outliers = (((dt[numerical_cols] < lower_bound) | (dt[numerical_cols] > upper_bound)) & (dt['SeniorCitizen'] != 1)).any(axis=1)

# Display the number of outliers
print(outliers.sum())

# Display the outlier records
dt[outliers]


# In[22]:


# Remove outliers based on IQR
dt =  dt[~outliers]

# Display the dataset after handling outliers
dt


# ### Univariate Analysis

# In[23]:


# ploting a donut chart to visualize 

GenderType =dt['gender'].value_counts()
plt.pie(GenderType.values,labels=GenderType.index,autopct='%1.1f%%')
plt.title("Gender Rate", size=12, weight='bold')

circle =plt.Circle((0,0),0.5,color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.show


# In[24]:


plt.figure(figsize=(6, 5))

ax = sns.countplot(data=dt, x="PaymentMethod")

# Rotate the x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Annotate the bars with percentages
for p in ax.patches:
    ax.text(x=p.get_x() + p.get_width() / 2, 
            y=p.get_height()/7,  # Adjust the position of the text
            s=f"{np.round(p.get_height() / len(dt) * 100, 0)}%", 
            ha='center', size=15, weight='bold', rotation=0, color='Black')

ax.set_xlabel("Payment Method")
ax.set_ylabel("Counts")
plt.title("Count of Payment Method")

plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()


# In[25]:


# Univariate analysis (Target column i.e. Churn) - Countplot

plt.figure(figsize=(6,4))
ax=sns.countplot(data=dt, x="Churn")

for i in ax.patches:
    ax.text(x=i.get_x()+i.get_width()/2, y=i.get_height()/7, s=f"{np.round(i.get_height()/len(dt)*100,0)}%",
            ha='center', size=15, weight='bold', rotation=30, color='Black')
    
plt.title("Customer Churn", size=12, weight='bold')
plt.annotate(text=" Non-churn", xytext=(0.5,3500), xy=(0.2,1250), 
             arrowprops =dict(arrowstyle="->", color='black', connectionstyle="angle3,angleA=0,angleB=60"), color='black')
plt.annotate(text="Churn", xytext=(0.8,2500), xy=(1.2,1000), 
             arrowprops =dict(arrowstyle="->", color='black',  connectionstyle="angle3,angleA=0,angleB=90"), color='black')


# In[26]:


# Univariate analysis (Target column i.e. Churn) - Countplot

plt.figure(figsize=(6,4))
ax=sns.countplot(data=dt, x="Contract")

for i in ax.patches:
    ax.text(x=i.get_x()+i.get_width()/2, y=i.get_height()/7, s=f"{np.round(i.get_height()/len(dt)*100,0)}%",
            ha='center', size=15, weight='bold', rotation=30, color='Black')


# In[ ]:





# In[27]:


# Calculate the frequency of each 'tenure_category'
category_order = dt['tenure_category'].value_counts().index

# Create the count plot with the 'Churn' hue
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='tenure_category', data=dt, order=category_order)
plt.xlabel('Tenure Category in Months')
plt.ylabel('Frequency')
plt.title('Total Number of Customers by Tenure ')

# Add percentage labels
total_counts = len(dt)
for p in ax.patches:
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    value = np.round((y / total_counts) * 100, 2)
    ax.text(x, y, f"{value}%", ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[28]:


import numpy as np

# List of categorical columns to visualize
categorical_cols = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", 
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
                    "Contract", "PaperlessBilling"]

# Number of plots
num_plots = len(categorical_cols)

# Create subplots
fig, axes = plt.subplots(nrows=(num_plots // 2) + (num_plots % 2), ncols=2, figsize=(15, 5 * (
    (num_plots // 2) + (num_plots % 2))))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through the categorical columns and create count plots
for i, col in enumerate(categorical_cols):
    sns.countplot(data=dt, x=col, ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Counts")
    axes[i].set_title(f"Count of {col}")
    
    # Add percentage labels
    for patch in axes[i].patches:
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height() / 7
        value = np.round(patch.get_height() / len(dt) * 100, 0)
        axes[i].text(x=x, y=y, s=f"{value}%", ha='center', size=15, weight='bold', rotation=30, color='Black')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# 
# 
# 
# 
# ### Bivariate Analysis

# In[29]:


# Calculate the frequency of each 'tenure_category'
# Plotting the sorted countplot with 'Churn' as hue
category_order = dt['tenure_category'].value_counts().index

plt.figure(figsize=(10, 6))
ax = sns.countplot(x='tenure_category', data=dt, order=category_order, hue='Churn')
plt.xlabel('Tenure Category in Months')
plt.ylabel('Frequency')
plt.title('Total Number of Customers by Tenure and Churn')
plt.legend(title='Churn', loc='upper right')

# Add percentage labels
total_counts = len(dt)
for p in ax.patches:
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    value = np.round((y / total_counts) * 100, 2)
    ax.text(x, y, f"{value}%", ha='center', va='bottom')

plt.tight_layout()
plt.show()



# In[30]:


# List of categorical columns to visualize
categorical_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
                    "Contract", "PaperlessBilling", "PaymentMethod"]

# Number of plots
num_plots = len(categorical_cols)

# Create subplots
fig, axes = plt.subplots(nrows=(num_plots // 2) + (num_plots % 2), ncols=2, figsize=(15, 5 * (
    (num_plots // 2) + (num_plots % 2))))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through the categorical columns and create count plots
for i, col in enumerate(categorical_cols):
    sns.countplot(data=dt, x=col, hue="Churn", ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Counts")
    axes[i].set_title(f"Count of {col} with Churn")
    
    
     # Add percentage labels
    for p in axes[i].patches:
        axes[i].text(x=p.get_x() + p.get_width() / 2, 
                     y=p.get_height()/7,  # Adjust the position of the text
                     s=f"{np.round(p.get_height() / len(dt) * 100, 0)}%", 
                     ha='center', size=15, weight='bold', rotation=0, color='Black')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()



# In[31]:


# Bivariate analysis - Pairplot

# Create a copy of the numerical columns list
numerical_cols_copy = numerical_cols.copy()

# Remove 'SeniorCitizen' from the list of numerical columns if it exists
numerical_cols_copy = [col for col in numerical_cols_copy if col != 'SeniorCitizen']

# Bivariate analysis - Pairplot
sns.pairplot(dt[numerical_cols_copy + ["Churn"]], hue="Churn")
plt.suptitle("Pairplot of Features with Churn", y=1.02)
plt.show()


# ### Multivariate Analysis

# In[32]:


# Multivariate analysis - Correlation matrix
dt['Churn'] = dt['Churn'].map({'No': 0, 'Yes': 1})
correlation_matrix = dt[numerical_cols + ['Churn']].corr()
correlation_matrix


# In[33]:


# correllation heatmap showing levl of correlation in the features

# Create the correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[34]:


from sklearn.preprocessing import LabelEncoder

# Select categorical columns to be encoded
cat_cols = dt.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
label_encoders = {}

# Apply Label Encoding for each categorical column
for col in cat_cols:
    le = LabelEncoder()
    dt[col] = le.fit_transform(dt[col])
    label_encoders[col] = le

# Print the encoded DataFrame
# print(dt)

# If you want to revert the encoding later, you can use the inverse_transform method of LabelEncoder
# For example:
# dt['gender'] = label_encoders['gender'].inverse_transform(dt['gender'])



# In[35]:


# Feature selections
labels = dt[['Churn']]
features = dt.drop(['customerID','Churn'], axis=1)


# In[36]:


labels.head()


# In[37]:


features.head()


# In[38]:


print(labels.shape)
print(labels.squeeze().shape)


# In[39]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels.squeeze(), test_size=0.2, random_state=42)

# Standardize our training data

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['number']))

X_test_scaled = scaler.transform(X_test.select_dtypes(include=['number']))


# ### Machine Learning Modelling

# In[40]:


X_train


# In[41]:


X_train_scaled.shape


# In[42]:


X_test_scaled.shape


# In[43]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Initialize the model
log_reg_model = LogisticRegression(random_state=42)


# In[44]:


# Train the model

log_reg_model.fit(X_train_scaled, y_train)


# In[45]:


# Make predictions on the test set
y_pred = log_reg_model.predict(X_test_scaled)


# In[46]:


y_pred


# In[47]:


# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)


# In[48]:


# Display results
print("Accuracy: ", accuracy*100)
print("Classification Report: \n", report)
print("Confusion Matrix: \n", matrix)


# In[49]:


cm = confusion_matrix(y_test,y_pred)

ax = sns.heatmap(cm, cmap='flare',annot=True, fmt='d')

plt.xlabel("Predicted Class",fontsize=12) 
plt.ylabel("True Class",fontsize=12) 
plt.title("Confusion Matrix",fontsize=12)

plt.show()


# ### Leveraging Different Models to Obtain the Best result

# In[50]:


# Initialize and train the models
models = {
    "Logistic Regression" : LogisticRegression(random_state=42),
    "Decision Tree" : DecisionTreeClassifier(random_state=42),
    "Random Forest" : RandomForestClassifier(random_state=42),
    "AdaBoost" : AdaBoostClassifier(random_state=42),
    "XG Boost" : XGBClassifier(random_state=42),
    "SVC" : SVC(random_state=42)
         }


# In[51]:


for model_name, model in models.items():
    
    # Training and prediction
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Display results
    print(model_name)
    print("Accuracy: ", accuracy*100)
    print("Classification Report: \n", report)
    plt.figure(figsize=(5,3))
    
    #sns.heatmap(matrix, annot=True)
    sns.heatmap(matrix, cmap='flare',annot=True, fmt='d')
    plt.xlabel("Predicted Class",fontsize=10) 
    plt.ylabel("True Class",fontsize=10) 
    plt.title("Confusion Matrix",fontsize=10)
    plt.show()
    print("\n")


# ### Feature Selection 

# In[52]:


# Initialize and train the Random Forest model
radom_forest_model = RandomForestClassifier(random_state=42)
radom_forest_model.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances =  radom_forest_model.feature_importances_

# Create a DataFrame to store feature names and their importance scores
feature_importances_df = pd.DataFrame({'Features': X_train.columns, 'Importance': feature_importances})

# Sort features by importance in descending order
feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot the feature importance
sns.barplot(x='Importance', y='Features', data=feature_importances_df, palette='viridis')


# In[53]:


import pandas as pd

# Check the lengths of X_train.columns and feature_importances
print(f"Number of features in X_train: {len(X_train.columns)}")
print(f"Number of feature importances: {len(feature_importances)}")

# Ensure the lengths match
if len(X_train.columns) != len(feature_importances):
    raise ValueError("Mismatch in number of features and feature importances")

# Create a DataFrame to store feature names and their importance scores
feature_importances_df = pd.DataFrame({
    'Features': X_train.columns,
    'Importance': feature_importances
})

# Sort features by importance in descending order
feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

# Display the sorted feature importances
print(feature_importances_df)


# In[54]:


selected_features = ['TotalCharges','MonthlyCharges', 'tenure', 'tenure_category', 'Contract', 'PaymentMethod', 'OnlineSecurity',
                    'TechSupport', 'gender', 'PaperlessBilling', 'OnlineBackup', 'InternetService', 'DeviceProtection',
                     'Partner', 'MultipleLines']

  


# In[55]:


# Split the data into training and testing sets

X_train_, X_test_, y_train_, y_test_ = train_test_split(dt[selected_features], labels.squeeze(), test_size=0.2, 
                                                        random_state=42)

# Standardize our training data
scaler = StandardScaler()

X_train_scaled_ = scaler.fit_transform(X_train_.select_dtypes(include=['number']))

X_test_scaled_ = scaler.transform(X_test_.select_dtypes(include=['number']))


# In[56]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Train, Test, and Evaluate Model 
for model_name, model in models.items():
    
    # Training and prediction
    model.fit(X_train_scaled_, y_train_)
    y_pred = model.predict(X_test_scaled_)
    
    # If the model has a 'predict_proba' method, use it for ROC-AUC calculations
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_scaled_)[:, 1]
    else:
        # For models without predict_proba (like SVM), decision_function can be used
        y_pred_proba = model.decision_function(X_test_scaled_)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test_, y_pred)
    report = classification_report(y_test_, y_pred)
    matrix = confusion_matrix(y_test_, y_pred)
    
    # Calculate AUC-ROC score
    auc_roc = roc_auc_score(y_test_, y_pred_proba)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test_, y_pred_proba)
    
    # Display results
    print(model_name)
    print("Accuracy: ", accuracy * 100)
    print("AUC-ROC Score: ", auc_roc)
    print("Classification Report: \n", report)
    
    # Confusion Matrix Plot
    plt.figure(figsize=(4, 2))
    sns.heatmap(matrix, cmap='flare', annot=True, fmt='d')
    plt.xlabel("Predicted Class", fontsize=10) 
    plt.ylabel("True Class", fontsize=10) 
    plt.title("Confusion Matrix", fontsize=10)
    plt.show()
    
    # ROC Curve Plot
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    
    print("\n")


# ## Hyper-parameter Tunning
# ### How can I automate this process of selecting the best parameters to train my model ?

# In[57]:


# Split the data into training and testing sets
X_train_, X_test_, y_train_, y_test_ = train_test_split(dt[selected_features], labels.squeeze(),
                                                        test_size=0.2, random_state=42)

# Standardize the training data
scaler = StandardScaler()

X_train_scaled_ = scaler.fit_transform(X_train_.select_dtypes(include=['number']))
X_test_scaled_ = scaler.transform(X_test_.select_dtypes(include=['number']))

# Define the models and their parameter grids
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XG Boost": XGBClassifier(random_state=42),
    "SVC": SVC(random_state=42)
}

param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Random Forest": {
        'n_estimators': [50, 75, 100, 125, 150],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 3, 4]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    "XG Boost": {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "SVC": {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf']
    }
}

# Initialize variables to store the best model and parameters
best_model = None
best_params = None
best_score = 0

# Loop through each model and perform GridSearchCV
for model_name, model in models.items():
    print(f"Training {model_name}...")
    param_grid = param_grids[model_name]
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled_, y_train_)
    
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = model_name
        best_params = grid_search.best_params_

# Print the best model and its parameters
print(f"Best Model: {best_model}")
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score}")


# In[58]:


# Train the model with the best hyperparameters
best_model = RandomForestClassifier(random_state=42, **best_params)
best_model.fit(X_train_scaled_, y_train_)

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled_)

# Get predicted probabilities for the positive class (class 1)
y_pred_proba = best_model.predict_proba(X_test_scaled_)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test_, y_pred)
report = classification_report(y_test_, y_pred)
matrix = confusion_matrix(y_test_, y_pred)

# Calculate AUC-ROC score
auc_roc = roc_auc_score(y_test_, y_pred_proba)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test_, y_pred_proba)

# Display results
print("Accuracy: ", accuracy * 100)
print("AUC-ROC Score: ", auc_roc)
print("Classification Report: \n", report)

# Confusion Matrix Plot
plt.figure(figsize=(4, 2))
sns.heatmap(matrix, cmap='flare', annot=True, fmt='d')
plt.xlabel("Predicted Class", fontsize=10) 
plt.ylabel("True Class", fontsize=10) 
plt.title("Confusion Matrix", fontsize=10)
plt.show()

# ROC Curve Plot
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for RandomForestClassifier')
plt.legend(loc="lower right")
plt.show()
print("\n")


# In[ ]:





# In[ ]:




