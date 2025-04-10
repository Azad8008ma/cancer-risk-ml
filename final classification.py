import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, accuracy_score, 
                             precision_score, recall_score, f1_score, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn import tree, metrics
from scipy.stats import friedmanchisquare
import shap
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------
# 1. Load Data
# ------------------------------
filepath = './early_onset_LARC_HR_combinations.csv'
df = pd.read_csv(filepath)

print("Sample of data:")
print(df.head())

print("\nStatistical summary of data:")
print(df.describe())

# ------------------------------
# 2. Create target column 'risk' by binning HR Total based on median
# ------------------------------
median_hr = df["HR Total"].median()
df["risk"] = (df["HR Total"] >= median_hr).astype(int)

print("\nHR Total median:", median_hr)
print("Risk distribution (0: low, 1: high):")
print(df["risk"].value_counts())

# ------------------------------
# 3. Encoding categorical features
# ------------------------------
df_encoded = df.copy()
label_encoders = {}

for col in df.columns[:-1]:  # Exclude HR Total column
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save label encoder for later inspection

# ------------------------------
# 4. Calculate correlation matrix and plot Heatmap
# ------------------------------
correlation_matrix = df_encoded.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Feature Correlations", fontweight='bold')
plt.show()

# ------------------------------
# 5. Convert categorical features to dummy variables (One-Hot Encoding)
# ------------------------------
categorical_features = ["Radiotherapy", "Gender", "Marital status", "Race", 
                        "Pathologic grade", "Histologic type", "T staging", "N staging",
                        "Chemotherapy", "RNE", "CEA", "Tumor size (cm)"]
df_encoded = pd.get_dummies(df, columns=categorical_features)

# ------------------------------
# 6. Define feature matrix (X) and label (y)
# ------------------------------
df_encoded=df_encoded.sample(frac=1 , random_state=42)
X = df_encoded.drop(["HR Total", "risk"], axis=1)
y = df_encoded["risk"]

# Check for missing values
print("\nMissing values in data:")
print(df_encoded.isnull().sum())

# ------------------------------
# 7. Feature scaling with MinMaxScaler
# ------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 8. Split data into training and test sets (70%-30%)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

# ------------------------------
# 9. Exploratory Data Analysis (EDA)
# ------------------------------

print("\n--- Combined Data Distribution Plots ---")

# Define the categorical features
categorical_features = ["Radiotherapy", "Gender", "Marital status", "Race", 
                        "Pathologic grade", "Histologic type", "T staging", "N staging",
                        "Chemotherapy", "RNE", "CEA", "Tumor size (cm)"]

# 9.1. Plot histogram for HR Total (separate plot)
print("\n--- Histogram for HR Total ---")
plt.figure(figsize=(8, 6))
sns.histplot(df['HR Total'], bins=20, kde=True, color='skyblue', edgecolor='black')
plt.title("HR Total Distribution", fontsize=14, fontweight='bold')
plt.xlabel("HR Total", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("hr_total_histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# 9.2. Combine all other distribution plots into a single figure
print("\n--- Combined Distribution Analysis ---")

# Calculate total number of plots needed
num_numerical_plots = len(df.select_dtypes(include=['float64', 'int64']).columns) - 1  # Exclude HR Total
num_categorical_plots = len(categorical_features)
total_plots = num_numerical_plots * 2 + num_categorical_plots  # Histograms + Boxplots for numerical data

# Create a grid layout for subplots
num_cols = 3  # Number of columns in the grid
num_rows = (total_plots + num_cols - 1) // num_cols  # Calculate number of rows dynamically

# Create the figure and axes for subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Plot numerical data distributions (Histograms and Boxplots)
plot_index = 0
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if col != "HR Total":  # Skip HR Total
        # Histogram
        sns.histplot(df[col], bins=20, kde=True, color='skyblue', edgecolor='black', ax=axes[plot_index])
        axes[plot_index].set_title(f"Histogram of {col}", fontsize=12, fontweight='bold')
        axes[plot_index].grid(linestyle='--', linewidth=0.5)
        plot_index += 1
        
        # Boxplot
        sns.boxplot(data=df[col], color='lightgreen', ax=axes[plot_index])
        axes[plot_index].set_title(f"Boxplot of {col}", fontsize=12, fontweight='bold')
        plot_index += 1

# Plot categorical data distributions (Count Plots)
for col in categorical_features:
    sns.countplot(x=col, data=df, palette='Set2', ax=axes[plot_index])
    axes[plot_index].set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
    axes[plot_index].tick_params(axis='x', rotation=45)
    plot_index += 1

# Hide any unused subplots
for j in range(plot_index, len(axes)):
    axes[j].axis('off')

# Final adjustments
plt.tight_layout()
plt.suptitle("Combined Data Distribution Analysis", fontsize=18, fontweight='bold', y=1.02)
plt.savefig("combined_data_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
# ------------------------------
# 10. Model Training and Evaluation
# ------------------------------

# Define StratifiedKFold object for cross-validation
skf = StratifiedKFold(n_splits=5)

# -----------------------------------------------------------
# AdaBoost Model
# -----------------------------------------------------------
param_grid_adaboost = {'n_estimators': [100, 500, 1000, 1500]}
grid_adaboost = GridSearchCV(estimator=AdaBoostClassifier(), 
                             param_grid=param_grid_adaboost, cv=4)
grid_adaboost.fit(X_train, y_train)
print("\nBest AdaBoost parameters:", grid_adaboost.best_params_)
adaboost = AdaBoostClassifier(n_estimators=grid_adaboost.best_params_['n_estimators'])

# Cross-validation for AdaBoost
scores_acc = cross_val_score(adaboost, X_train, y_train, cv=skf, scoring='accuracy')
scores_prec = cross_val_score(adaboost, X_train, y_train, cv=skf, scoring='precision')
scores_rec = cross_val_score(adaboost, X_train, y_train, cv=skf, scoring='recall')
scores_f1  = cross_val_score(adaboost, X_train, y_train, cv=skf, scoring='f1')

print("AdaBoost - CV Accuracy:", scores_acc, "Mean:", scores_acc.mean(), "Std:", scores_acc.std())
print("AdaBoost - CV Precision:", scores_prec, "Mean:", scores_prec.mean(), "Std:", scores_prec.std())
print("AdaBoost - CV Recall:", scores_rec, "Mean:", scores_rec.mean(), "Std:", scores_rec.std())
print("AdaBoost - CV F1:", scores_f1, "Mean:", scores_f1.mean(), "Std:", scores_f1.std())

adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
y_pred_adaboost_prob = adaboost.predict_proba(X_test)[:,1]

# -----------------------------------------------------------
# Bagging Model
# -----------------------------------------------------------
param_grid_bag = {'n_estimators': [100, 500, 1000, 1500]}
grid_bag = GridSearchCV(estimator=BaggingClassifier(), param_grid=param_grid_bag, cv=4)
grid_bag.fit(X_train, y_train)
print("\nBest Bagging parameters:", grid_bag.best_params_)
bagging = BaggingClassifier(n_estimators=grid_bag.best_params_['n_estimators'], oob_score=True)

# Cross-validation for Bagging
scores_acc_bag = cross_val_score(bagging, X_train, y_train, cv=skf, scoring='accuracy')
scores_prec_bag = cross_val_score(bagging, X_train, y_train, cv=skf, scoring='precision')
scores_rec_bag = cross_val_score(bagging, X_train, y_train, cv=skf, scoring='recall')
scores_f1_bag  = cross_val_score(bagging, X_train, y_train, cv=skf, scoring='f1')

print("Bagging - CV Accuracy:", scores_acc_bag, "Mean:", scores_acc_bag.mean(), "Std:", scores_acc_bag.std())
print("Bagging - CV Precision:", scores_prec_bag, "Mean:", scores_prec_bag.mean(), "Std:", scores_prec_bag.std())
print("Bagging - CV Recall:", scores_rec_bag, "Mean:", scores_rec_bag.mean(), "Std:", scores_rec_bag.std())
print("Bagging - CV F1:", scores_f1_bag, "Mean:", scores_f1_bag.mean(), "Std:", scores_f1_bag.std())

bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
y_pred_bagging_prob = bagging.predict_proba(X_test)[:,1]

# -----------------------------------------------------------
# Random Forest Model
# -----------------------------------------------------------
param_grid_rf = {'n_estimators': [100, 500, 1000, 1500]}
grid_rf = GridSearchCV(estimator=RandomForestClassifier(max_features='sqrt', oob_score=True), 
                       param_grid=param_grid_rf, cv=4)
grid_rf.fit(X_train, y_train)
print("\nBest Random Forest parameters:", grid_rf.best_params_)
rf = RandomForestClassifier(n_estimators=grid_rf.best_params_['n_estimators'], 
                            max_features='sqrt', oob_score=True)

# Cross-validation for Random Forest
scores_acc_rf = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
scores_prec_rf = cross_val_score(rf, X_train, y_train, cv=skf, scoring='precision')
scores_rec_rf = cross_val_score(rf, X_train, y_train, cv=skf, scoring='recall')
scores_f1_rf  = cross_val_score(rf, X_train, y_train, cv=skf, scoring='f1')

print("Random Forest - CV Accuracy:", scores_acc_rf, "Mean:", scores_acc_rf.mean(), "Std:", scores_acc_rf.std())
print("Random Forest - CV Precision:", scores_prec_rf, "Mean:", scores_prec_rf.mean(), "Std:", scores_prec_rf.std())
print("Random Forest - CV Recall:", scores_rec_rf, "Mean:", scores_rec_rf.mean(), "Std:", scores_rec_rf.std())
print("Random Forest - CV F1:", scores_f1_rf, "Mean:", scores_f1_rf.mean(), "Std:", scores_f1_rf.std())

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_rf_prob = rf.predict_proba(X_test)[:,1]

# -----------------------------------------------------------
# Decision Tree Model
# -----------------------------------------------------------
tree_param = {'criterion': ['gini', 'entropy'], 'max_depth': [4,5,6,7,8,9,10]}
grid_tree = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=tree_param, scoring='f1', cv=4)
grid_tree.fit(X_train, y_train)
print("\nBest Decision Tree parameters:", grid_tree.best_params_)
dectree = tree.DecisionTreeClassifier(max_depth=grid_tree.best_params_['max_depth'], 
                                      criterion=grid_tree.best_params_['criterion'])

# Cross-validation for Decision Tree
scores_acc_tree = cross_val_score(dectree, X_train, y_train, cv=skf, scoring='accuracy')
scores_prec_tree = cross_val_score(dectree, X_train, y_train, cv=skf, scoring='precision')
scores_rec_tree = cross_val_score(dectree, X_train, y_train, cv=skf, scoring='recall')
scores_f1_tree  = cross_val_score(dectree, X_train, y_train, cv=skf, scoring='f1')

print("Decision Tree - CV Accuracy:", scores_acc_tree, "Mean:", scores_acc_tree.mean(), "Std:", scores_acc_tree.std())
print("Decision Tree - CV Precision:", scores_prec_tree, "Mean:", scores_prec_tree.mean(), "Std:", scores_prec_tree.std())
print("Decision Tree - CV Recall:", scores_rec_tree, "Mean:", scores_rec_tree.mean(), "Std:", scores_rec_tree.std())
print("Decision Tree - CV F1:", scores_f1_tree, "Mean:", scores_f1_tree.mean(), "Std:", scores_f1_tree.std())

dectree.fit(X_train, y_train)
y_pred_tree = dectree.predict(X_test)
y_pred_tree_prob = dectree.predict_proba(X_test)[:,1]

# -----------------------------------------------------------
# Plot ROC curves for all models (in a 2x2 grid)
# -----------------------------------------------------------
# Model data
models_dict = {
    "AdaBoost": y_pred_adaboost_prob,
    "Bagging": y_pred_bagging_prob,
    "RandomForest": y_pred_rf_prob,
    "DecisionTree": y_pred_tree_prob
}

# Create a figure with 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("ROC Curve Analysis", fontweight='bold', fontsize=16)

# Flatten axes for easier use
axes = axes.flatten()

# Plot ROC curves for each model
for ax, (name, y_proba) in zip(axes, models_dict.items()):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}", color='blue')
    ax.plot([0, 1], [0, 1], color='orange', linestyle='--')  # Reference line
    
    # Chart settings
    ax.set_title(name, fontweight='bold', fontsize=12)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.legend(prop={'size': 10}, loc='lower right')
    ax.grid(linestyle='-', linewidth=0.25)

# Adjust spacing between charts
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Top space for main title
plt.savefig('roc_curves_grid.png', dpi=400, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# Extract and plot feature importance from Random Forest (Top 6 features)
# -----------------------------------------------------------
feature_importances = rf.feature_importances_
features = X.columns
feature_scores = pd.Series(feature_importances, index=features).sort_values(ascending=False)
top_features = feature_scores.head(6).round(3)

print("\nTop 6 Features (Feature Importances):")
print(top_features)

ax = top_features.plot(kind='bar', color='blue', figsize=(8,6))
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() * 1.005, p.get_height() * 1.005), weight='bold')

plt.title('Feature Importance (Top 6)', fontweight='bold', fontsize=13)
plt.grid(linestyle='-', linewidth=0.25)
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# Plot confusion matrices for all models in one figure
# -----------------------------------------------------------
from sklearn.metrics import ConfusionMatrixDisplay

# Define models and their names
models = {
    "AdaBoost": adaboost,
    "Bagging": bagging,
    "RandomForest": rf,
    "DecisionTree": dectree
}

# Create a figure with subplots for each confusion matrix
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows and 2 columns
axes = axes.flatten()  # Convert 2D array to 1D for easier access

# Plot confusion matrices
for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    
    # Add labels (A), (B), (C), (D) to each subplot
    axes[i].set_title(f"({chr(65 + i)}) {name}", fontsize=14, fontweight='bold')
    axes[i].set_xlabel("Predicted", fontsize=12)
    axes[i].set_ylabel("Actual", fontsize=12)

# Final figure adjustments
plt.tight_layout()
plt.suptitle("Confusion Matrices of All Models", fontsize=16, fontweight='bold', y=1.02)
plt.savefig('combined_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# Create model performance comparison table
# -----------------------------------------------------------
performance_scores = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    performance_scores.append([name, acc, prec, rec, f1])

performance_df = pd.DataFrame(performance_scores, 
                              columns=["Model Name", "Accuracy", "Precision", "Recall", "F1"])
performance_df = performance_df.round(2)

print("\nModel Performance Comparison Table:")
print(performance_df)

# Save performance table to Excel file
performance_df.to_excel("performance_comparison.xlsx", index=False)
print("Performance table saved to performance_comparison.xlsx file.")

# Plot bar chart for performance comparison
ax = performance_df.plot(x="Model Name", y=["Accuracy", "Precision", "Recall", "F1"], 
                         kind="bar", figsize=(10,6), rot=70, width=0.7)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() * 1.005, p.get_height() * 1.005), weight='bold')

ax.set_title('Performance Comparison of Classifiers', fontweight="bold") 
ax.set_ylabel("Score")
plt.grid(linestyle='-', linewidth=0.25)
plt.savefig('performance_comparison.png', bbox_inches='tight')
plt.show()

# ========================
# 1. Synthetic Data Generation
# ========================
np.random.seed(42)
n_samples = 2000

data = pd.DataFrame({
    'Age': np.clip(np.random.normal(60, 15, n_samples), 30, 90).astype(int),
    'T_stage': np.random.choice(['T1', 'T2', 'T3', 'T4'], n_samples, p=[0.1, 0.3, 0.5, 0.1]),
    'N_stage': np.random.choice(['N0', 'N1', 'N2'], n_samples, p=[0.4, 0.4, 0.2]),
    'Tumor_size': np.abs(np.random.normal(4.0, 2.0, n_samples)),
    'Marital_status': np.random.choice(['Married', 'Single'], n_samples, p=[0.6, 0.4]),
    'Pathologic_grade': np.random.choice(['I', 'II', 'III'], n_samples, p=[0.2, 0.5, 0.3]),
    'CEA_level': np.random.choice(['Normal', 'Elevated'], n_samples, p=[0.7, 0.3]),
    'Hazard_risk': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])  # Binary classification target
})

# ========================
# 2. Preprocessing Pipeline
# ========================
X = data.drop('Hazard_risk', axis=1)
y = data['Hazard_risk']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Tumor_size']),
        ('cat', OneHotEncoder(handle_unknown="ignore"), ['T_stage', 'N_stage', 'Marital_status', 'Pathologic_grade', 'CEA_level'])
    ]
)

# ========================
# 3. Model Training
# ========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# ========================
# 4. SHAP Analysis & Visualization
# ========================
# Process test data
X_test_processed = model.named_steps['preprocessor'].transform(X_test)

# Extract feature names after OneHotEncoding
cat_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
feature_names = ['Age', 'Tumor_size'] + list(cat_features)

# Using KernelExplainer for AdaBoost
explainer = shap.KernelExplainer(model.named_steps['classifier'].predict, X_test_processed)
shap_values = explainer.shap_values(X_test_processed)

# SHAP summary plot
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, plot_type="bar", show=False)
plt.title("Top Predictive Features for CRC Hazard Risk", fontsize=14)
plt.xlabel("Mean Absolute SHAP Value (Impact on Risk Prediction)", fontsize=12)
plt.ylabel("Clinical Features", fontsize=12)
plt.tight_layout()
plt.show()
