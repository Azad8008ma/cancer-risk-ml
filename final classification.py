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

# ------------------------------
# 1. بارگذاری داده‌ها
# ------------------------------
filepath = './early_onset_LARC_HR_combinations.csv'
df = pd.read_csv(filepath)

print("نمونه‌ای از داده‌ها:")
print(df.head())

print("\nخلاصه آماری داده‌ها:")
print(df.describe())

# ------------------------------
# 2. ایجاد ستون هدف 'risk' با باین‌بندی HR Total بر اساس میانه
# ------------------------------
median_hr = df["HR Total"].median()
df["risk"] = (df["HR Total"] >= median_hr).astype(int)

print("\nمیانه HR Total:", median_hr)
print("توزیع ریسک (0: پایین، 1: بالا):")
print(df["risk"].value_counts())

# ------------------------------
# 3. کدگذاری ویژگی‌های کاتگوریک
# ------------------------------
df_encoded = df.copy()
label_encoders = {}

for col in df.columns[:-1]:  # به جز ستون HR Total
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # ذخیره label encoder برای بررسی‌های بعدی

# ------------------------------
# 4. محاسبه ماتریس همبستگی و رسم Heatmap
# ------------------------------
correlation_matrix = df_encoded.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Feature Correlations", fontweight='bold')
plt.show()

# ------------------------------
# 5. تبدیل ویژگی‌های کاتگوریک به متغیرهای دامی (One-Hot Encoding)
# ------------------------------
categorical_features = ["Radiotherapy", "Gender", "Marital status", "Race", 
                        "Pathologic grade", "Histologic type", "T staging", "N staging",
                        "Chemotherapy", "RNE", "CEA", "Tumor size (cm)"]
df_encoded = pd.get_dummies(df, columns=categorical_features)

# ------------------------------
# 6. تعریف ماتریس ویژگی‌ها (X) و برچسب (y)
# ------------------------------
X = df_encoded.drop(["HR Total", "risk"], axis=1)
y = df_encoded["risk"]

# بررسی مقدارهای گمشده
print("\nمقدارهای گمشده در داده‌ها:")
print(df_encoded.isnull().sum())

# ------------------------------
# 7. مقیاس‌بندی ویژگی‌ها با MinMaxScaler
# ------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 8. تقسیم داده به مجموعه‌های آموزش و تست (70%-30%)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

# ------------------------------
# 9. تحلیل اکتشافی داده (EDA)
# ------------------------------

# 9.1. رسم هیستوگرام برای HR Total
print("\n--- هیستوگرام برای HR Total ---")
plt.figure(figsize=(8, 6))
sns.histplot(df['HR Total'], bins=20, kde=True, color='skyblue', edgecolor='black')
plt.title("توزیع HR Total", fontsize=14, fontweight='bold')
plt.xlabel("HR Total", fontsize=12)
plt.ylabel("فراوانی", fontsize=12)
plt.grid(linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 9.2. شناسایی ناهنجاری‌ها (Outliers) برای HR Total با نمودار جعبه‌ای
print("\n--- شناسایی ناهنجاری‌ها (Outliers) برای HR Total ---")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df['HR Total'], color='lightgreen')
plt.title("Boxplot for HR Total", fontsize=14, fontweight='bold')
plt.xlabel("HR Total", fontsize=12)
plt.grid(linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 9.3. بررسی تعادل کلاس‌ها
print("\n--- بررسی تعادل کلاس‌ها ---")
plt.figure(figsize=(6, 4))
sns.countplot(x='risk', data=df, palette='Set1')
plt.title("Class Distribution (Risk)", fontsize=14, fontweight='bold')
plt.xlabel("Risk (0: Low, 1: High)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------------------
# 10. آموزش و ارزیابی مدل‌ها
# ------------------------------

# تعریف یک شی StratifiedKFold برای اعتبارسنجی متقابل
skf = StratifiedKFold(n_splits=5)

# -----------------------------------------------------------
# مدل AdaBoost
# -----------------------------------------------------------
param_grid_adaboost = {'n_estimators': [100, 500, 1000, 1500]}
grid_adaboost = GridSearchCV(estimator=AdaBoostClassifier(), 
                             param_grid=param_grid_adaboost, cv=4)
grid_adaboost.fit(X_train, y_train)
print("\nبهترین پارامترهای AdaBoost:", grid_adaboost.best_params_)
adaboost = AdaBoostClassifier(n_estimators=grid_adaboost.best_params_['n_estimators'])

# اعتبارسنجی متقابل برای AdaBoost
scores_acc = cross_val_score(adaboost, X_train, y_train, cv=skf, scoring='accuracy')
scores_prec = cross_val_score(adaboost, X_train, y_train, cv=skf, scoring='precision')
scores_rec = cross_val_score(adaboost, X_train, y_train, cv=skf, scoring='recall')
scores_f1  = cross_val_score(adaboost, X_train, y_train, cv=skf, scoring='f1')

print("AdaBoost - دقت CV:", scores_acc, "میانگین:", scores_acc.mean(), "انحراف معیار:", scores_acc.std())
print("AdaBoost - دقت مثبت CV:", scores_prec, "میانگین:", scores_prec.mean(), "انحراف معیار:", scores_prec.std())
print("AdaBoost - بازیابی CV:", scores_rec, "میانگین:", scores_rec.mean(), "انحراف معیار:", scores_rec.std())
print("AdaBoost - F1 CV:", scores_f1, "میانگین:", scores_f1.mean(), "انحراف معیار:", scores_f1.std())

adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
y_pred_adaboost_prob = adaboost.predict_proba(X_test)[:,1]

# -----------------------------------------------------------
# مدل Bagging
# -----------------------------------------------------------
param_grid_bag = {'n_estimators': [100, 500, 1000, 1500]}
grid_bag = GridSearchCV(estimator=BaggingClassifier(), param_grid=param_grid_bag, cv=4)
grid_bag.fit(X_train, y_train)
print("\nبهترین پارامترهای Bagging:", grid_bag.best_params_)
bagging = BaggingClassifier(n_estimators=grid_bag.best_params_['n_estimators'], oob_score=True)

# اعتبارسنجی متقابل برای Bagging
scores_acc_bag = cross_val_score(bagging, X_train, y_train, cv=skf, scoring='accuracy')
scores_prec_bag = cross_val_score(bagging, X_train, y_train, cv=skf, scoring='precision')
scores_rec_bag = cross_val_score(bagging, X_train, y_train, cv=skf, scoring='recall')
scores_f1_bag  = cross_val_score(bagging, X_train, y_train, cv=skf, scoring='f1')

print("Bagging - دقت CV:", scores_acc_bag, "میانگین:", scores_acc_bag.mean(), "انحراف معیار:", scores_acc_bag.std())
print("Bagging - دقت مثبت CV:", scores_prec_bag, "میانگین:", scores_prec_bag.mean(), "انحراف معیار:", scores_prec_bag.std())
print("Bagging - بازیابی CV:", scores_rec_bag, "میانگین:", scores_rec_bag.mean(), "انحراف معیار:", scores_rec_bag.std())
print("Bagging - F1 CV:", scores_f1_bag, "میانگین:", scores_f1_bag.mean(), "انحراف معیار:", scores_f1_bag.std())

bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
y_pred_bagging_prob = bagging.predict_proba(X_test)[:,1]

# -----------------------------------------------------------
# مدل Random Forest
# -----------------------------------------------------------
param_grid_rf = {'n_estimators': [100, 500, 1000, 1500]}
grid_rf = GridSearchCV(estimator=RandomForestClassifier(max_features='sqrt', oob_score=True), 
                       param_grid=param_grid_rf, cv=4)
grid_rf.fit(X_train, y_train)
print("\nبهترین پارامترهای Random Forest:", grid_rf.best_params_)
rf = RandomForestClassifier(n_estimators=grid_rf.best_params_['n_estimators'], 
                            max_features='sqrt', oob_score=True)

# اعتبارسنجی متقابل برای Random Forest
scores_acc_rf = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
scores_prec_rf = cross_val_score(rf, X_train, y_train, cv=skf, scoring='precision')
scores_rec_rf = cross_val_score(rf, X_train, y_train, cv=skf, scoring='recall')
scores_f1_rf  = cross_val_score(rf, X_train, y_train, cv=skf, scoring='f1')

print("Random Forest - دقت CV:", scores_acc_rf, "میانگین:", scores_acc_rf.mean(), "انحراف معیار:", scores_acc_rf.std())
print("Random Forest - دقت مثبت CV:", scores_prec_rf, "میانگین:", scores_prec_rf.mean(), "انحراف معیار:", scores_prec_rf.std())
print("Random Forest - بازیابی CV:", scores_rec_rf, "میانگین:", scores_rec_rf.mean(), "انحراف معیار:", scores_rec_rf.std())
print("Random Forest - F1 CV:", scores_f1_rf, "میانگین:", scores_f1_rf.mean(), "انحراف معیار:", scores_f1_rf.std())

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_rf_prob = rf.predict_proba(X_test)[:,1]

# -----------------------------------------------------------
# مدل Decision Tree
# -----------------------------------------------------------
tree_param = {'criterion': ['gini', 'entropy'], 'max_depth': [4,5,6,7,8,9,10]}
grid_tree = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=tree_param, scoring='f1', cv=4)
grid_tree.fit(X_train, y_train)
print("\nبهترین پارامترهای Decision Tree:", grid_tree.best_params_)
dectree = tree.DecisionTreeClassifier(max_depth=grid_tree.best_params_['max_depth'], 
                                      criterion=grid_tree.best_params_['criterion'])

# اعتبارسنجی متقابل برای Decision Tree
scores_acc_tree = cross_val_score(dectree, X_train, y_train, cv=skf, scoring='accuracy')
scores_prec_tree = cross_val_score(dectree, X_train, y_train, cv=skf, scoring='precision')
scores_rec_tree = cross_val_score(dectree, X_train, y_train, cv=skf, scoring='recall')
scores_f1_tree  = cross_val_score(dectree, X_train, y_train, cv=skf, scoring='f1')

print("Decision Tree - دقت CV:", scores_acc_tree, "میانگین:", scores_acc_tree.mean(), "انحراف معیار:", scores_acc_tree.std())
print("Decision Tree - دقت مثبت CV:", scores_prec_tree, "میانگین:", scores_prec_tree.mean(), "انحراف معیار:", scores_prec_tree.std())
print("Decision Tree - بازیابی CV:", scores_rec_tree, "میانگین:", scores_rec_tree.mean(), "انحراف معیار:", scores_rec_tree.std())
print("Decision Tree - F1 CV:", scores_f1_tree, "میانگین:", scores_f1_tree.mean(), "انحراف معیار:", scores_f1_tree.std())

dectree.fit(X_train, y_train)
y_pred_tree = dectree.predict(X_test)
y_pred_tree_prob = dectree.predict_proba(X_test)[:,1]

# -----------------------------------------------------------
# رسم منحنی ROC برای همه مدل‌ها
# -----------------------------------------------------------
plt.figure(figsize=(6,4))
models_dict = {
    "AdaBoost": y_pred_adaboost_prob,
    "Bagging": y_pred_bagging_prob,
    "RandomForest": y_pred_rf_prob,
    "DecisionTree": y_pred_tree_prob
}

for name, y_proba in models_dict.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name}, AUC={auc_score:.3f}")

plt.plot([0,1], [0,1], color='orange', linestyle='--')
plt.xlabel("False Positive Rate", fontsize=13)
plt.ylabel("True Positive Rate", fontsize=13)
plt.title("ROC Curve Analysis", fontweight='bold', fontsize=13)
plt.legend(prop={'size':10}, loc='lower right')
plt.grid(linestyle='-', linewidth=0.25)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# استخراج و رسم اهمیت ویژگی‌ها از مدل Random Forest (انتخاب 6 ویژگی برتر)
# -----------------------------------------------------------
feature_importances = rf.feature_importances_
features = X.columns
feature_scores = pd.Series(feature_importances, index=features).sort_values(ascending=False)
top_features = feature_scores.head(6).round(3)

print("\n6 ویژگی برتر (Feature Importances):")
print(top_features)

ax = top_features.plot(kind='bar', color='blue', figsize=(8,6))
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() * 1.005, p.get_height() * 1.005), weight='bold')

plt.title('Feature Importance (Top 6)', fontweight='bold', fontsize=13)
plt.grid(linestyle='-', linewidth=0.25)
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# رسم ماتریس‌های درهمی برای همه مدل‌ها در یک شکل
# -----------------------------------------------------------
from sklearn.metrics import ConfusionMatrixDisplay

# تعریف لیست مدل‌ها و نام‌های آن‌ها
models = {
    "AdaBoost": adaboost,
    "Bagging": bagging,
    "RandomForest": rf,
    "DecisionTree": dectree
}

# ایجاد یک شکل با زیرشکل‌های جداگانه برای هر ماتریس درهمی
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 ردیف و 2 ستون
axes = axes.flatten()  # تبدیل آرایه 2D به 1D برای دسترسی آسان‌تر

# رسم ماتریس‌های درهمی
for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # رسم ماتریس درهمی با استفاده از ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    
    # افزودن برچسب (A), (B), (C), (D) به هر زیرشکل
    axes[i].set_title(f"({chr(65 + i)}) {name}", fontsize=14, fontweight='bold')
    axes[i].set_xlabel("Predicted", fontsize=12)
    axes[i].set_ylabel("Actual", fontsize=12)

# تنظیمات نهایی شکل
plt.tight_layout()
plt.suptitle("Confusion Matrices of All Models", fontsize=16, fontweight='bold', y=1.02)
plt.savefig('combined_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# ایجاد جدول مقایسه عملکرد مدل‌ها
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

print("\nجدول مقایسه عملکرد مدل‌ها:")
print(performance_df)

# ذخیره جدول عملکرد در فایل اکسل
performance_df.to_excel("performance_comparison.xlsx", index=False)
print("جدول عملکرد در فایل performance_comparison.xlsx ذخیره شد.")

# رسم نمودار میله‌ای برای مقایسه عملکرد
ax = performance_df.plot(x="Model Name", y=["Accuracy", "Precision", "Recall", "F1"], 
                         kind="bar", figsize=(10,6), rot=70, width=0.7)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() * 1.005, p.get_height() * 1.005), weight='bold')

ax.set_title('Performance Comparison of Classifiers', fontweight="bold") 
ax.set_ylabel("Score")
plt.grid(linestyle='-', linewidth=0.25)
plt.savefig('performance_comparison.png', bbox_inches='tight')
plt.show()