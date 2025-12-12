# ================= Wine Quality Classification Pipeline =================

#Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

#Load Data
df = pd.read_csv(r"C:\Users\maria\Documents\ai-foundation\winequality-red.csv", sep=';')
print(df.head())
df.info()
print(df['quality'].value_counts().sort_index())

#Data Cleaning
print(df.isnull().sum())
print(df.nunique())
print(f"Duplicates before: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Duplicates after: {df.duplicated().sum()}")

#Feature Selection
num_cols = [col for col in df.columns if (df[col].dtype in ["int64","float64"]) & (df[col].nunique() > 50)]
print(f"Numeric cols (continuous features): {num_cols}")
target = [col for col in df.columns if df[col].nunique()<10]
print(f"Target column: {target}")

#Target Encoding
y = df['quality']
X = df.drop(['quality'], axis=1)

# Encode target labels 3-8 → 0-5
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

#Feature Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
print(f"Training set size: {X_train_scaled.shape[0]} rows")

#Model Definitions
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='multinomial', solver='lbfgs', random_state=42)
svc = SVC(probability=True, class_weight='balanced', random_state=42)

xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Voting ensemble
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('svc', svc), ('xgb', xgb)],
    voting='soft'
)

#Train Random Forest separately for feature importances
rf.fit(X_train_scaled, y_train)
rf_feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Top 5 Random Forest Feature Importances ---")
print(rf_feature_importances.head())

# Train XGBoost separately for feature importances
xgb.fit(X_train_scaled, y_train)
xgb_importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Top 5 XGBoost Feature Importances ---")
print(xgb_importances.head())

#Fit Voting Classifier 
voting_clf.fit(X_train_scaled, y_train)
y_pred = voting_clf.predict(X_test_scaled)

#Classification Report 
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

#Confusion Matrix
y_test_orig = le.inverse_transform(y_test)
y_pred_orig = le.inverse_transform(y_pred)

cm = confusion_matrix(y_test_orig, y_pred_orig)

group_labels = {
    3: "Low (3)",
    4: "Low (4)",
    5: "Mid (5)",
    6: "Mid (6)",
    7: "High (7)",
    8: "High (8)"
}

class_labels = np.sort(np.unique(y_test_orig))
xtick_labels = [group_labels[c] for c in class_labels]
ytick_labels = [group_labels[c] for c in class_labels]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', xticklabels=xtick_labels, yticklabels=ytick_labels)
plt.title('Confusion Matrix (Wine Quality 3–8)')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Correlation Heatmap 
corr = df.corr()
corr_quality = corr['quality'].sort_values(ascending=False)
ordered_cols = corr_quality.index
corr_ordered = corr.loc[ordered_cols, ordered_cols]

plt.figure(figsize=(12,10))
sns.heatmap(corr_ordered, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap (Features Ordered by Correlation with Quality)', fontsize=16)
plt.show()

#Countplot
plt.figure(figsize=(8, 6))
sns.countplot(x='quality', data=df)
plt.title('Wine Quality Distribution')
plt.show()

print("\nProof of Concept Complete. Classification Report and Confusion Matrix demonstrate model performance.")
