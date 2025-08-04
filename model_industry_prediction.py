from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare data
X = df_encoded.drop(columns=['Target Industry', 'Country', 'Attack Source',
                           'Defense Mechanism Used', 'Incident Resolution Time (in Hours)',
                           'Financial Loss (in Million $)', 'Number of Affected Users'])  # relevant column drops
y = df_encoded['Target Industry']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression
log_reg = LogisticRegression(multi_class='multinomial', class_weight=class_weight_dict, max_iter=1000)
log_reg.fit(X_train, y_train)

# Train Random Forest
random_forest_model = RandomForestClassifier(random_state=0)
random_forest_model.fit(X_train, y_train)

# Train Decision Tree
dt = DecisionTreeClassifier(class_weight=class_weight_dict, random_state=42)
dt.fit(X_train, y_train)

# Metrics and confusion matrices
# Logistic Regression
# Predictions and metrics
y_pred_logreg = log_reg.predict(X_test)
print("Logistic Regression Results:")
print("Macro F1-Score:", f1_score(y_test, y_pred_logreg, average='macro'))
print("Macro AUC-ROC (OvR):", roc_auc_score(y_test, log_reg.predict_proba(X_test), multi_class='ovr'))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Industry')
plt.ylabel('Actual Industry')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix_logreg.png')
plt.show()

# Random Forest
# Predictions and metrics
y_pred_rf = random_forest_model.predict(X_test)
print("Random Forest Results:")
print("Macro F1-Score:", f1_score(y_test, y_pred_rf, average='macro'))
print("Macro AUC-ROC (OvR):", roc_auc_score(y_test, random_forest_model.predict_proba(X_test), multi_class='ovr'))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Industry')
plt.ylabel('Actual Industry')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix_randomforest.png')
plt.show()

# Decision Tree
# Predictions and metrics
y_pred_dt = dt.predict(X_test)
print("Decision Tree Results:")
print("Macro F1-Score:", f1_score(y_test, y_pred_dt, average='macro'))
print("Macro AUC-ROC (One-vs-Rest):", roc_auc_score(y_test, dt.predict_proba(X_test), multi_class='ovr'))

# Confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', xticklabels=class_weight_dict.keys(), yticklabels=class_weight_dict.keys())
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted Industry')
plt.ylabel('Actual Industry')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix_decisiontree.png')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Feature Importance (Decision Tree):")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
plt.title('Top 10 Feature Importance - Decision Tree')
plt.tight_layout()
plt.savefig('feature_importance_dt.png')
plt.show()
