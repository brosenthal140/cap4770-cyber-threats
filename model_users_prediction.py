# Prepare features and target
X = df_encoded.drop(columns=['Target Industry', 'Country', 'Attack Source',
                           'Defense Mechanism Used', 'Incident Resolution Time (in Hours)',
                           'Financial Loss (in Million $)', 'Number of Affected Users', 'Attack Source'])
# Split data into bins
df_encoded['UsersBin'] = pd.cut(
    df_encoded['Number of Affected Users'],
    bins=6,
    labels=False,
    include_lowest=True
)

# Prepare features (X) and the new binned target (y)
X = df_encoded.drop(columns=[
    'Number of Affected Users',
    'UsersBin',
])
X = pd.get_dummies(X, drop_first=True)

y = df_encoded['UsersBin']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest (Affected Users)
random_forest_model = RandomForestClassifier(random_state=0)
random_forest_model.fit(X_train, y_train)

# Predictions and metrics
y_pred = random_forest_model.predict(X_test)
print("Random Forest Results:")
print("Macro F1-Score:", f1_score(y_test, y_pred, average='macro'))
print("Macro AUC-ROC (OvR):", roc_auc_score(y_test, random_forest_model.predict_proba(X_test), multi_class='ovr'))

# Get bin intervals
bin_intervals = pd.cut(
    df_encoded['Number of Affected Users'],
    bins=6,
    include_lowest=True
).cat.categories.astype(str).tolist()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=bin_intervals, yticklabels=bin_intervals)
plt.title('Confusion Matrix - Random Forest (Affected Users)')
plt.xlabel('Predicted Number of Affected Users')
plt.ylabel('Actual Number of Affected Users')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix_randomforest_affected_users.png')
plt.show()
