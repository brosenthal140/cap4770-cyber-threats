from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['Attack Type', 'Security Vulnerability Type', 'ThreatLevel'])

# Scale features
scaler = StandardScaler()
df_encoded[['Financial_Loss_scaled', 'Affected_Users_scaled']] = scaler.fit_transform(
    df_encoded[['Financial Loss (in Million $)', 'Number of Affected Users']]
)

# Compute class weights
classes = df_encoded['Target Industry'].unique()
class_weights = compute_class_weight('balanced', classes=classes, y=df_encoded['Target Industry'])
class_weight_dict = dict(zip(classes, class_weights))
