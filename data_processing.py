import pandas as pd
import os
import kagglehub

# Download and load dataset
path = kagglehub.dataset_download("atharvasoundankar/global-cybersecurity-threats-2015-2024")
files = os.listdir(path)
file_path = os.path.join(path, files[0])
df = pd.read_csv(file_path)

# Check for missing data
df.info()
df.isnull().sum()

# Create threat levels
industry_threats = df.groupby(['Year', 'Target Industry']).size().reset_index(name='AttackCount')

def label_threat_level(count):
    if count <= 50:
        return 'Low'
    elif count <= 200:
        return 'Medium'
    else:
        return 'High'

industry_threats['ThreatLevel'] = industry_threats['AttackCount'].apply(label_threat_level)
df = df.merge(industry_threats[['Year', 'Target Industry', 'ThreatLevel']], on=['Year', 'Target Industry'], how='left')
