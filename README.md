# cap4770-cyber-threats
This project analyzes cyberattacks across industries using historical data. We predict:
- Which industries are likely targets
- Expected financial loss ranges for future attacks

Setup Instructions
- Install dependencies:
- pip install pandas scikit-learn matplotlib seaborn kagglehub
- Download the dataset:
- kagglehub.dataset_download("atharvasoundankar/global-cybersecurity-threats-2015-2024")

Each script can be executed separately:
- data_preprocessing.py loads and cleans the data
- visualization.py shows industry attack trends and loss distribution
- feature_engineering.py encodes categorical features, scales numerics, and computes class weights
- model_industry_prediction.py trains classifiers (Logistic Regression, Decision Tree, Random Forest) to predict target industries
- model_loss_prediction.py bins loss amounts and predicts financial loss categories
- model_users_prediction.py bins affected users and predicts the impact of an attack

The Output
- Saved plots include:
- Confusion matrices per model
- Bar charts showing attack distribution
- Feature importances
