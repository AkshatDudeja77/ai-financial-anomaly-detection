import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Select relevant features for anomaly detection
features = df[['Amount', 'Time']]

# Create and train the Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = model.fit_predict(features)

# Convert output to readable labels
df['Anomaly'] = df['Anomaly'].map({-1: 'Anomaly', 1: 'Normal'})

# Save results to a new file
df.to_csv("anomaly_results.csv", index=False)

print("Anomaly detection complete. Results saved as anomaly_results.csv")
