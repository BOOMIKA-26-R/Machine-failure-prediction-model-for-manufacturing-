import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('machine_data.csv')
X = df.drop('Machine_Failure', axis=1)
y = df['Machine_Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'machine_model.pkl')
print(f"Machine Failure Model saved. Accuracy: {model.score(X_test, y_test):.2f}")
