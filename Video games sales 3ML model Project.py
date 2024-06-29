import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
file_path = 'videogamesales/vgsales.csv'
df = pd.read_csv(file_path)

# Drop rows with any null values
df = df.dropna()

# Define features (X) and target variable (y)
X = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]  # Features based on regional sales
y = df['Global_Sales'] > 1  # Binary target: True if global sales > 1 million, False otherwise



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize models
logreg = LogisticRegression(max_iter=1000)
dt_classifier = DecisionTreeClassifier(random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)



# Train and evaluate Logistic Regression model
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression - Classification Report:")
print(classification_report(y_test, y_pred_logreg))
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print()



# Train and evaluate Decision Tree Classifier model
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
print("Decision Trees - Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print()



# Train and evaluate Random Forest Classifier model
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print("Random Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
