import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

# Step 1: Load dataset
file_path = 'videogamesales/vgsales.csv'
df = pd.read_csv(file_path)

# Step 2: Drop rows with any null values
df = df.dropna()  # Remove null values from the dataset

# Step 3: Select features and target variables for each region
X = df[['Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']]
y_na = df['NA_Sales']
y_eu = df['EU_Sales']
y_jp = df['JP_Sales']
y_global = df['Global_Sales']

# Step 4: One-hot encode categorical variables
categorical_features = ['Platform', 'Genre', 'Publisher']
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]))
X_encoded.columns = encoder.get_feature_names_out(categorical_features)
X_encoded.index = X.index
X = X.drop(categorical_features, axis=1)
X = pd.concat([X, X_encoded], axis=1)

# Step 5: Ensure the lengths match
assert len(X) == len(y_na) == len(y_eu) == len(y_jp) == len(y_global), "Mismatch between feature and target lengths."

# Function to train and save model
def train_and_save_model(X, y, model_filename):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'{model_filename} - Root Mean Squared Error: {rmse}')
    return model

# Step 6: Train models for each region
model_na = train_and_save_model(X, y_na, 'model_na_sales.pkl')  # Train model for NA sales
model_eu = train_and_save_model(X, y_eu, 'model_eu_sales.pkl')  # Train model for EU sales
model_jp = train_and_save_model(X, y_jp, 'model_jp_sales.pkl')  # Train model for JP sales
model_global = train_and_save_model(X, y_global, 'model_global_sales.pkl')  # Train model for global sales

# Function to predict regional sales for new data points
def predict_regional_sales(new_data, region):
    new_data_encoded = pd.DataFrame(encoder.transform(new_data[categorical_features]))
    new_data_encoded.columns = encoder.get_feature_names_out(categorical_features)
    new_data_encoded.index = new_data.index
    new_data = new_data.drop(categorical_features, axis=1)
    new_data = pd.concat([new_data, new_data_encoded], axis=1)
    
    if region == 'NA':
        model = joblib.load('model_na_sales.pkl')
    elif region == 'EU':
        model = joblib.load('model_eu_sales.pkl')
    elif region == 'JP':
        model = joblib.load('model_jp_sales.pkl')
    elif region == 'Global':
        model = joblib.load('model_global_sales.pkl')
    else:
        raise ValueError('Invalid region specified')
    
    predictions = model.predict(new_data)
    return predictions

# Step 7: Prepare new data for future years
future_years = pd.DataFrame({
    'Platform': ['PS4'] * 5,
    'Year': [2024, 2025, 2026, 2027, 2028],
    'Genre': ['Action'] * 5,
    'Publisher': ['Sony'] * 5,
    'Global_Sales': [3.2] * 5
})

# Step 8: Predict sales for each region for future years
predicted_na_sales = predict_regional_sales(future_years, 'NA')
predicted_eu_sales = predict_regional_sales(future_years, 'EU')
predicted_jp_sales = predict_regional_sales(future_years, 'JP')
predicted_global_sales = predict_regional_sales(future_years, 'Global')

# Step 9: Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    'Year': future_years['Year'],
    'NA_Sales': predicted_na_sales,
    'EU_Sales': predicted_eu_sales,
    'JP_Sales': predicted_jp_sales,
    'Global_Sales': predicted_global_sales
})

# Step 10: Include existing data from 2010 to 2023 for plotting
existing_data = df[(df['Year'] >= 2010) & (df['Year'] <= 2023)][['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales']]
combined_df = pd.concat([existing_data, predictions_df])

# Step 11: Plot the predictions and existing data on bar graphs (2010-2028)
def plot_yearwise_sales(region, title, color):
    yearwise_sales = combined_df.groupby('Year')[region].sum().loc[2010:2028]
    plt.figure(figsize=(12, 6))
    yearwise_sales.plot(kind='bar', color=color)
    for i, v in enumerate(yearwise_sales):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Total Sales (in millions)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

plot_yearwise_sales('NA_Sales', 'Yearly Sales in North America (2010 - 2028)', 'green')
plot_yearwise_sales('EU_Sales', 'Yearly Sales in Europe (2010 - 2028)', 'orange')
plot_yearwise_sales('JP_Sales', 'Yearly Sales in Japan (2010 - 2028)', 'skyblue')
plot_yearwise_sales('Global_Sales', 'Yearly Sales Globally (2010 - 2028)', 'turquoise')
