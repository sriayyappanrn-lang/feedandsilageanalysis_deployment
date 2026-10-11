app.pyimport pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

yield_path = '/content/extracted_data/global_cattle_milk_yield_prediction_dataset.csv'

if os.path.exists(yield_path):
    # Load and optimize dataset footprint
    df_yield = pd.read_csv(yield_path).sample(n=30000, random_state=42) # Sample for robust training speed in Colab
    print("Loaded yield dataset:", df_yield.shape)
    
    # Select nutrition/feed quality features and production targets
    features = ['Feed_Quantity_kg', 'Water_Intake_L', 'Age_Months', 'Weight_kg', 'Days_in_Milk', 'Milking_Interval_hrs', 'Body_Condition_Score']
    target = 'Milk_Yield_L'
    
    X = df_yield[features]
    y = df_yield[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize numerical features
    reg_scaler = StandardScaler()
    X_train_scaled = reg_scaler.fit_transform(X_train)
    X_test_scaled = reg_scaler.transform(X_test)
    
    # Train Regression model
    reg_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    reg_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = reg_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Regression Mean Absolute Error: {mae:.4f} Liters")
    print(f"Regression R² Score: {r2:.4f}")
    
    # Save Artifacts
    joblib.dump(reg_model, '/content/feed_regression_model.joblib')
    joblib.dump(reg_scaler, '/content/feed_scaler.joblib')
    joblib.dump(features, '/content/feed_features.joblib')
    print("Feed regression artifacts successfully saved!")
    
    # Plot Feature Importance and Save
    plt.figure(figsize=(10, 6))
    importances = reg_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sns.barplot(x=importances[indices], y=np.array(features)[indices], palette='viridis')
    plt.title('Feature Importances in Feed & Silage Impact Regression')
    plt.xlabel('Importance Value')
    plt.tight_layout()
    plt.savefig('/content/feature_importance.png')
    plt.close()
else:
    print("Milk yield prediction dataset not found.")
