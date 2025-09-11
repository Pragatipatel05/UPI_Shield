import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
import os

def create_models():
    """Train models and save them along with the scaler"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load the dataset
    try:
        df = pd.read_csv('upi_fraud_dataset.csv')
        print(f"Dataset loaded: {df.shape}")
    except FileNotFoundError:
        print("Creating sample dataset...")
        # Create a sample dataset if file doesn't exist
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'trans_hour': np.random.randint(0, 24, n_samples),
            'trans_day': np.random.randint(1, 32, n_samples),
            'trans_month': np.random.randint(1, 13, n_samples),
            'trans_year': np.random.choice([2022, 2023], n_samples),
            'category': np.random.randint(0, 15, n_samples),
            'upi_number': np.random.randint(9000000000, 9999999999, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'trans_amount': np.random.exponential(1000, n_samples),
            'state': np.random.randint(1, 36, n_samples),
            'zip': np.random.randint(100000, 999999, n_samples),
        }
        
        # Create fraud labels with some logic
        fraud_prob = (
            (data['trans_hour'] < 6) * 0.3 +  # Late night transactions
            (data['trans_amount'] > 5000) * 0.4 +  # Large amounts
            (data['age'] < 25) * 0.2 +  # Young users
            np.random.random(n_samples) * 0.3
        )
        data['fraud_risk'] = (fraud_prob > 0.6).astype(int)
        
        df = pd.DataFrame(data)
        df.to_csv('upi_fraud_dataset.csv', index=False)
        print(f"Sample dataset created: {df.shape}")
    
    # Prepare features and target
    X = df.iloc[:, :-1].values  # All columns except last (fraud_risk)
    y = df.iloc[:, -1].values   # Last column (fraud_risk)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    print("✓ Scaler saved")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    # Train models and calculate scores
    scores = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred) * 100
        scores[name] = round(accuracy, 2)
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
        
        print(f"✓ {name}: {accuracy:.2f}% accuracy")
    
    # Save the best model
    joblib.dump(best_model, 'models/project_model1.h5')
    print(f"✓ Best model saved: {max(scores, key=scores.get)}")
    
    # Save scores
    with open('scores.json', 'w') as f:
        json.dump(scores, f, indent=4)
    print("✓ Scores saved")
    
    # Print summary
    print(f"\nModel Training Complete!")
    print(f"Best Model: {max(scores, key=scores.get)} with {max(scores.values()):.2f}% accuracy")
    print(f"Files created:")
    print(f"  - models/scaler.joblib")
    print(f"  - models/project_model1.h5") 
    print(f"  - scores.json")
    
    return scores

if __name__ == "__main__":
    create_models()