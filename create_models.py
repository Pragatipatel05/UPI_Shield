import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
import os

from xgboost import XGBClassifier

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, InputLayer, Dropout
from tensorflow.keras.utils import to_categorical


def create_models():
    """Train models and save them along with the scaler"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load dataset or create a sample one
    try:
        df = pd.read_csv('upi_fraud_dataset.csv')
        print(f"Dataset loaded: {df.shape}")
    except FileNotFoundError:
        print("Creating sample dataset...")
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
        
        fraud_prob = (
            (data['trans_hour'] < 6) * 0.3 +
            (data['trans_amount'] > 5000) * 0.4 +
            (data['age'] < 25) * 0.2 +
            np.random.random(n_samples) * 0.3
        )
        data['fraud_risk'] = (fraud_prob > 0.6).astype(int)
        
        df = pd.DataFrame(data)
        df.to_csv('upi_fraud_dataset.csv', index=False)
        print(f"Sample dataset created: {df.shape}")
    
    # Features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Scaler saved")
    
    # Traditional models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }

    scores = {}
    best_model = None
    best_score = 0
    best_model_name = None

    # Train traditional models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred) * 100
        scores[name] = round(accuracy, 2)

        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_model_name = name

        print(f">> {name}: {accuracy:.2f}% accuracy")

    # ---------------- Hybrid Model 1: RF + CNN ---------------- #
    print("Training Hybrid Model (Random Forest + CNN)...")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    rf_train_probs = rf.predict_proba(X_train_scaled)
    rf_test_probs = rf.predict_proba(X_test_scaled)

    X_train_hybrid = np.hstack([X_train_scaled, rf_train_probs])
    X_test_hybrid = np.hstack([X_test_scaled, rf_test_probs])

    X_train_cnn = X_train_hybrid.reshape((X_train_hybrid.shape[0], X_train_hybrid.shape[1], 1))
    X_test_cnn = X_test_hybrid.reshape((X_test_hybrid.shape[0], X_test_hybrid.shape[1], 1))

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    cnn_model = Sequential([
        InputLayer(input_shape=(X_train_hybrid.shape[1], 1)),
        Conv1D(32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train_cat, epochs=10, batch_size=32, verbose=0)

    _, accuracy = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    accuracy *= 100
    scores['Hybrid RF + CNN'] = float(round(accuracy, 2))

    if accuracy > best_score:
        best_score = accuracy
        best_model = cnn_model
        best_model_name = "Hybrid RF + CNN"

    print(f">> Hybrid RF + CNN: {accuracy:.2f}% accuracy")

    # ---------------- Hybrid Model 2: XGB + DNN ---------------- #
    print("Training Hybrid Model (XGBoost + DNN)...")

    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                        random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)

    xgb_train_probs = xgb.predict_proba(X_train_scaled)
    xgb_test_probs = xgb.predict_proba(X_test_scaled)

    X_train_hybrid2 = np.hstack([X_train_scaled, xgb_train_probs])
    X_test_hybrid2 = np.hstack([X_test_scaled, xgb_test_probs])

    dnn_model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train_hybrid2.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])

    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_train_hybrid2, to_categorical(y_train), epochs=15, batch_size=32, verbose=0)

    _, accuracy = dnn_model.evaluate(X_test_hybrid2, to_categorical(y_test), verbose=0)
    accuracy *= 100
    scores['Hybrid XGB + DNN'] = float(round(accuracy, 2))

    if accuracy > best_score:
        best_score = accuracy
        best_model = dnn_model
        best_model_name = "Hybrid XGB + DNN"

    print(f">> Hybrid XGB + DNN: {accuracy:.2f}% accuracy")

    # ---------------- Save the best model ---------------- #
    if isinstance(best_model, Sequential):
        best_model.save("models/project_model.keras")
        print(f"✓ Best deep learning model saved: {best_model_name}")
    else:
        joblib.dump(best_model, "models/project_model.pkl")
        print(f"✓ Best sklearn/XGBoost model saved: {best_model_name}")

    # Save scores
    with open('scores.json', 'w') as f:
        json.dump(scores, f, indent=4)
    print("✓ Scores saved")

    # Print summary
    print(f"\nModel Training Complete!")
    print(f"Best Model: {best_model_name} with {best_score:.2f}% accuracy")
    print("Files created:")
    print("  - models/scaler.joblib")
    if isinstance(best_model, Sequential):
        print("  - models/project_model.keras")
    else:
        print("  - models/project_model.pkl")
    print("  - scores.json")

    return scores


if __name__ == "__main__":
    create_models()















# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# import joblib
# import json
# import os

# from xgboost import XGBClassifier


# # TensorFlow/Keras imports
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, Flatten, InputLayer, Dropout
# from tensorflow.keras.utils import to_categorical

# def create_models():
#     """Train models and save them along with the scaler"""
    
#     # Create models directory if it doesn't exist
#     os.makedirs('models', exist_ok=True)
    
#     # Load the dataset
#     try:
#         df = pd.read_csv('upi_fraud_dataset.csv')
#         print(f"Dataset loaded: {df.shape}")
#     except FileNotFoundError:
#         print("Creating sample dataset...")
#         np.random.seed(42)
#         n_samples = 1000
        
#         data = {
#             'trans_hour': np.random.randint(0, 24, n_samples),
#             'trans_day': np.random.randint(1, 32, n_samples),
#             'trans_month': np.random.randint(1, 13, n_samples),
#             'trans_year': np.random.choice([2022, 2023], n_samples),
#             'category': np.random.randint(0, 15, n_samples),
#             'upi_number': np.random.randint(9000000000, 9999999999, n_samples),
#             'age': np.random.randint(18, 80, n_samples),
#             'trans_amount': np.random.exponential(1000, n_samples),
#             'state': np.random.randint(1, 36, n_samples),
#             'zip': np.random.randint(100000, 999999, n_samples),
#         }
        
#         fraud_prob = (
#             (data['trans_hour'] < 6) * 0.3 +
#             (data['trans_amount'] > 5000) * 0.4 +
#             (data['age'] < 25) * 0.2 +
#             np.random.random(n_samples) * 0.3
#         )
#         data['fraud_risk'] = (fraud_prob > 0.6).astype(int)
        
#         df = pd.DataFrame(data)
#         df.to_csv('upi_fraud_dataset.csv', index=False)
#         print(f"Sample dataset created: {df.shape}")
    
#     # Prepare features and target
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values
    
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Create and fit scaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Save the scaler
#     joblib.dump(scaler, 'models/scaler.joblib')
#     print("Scaler saved")
    
#     # Define traditional models (excluding Gradient Boosting)
#     models = {
#         'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#         'Logistic Regression': LogisticRegression(random_state=42),
#         'SVM': SVC(probability=True, random_state=42),
#         'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
#     }

#     scores = {}
#     best_model = None
#     best_score = 0

#     # Train traditional models
#     for name, model in models.items():
#         print(f"Training {name}...")
#         model.fit(X_train_scaled, y_train)
#         y_pred = model.predict(X_test_scaled)
#         accuracy = accuracy_score(y_test, y_pred) * 100
#         scores[name] = round(accuracy, 2)

#         if accuracy > best_score:
#             best_score = accuracy
#             best_model = model

#         print(f">> {name}: {accuracy:.2f}% accuracy")

#     # --- HYBRID MODEL: Random Forest + CNN ---
#     print("Training Hybrid Model (Random Forest + CNN)...")

#     # Train Random Forest
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(X_train_scaled, y_train)

#     # Get class probabilities
#     rf_train_probs = rf.predict_proba(X_train_scaled)
#     rf_test_probs = rf.predict_proba(X_test_scaled)

#     # Combine RF probabilities with scaled input
#     X_train_hybrid = np.hstack([X_train_scaled, rf_train_probs])
#     X_test_hybrid = np.hstack([X_test_scaled, rf_test_probs])

#     # Reshape for 1D CNN input
#     X_train_cnn = X_train_hybrid.reshape((X_train_hybrid.shape[0], X_train_hybrid.shape[1], 1))
#     X_test_cnn = X_test_hybrid.reshape((X_test_hybrid.shape[0], X_test_hybrid.shape[1], 1))

#     # One-hot encode targets
#     y_train_cat = to_categorical(y_train)
#     y_test_cat = to_categorical(y_test)

#     # Define 1D CNN model
#     cnn_model = Sequential([
#         InputLayer(input_shape=(X_train_hybrid.shape[1], 1)),
#         Conv1D(32, kernel_size=3, activation='relu'),
#         Flatten(),
#         Dense(64, activation='relu'),
#         Dense(2, activation='softmax')
#     ])

#     cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # Train CNN
#     cnn_model.fit(X_train_cnn, y_train_cat, epochs=10, batch_size=32, verbose=0)

#     # Evaluate CNN
#     loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
#     accuracy *= 100

#     # Fix: convert to standard float before saving
#     scores['Hybrid RF + CNN'] = float(round(accuracy, 2))

#     if accuracy > best_score:
#         best_score = accuracy
#         best_model = cnn_model

#     print(f">> Hybrid RF + CNN: {accuracy:.2f}% accuracy")




#     # ---------------- Hybrid Model 2: XGBoost + DNN ---------------- #
#     print("Training Hybrid Model (XGBoost + DNN)...")

#     # Train XGBoost
#     xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
#     xgb.fit(X_train_scaled, y_train)

#     # Get class probabilities
#     xgb_train_probs = xgb.predict_proba(X_train_scaled)
#     xgb_test_probs = xgb.predict_proba(X_test_scaled)

#     # Combine XGB probabilities with scaled features
#     X_train_hybrid2 = np.hstack([X_train_scaled, xgb_train_probs])
#     X_test_hybrid2 = np.hstack([X_test_scaled, xgb_test_probs])

#     # Define Deep Neural Network
#     dnn_model = Sequential([
#         Dense(256, activation='relu', input_shape=(X_train_hybrid2.shape[1],)),
#         Dropout(0.3),
#         Dense(128, activation='relu'),
#         Dropout(0.2),
#         Dense(64, activation='relu'),
#         Dense(2, activation='softmax')
#     ])

#     dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # One-hot encode targets
#     y_train_cat = to_categorical(y_train)
#     y_test_cat = to_categorical(y_test)

#     # Train DNN
#     dnn_model.fit(X_train_hybrid2, y_train_cat, epochs=15, batch_size=32, verbose=0)

#     # Evaluate
#     loss, accuracy = dnn_model.evaluate(X_test_hybrid2, y_test_cat, verbose=0)
#     accuracy *= 100

#     scores['Hybrid XGB + DNN'] = float(round(accuracy, 2))

#     if accuracy > best_score:
#         best_score = accuracy
#         best_model = dnn_model

#     print(f">> Hybrid XGB + DNN: {accuracy:.2f}% accuracy")


#    # # Save the best model
#    # if isinstance(best_model, Sequential):
#    #     best_model.save('models/project_model1.keras')
#    #     print(f"Best model (CNN) saved: Hybrid RF + CNN")
#    # else:
#    #     joblib.dump(best_model, 'models/project_model1.pkl')
#    #     print(f"Best model saved: {max(scores, key=scores.get)}")

#     # Save scores
#     with open('scores.json', 'w') as f:
#         json.dump(scores, f, indent=4)
#     print("✓ Scores saved")

#     # Print summary
#     print(f"\nModel Training Complete!")
#     print(f"Best Model: {max(scores, key=scores.get)} with {max(scores.values()):.2f}% accuracy")
#     print(f"Files created:")
#     print(f"  - models/scaler.joblib")
#     if isinstance(best_model, Sequential):
#         print(f"  - models/project_model1.h5")
#     else:
#         print(f"  - models/project_model1.keras")
#     print(f"  - scores.json")

#     return scores

# if __name__ == "__main__":
#     create_models()

