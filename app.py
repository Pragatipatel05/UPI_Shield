import os
import io
import base64
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('dark_background')  # Dark theme for plots

from flask import Flask, render_template, request, flash, redirect, url_for
import joblib

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Paths for saved artifacts (exported from notebook)
MODEL_PATH = "models/project_model1.h5"
SCALER_PATH = "models/scaler.joblib"
SCORES_PATH = "scores.json"
PLOT_PATH = "accuracy_plot.png"

# Globals
model = None
scaler = None
last_scores = None
last_head_html = None
last_train_plot = None


def load_artifacts():
    """Load model, scaler, scores, and plot from disk."""
    global model, scaler, last_scores, last_train_plot

    # Load model + scaler
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"Error loading scaler: {e}")

    # Load scores
    if os.path.exists(SCORES_PATH):
        try:
            with open(SCORES_PATH, "r") as f:
                last_scores = json.load(f)
        except Exception as e:
            print(f"Error loading scores: {e}")

    # Load accuracy plot and convert to base64
    if os.path.exists(PLOT_PATH):
        try:
            with open(PLOT_PATH, "rb") as f:
                last_train_plot = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error loading plot: {e}")


def generate_dataset_visualizations(df):
    """Generate comprehensive visualizations for the uploaded dataset"""
    visualizations = {}
    
    try:
        # Set up the color palette for dark theme
        colors = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        fraud_colors = ['#ff6b6b','#00d4ff']
        
        # 1. Fraud Distribution Pie Chart
        if 'fraud_risk' in df.columns:
            fig, ax = plt.subplots(figsize=(4, 4), facecolor='#1a1a1a')
            fraud_counts = df['fraud_risk'].value_counts()
            labels = ['Fraud Transactions', 'Valid Transactions']
            ax.pie(fraud_counts.values, labels=labels, autopct='%1.1f%%', 
                   colors=fraud_colors, startangle=90)
            ax.set_title('Transaction Distribution: Fraud vs Valid', fontsize=8, fontweight='bold', color='white')
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a1a', dpi=150)
            buf.seek(0)
            visualizations['fraud_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        
        # 2. Transaction Amount Distribution by Fraud Status
        if 'trans_amount' in df.columns and 'fraud_risk' in df.columns:
            fig, ax = plt.subplots(figsize=(4, 3), facecolor='#1a1a1a')
            valid_amounts = df[df['fraud_risk'] == 0]['trans_amount']
            fraud_amounts = df[df['fraud_risk'] == 1]['trans_amount']
            
            bins = np.linspace(0, min(df['trans_amount'].max(), 10000), 30)
            ax.hist(valid_amounts, bins=bins, alpha=0.7, label='Valid', color=colors[0], edgecolor='white')
            ax.hist(fraud_amounts, bins=bins, alpha=0.7, label='Fraud', color=colors[1], edgecolor='white')
            
            ax.set_xlabel('Transaction Amount (₹)', fontsize = 8, color='white')
            ax.set_ylabel('Frequency', fontsize=8, color='white')
            ax.set_title('Transaction Amount Distribution by Fraud Status', fontsize=8, fontweight='bold', color='white')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a1a', dpi=150)
            buf.seek(0)
            visualizations['amount_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        
        # 3. Transaction Hour Patterns
        if 'trans_hour' in df.columns and 'fraud_risk' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1a1a1a')
            hour_fraud = df.groupby(['trans_hour', 'fraud_risk']).size().unstack(fill_value=0)
            
            x_pos = np.arange(24)
            width = 0.35
            
            if 0 in hour_fraud.columns:
                ax.bar(x_pos - width/2, hour_fraud[0], width, label='Valid', color=colors[0], alpha=0.8)
            if 1 in hour_fraud.columns:
                ax.bar(x_pos + width/2, hour_fraud[1], width, label='Fraud', color=colors[1], alpha=0.8)
            
            ax.set_xlabel('Hour of Day', fontsize=8, color='white')
            ax.set_ylabel('Number of Transactions', fontsize=8, color='white')
            ax.set_title('Transaction Patterns by Hour of Day', fontsize=8, fontweight='bold', color='white')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a1a', dpi=150)
            buf.seek(0)
            visualizations['hour_patterns'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        
        # 4. Age Distribution by Fraud Status
        if 'age' in df.columns and 'fraud_risk' in df.columns:
            fig, ax = plt.subplots(figsize=(4, 3), facecolor='#1a1a1a')
            valid_ages = df[df['fraud_risk'] == 0]['age']
            fraud_ages = df[df['fraud_risk'] == 1]['age']
            
            bins = np.linspace(df['age'].min(), df['age'].max(), 20)
            ax.hist(valid_ages, bins=bins, alpha=0.7, label='Valid', color=colors[0], edgecolor='white')
            ax.hist(fraud_ages, bins=bins, alpha=0.7, label='Fraud', color=colors[1], edgecolor='white')
            
            ax.set_xlabel('Age', fontsize=8, color='white')
            ax.set_ylabel('Frequency', fontsize=8, color='white')
            ax.set_title('Age Distribution by Fraud Status', fontsize=8, fontweight='bold', color='white')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a1a', dpi=150)
            buf.seek(0)
            visualizations['age_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        
        # 5. State-wise Fraud Analysis (Top 10 states)
        if 'state' in df.columns and 'fraud_risk' in df.columns:
            fig, ax = plt.subplots(figsize=(4, 3), facecolor='#1a1a1a')
            state_fraud = df.groupby('state')['fraud_risk'].agg(['count', 'sum']).reset_index()
            state_fraud['fraud_rate'] = (state_fraud['sum'] / state_fraud['count'] * 100).round(2)
            state_fraud = state_fraud.sort_values('count', ascending=False).head(10)
            
            bars = ax.bar(range(len(state_fraud)), state_fraud['fraud_rate'], color=colors[2], alpha=0.8)
            ax.set_xlabel('State Code', fontsize=8, color='white')
            ax.set_ylabel('Fraud Rate (%)', fontsize=8, color='white')
            ax.set_title('Fraud Rate by State (Top 10 States by Transaction Volume)', fontsize=8, fontweight='bold', color='white')
            ax.set_xticks(range(len(state_fraud)))
            ax.set_xticklabels(state_fraud['state'], rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}%', ha='center', va='bottom', color='white', fontsize=6)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a1a', dpi=150)
            buf.seek(0)
            visualizations['state_analysis'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        
        # 6. Monthly Transaction Trends
        if 'trans_month' in df.columns and 'fraud_risk' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#1a1a1a')
            monthly_stats = df.groupby('trans_month')['fraud_risk'].agg(['count', 'sum']).reset_index()
            monthly_stats['fraud_rate'] = (monthly_stats['sum'] / monthly_stats['count'] * 100).round(2)
            
            ax2 = ax.twinx()
            bars = ax.bar(monthly_stats['trans_month'], monthly_stats['count'], 
                         color=colors[3], alpha=0.6, label='Total Transactions')
            line = ax2.plot(monthly_stats['trans_month'], monthly_stats['fraud_rate'], 
                           color=colors[1], marker='o', linewidth=3, markersize=8, label='Fraud Rate')
            
            ax.set_xlabel('Month', fontsize=8, color='white')
            ax.set_ylabel('Total Transactions', fontsize=8, color='white')
            ax2.set_ylabel('Fraud Rate (%)', fontsize=8, color='white')
            ax.set_title('Monthly Transaction Volume and Fraud Rate', fontsize=8, fontweight='bold', color='white')
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(monthly_stats['trans_month'])
            ax.set_xticklabels([months[i-1] for i in monthly_stats['trans_month']])
            
            ax.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a1a', dpi=150)
            buf.seek(0)
            visualizations['monthly_trends'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        
        # 7. Dataset Statistics Summary
        stats_summary = {
            'total_transactions': len(df),
            'fraud_transactions': df['fraud_risk'].sum() if 'fraud_risk' in df.columns else 0,
            'fraud_rate': (df['fraud_risk'].sum() / len(df) * 100) if 'fraud_risk' in df.columns else 0,
            'avg_amount': df['trans_amount'].mean() if 'trans_amount' in df.columns else 0,
            'unique_users': df['upi_number'].nunique() if 'upi_number' in df.columns else 0,
            'date_range': {
                'start_month': df['trans_month'].min() if 'trans_month' in df.columns else None,
                'end_month': df['trans_month'].max() if 'trans_month' in df.columns else None,
                'start_year': df['trans_year'].min() if 'trans_year' in df.columns else None,
                'end_year': df['trans_year'].max() if 'trans_year' in df.columns else None
            }
        }
        
        visualizations['stats_summary'] = stats_summary
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        visualizations['error'] = str(e)
    
    return visualizations


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/train", methods=["POST"])
def train():
    """Show results using notebook-trained model and scores with dataset visualizations."""
    global last_scores, last_head_html, last_train_plot

    # Load artifacts once
    load_artifacts()

    # Dataset analysis and visualization
    file = request.files.get("datasetfile")
    visualizations = {}
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            last_head_html = df.head().to_html(classes="table table-dark table-striped", index=False)
            
            # Generate comprehensive visualizations
            visualizations = generate_dataset_visualizations(df)
            
            flash("Dataset uploaded and analyzed successfully!", "success")
        except Exception as e:
            flash(f"Error reading dataset: {str(e)}", "error")
            return redirect(url_for('upload_page'))
    else:
        flash("Please upload a valid CSV file.", "error")
        return redirect(url_for('upload_page'))

    if not last_scores:
        return render_template("result.html", 
                             title="Training Failed",
                             OUTPUT="No training results found. Please ensure the model has been trained first.",
                             error=True)

    # Pick best model
    best_name = max(last_scores, key=last_scores.get)
    OUTPUT = f"Best Model: {best_name} with {last_scores[best_name]:.2f}% accuracy"

    return render_template(
        "result.html",
        title="Dataset Analysis Results",
        OUTPUT=OUTPUT,
        GRAPH=last_train_plot,
        SCORES=last_scores,
        TABLE_HTML=last_head_html,
        VISUALIZATIONS=visualizations,
        success=True
    )


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Predict single transaction using notebook-trained model."""
    global model, scaler

    if model is None or scaler is None:
        load_artifacts()
        if model is None or scaler is None:
            return render_template("result.html", 
                                 title="Prediction Failed",
                                 OUTPUT="Model not found. Please ensure the model has been trained in the notebook first.",
                                 error=True)

    # Collect 10 inputs
    try:
        features = []
        for i in range(1, 11):
            value = request.form.get(f"f{i}")
            if not value:
                raise ValueError(f"Feature {i} is missing")
            features.append(float(value))
    except (ValueError, TypeError) as e:
        return render_template("result.html", 
                             title="Prediction Failed",
                             OUTPUT=f"Invalid inputs: {str(e)}. Please enter 10 valid numeric values.",
                             error=True)

    try:
        X_test = scaler.transform([features])
        pred = int(model.predict(X_test)[0])
    except Exception as e:
        return render_template("result.html", 
                             title="Prediction Failed",
                             OUTPUT=f"Model prediction error: {str(e)}",
                             error=True)

    # Prediction probability (if available)
    graph_b64 = None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X_test)[0]
            fig, ax = plt.subplots(figsize=(4, 4))
            colors = ['#28a745', '#dc3545']  # Green for valid, red for fraud
            bars = ax.bar(["Valid Transaction", "Fraud Transaction"], [p[0], p[1]], color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for bar, prob in zip(bars, [p[0], p[1]]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor='white')
            buf.seek(0)
            graph_b64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
        except Exception as e:
            print(f"Error generating probability plot: {e}")

    text = "🚨 FRAUD TRANSACTION DETECTED" if pred == 1 else "✅ VALID TRANSACTION"
    is_fraud = pred == 1

    return render_template("result.html", 
                         title="Fraud Detection Result", 
                         OUTPUT=text, 
                         GRAPH=graph_b64,
                         fraud_detected=is_fraud,
                         success=True)


@app.errorhandler(404)
def not_found_error(error):
    return render_template('result.html', 
                         title="Page Not Found", 
                         OUTPUT="The page you're looking for doesn't exist.",
                         error=True), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('result.html', 
                         title="Internal Server Error", 
                         OUTPUT="An internal server error occurred. Please try again later.",
                         error=True), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
