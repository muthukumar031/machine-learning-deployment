from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# --------------------------------------------------------
# Load model, scaler, and training columns (if available)
# --------------------------------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Optional: training data columns for correct order
try:
    X = pd.read_csv("training_data.csv")
    expected_columns = X.columns.tolist()
except:
    X = None
    expected_columns = None


# --------------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Read JSON input
        data = request.get_json(force=True)

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # 2. Convert dict / list → DataFrame
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid input format'}), 400

        # 3. Reorder & fill missing columns
        if expected_columns is not None:
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_columns]
        else:
            # If no training column file available → expect exactly 8 features
            if input_df.shape[1] != 8:
                return jsonify({'error': 'Input must contain exactly 8 features.'}), 400

        # 4. Scale input data
        scaled_input = scaler.transform(input_df)

        # 5. Predict
        predictions = model.predict(scaled_input)

        # 6. Return prediction result
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --------------------------------------------------------
# Run Flask App
# --------------------------------------------------------
if __name__ == "__main__":    # FIXED
    app.run(debug=True)
