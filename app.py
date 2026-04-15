from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('machine_model.pkl')

@app.route('/predict_failure', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Input: [Air_Temp, Process_Temp, Rotational_Speed, Torque, Tool_Wear]
        features = np.array(data['features']).reshape(1, -1)
        
        prediction = model.predict(features)
        prob = model.predict_proba(features)
        
        return jsonify({
            'failure_predicted': int(prediction),
            'failure_probability': round(float(prob[0][1]), 4),
            'action': 'MAINTENANCE REQUIRED' if prediction == 1 else 'OPERATIONAL',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
