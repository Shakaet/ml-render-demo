# # app.py

# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np

# # Load the trained model
# model_path = 'central_model.pkl'
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract data from form
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
    
#     # Make prediction
#     prediction = model.predict(final_features)
#     output = 'Placed' if prediction[0] == 1 else 'Not Placed'

#     return render_template('index.html', prediction_text='Prediction: {}'.format(output))

# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, request, render_template
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Load the trained model
# model = joblib.load('central_model.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     features = [float(x) for x in request.form.values()]
#     final_features = [np.array(features)]
#     prediction = model.predict(final_features)
#     output = 'Yes' if prediction[0] == 1 else 'No'
#     return render_template('index.html', prediction_text=f'Prediction: {output}')

# if __name__ == "__main__":
#     app.run(debug=True)

# app.py

# from flask import Flask, request, render_template
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Load the trained model
# model = joblib.load('central_model.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract numeric features from form and convert to float
#         features = [float(x) for x in request.form.values()]
#         final_features = [np.array(features)]

#           # Print the features to see what the model gets
#         print("Features received by model:", final_features)
        
#         # Get prediction probability
#         prob = model.predict_proba(final_features)
#         print("Prediction probability:", prob)
#         probability = prob[0][1] * 100  # Probability of class 1 ("Placed")
        
#         # Determine output text
#         if probability > 50:
#             output = f"Placed ({probability:.2f}%)"
#         else:
#             output = f"Not Placed ({100-probability:.2f}%)"

#         return render_template('index.html', prediction_text=f'Prediction: {output}')
    
#     except Exception as e:
#         return render_template('index.html', prediction_text=f"Error: {str(e)}")

# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, request, render_template
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Load the trained model
# model = joblib.load('central_model.pkl')

# # Class mapping: adjust according to your dataset
# class_mapping = {
#     0: "Placed",
#     1: "Not Placed",
#     2: "Maybe",
#     3: "Risk",
#     4: "Other"
# }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract features from form and convert to float
#         features = [float(x) for x in request.form.values()]
#         final_features = [np.array(features)]
        
#         # Get prediction probabilities
#         prob = model.predict_proba(final_features)  # shape = [1, n_classes]
        
#         # Get class with highest probability
#         pred_class = np.argmax(prob)
#         pred_prob = prob[0][pred_class] * 100  # convert to percentage
        
#         # Map to human-readable label
#         output = f"{class_mapping.get(pred_class, 'Unknown')} ({pred_prob:.2f}%)"
        
#         # Debugging logs (optional)
#         print(f"Features received by model: {final_features}")
#         print(f"Prediction probability: {prob}")
#         print(f"Predicted class: {pred_class}, Output: {output}")
        
#         return render_template('index.html', prediction_text=f'Prediction: {output}')
    
#     except Exception as e:
#         return render_template('index.html', prediction_text=f"Error: {e}")

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Define the model path
MODEL_PATH = 'central_model.pkl'

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please ensure the model is in the correct location.")

# Load the trained model
model = joblib.load(MODEL_PATH)

# Define class mapping for the prediction results
class_mapping = {
    0: "No Dialysis Required",
    1: "Dialysis Required"
}

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process the form data and make a prediction"""
    try:
        # Extract form data in the same order as the model expects
        # Note: The order must match the order of features used during training
        features = [
            float(request.form['age']),
            float(request.form['bp']),
            float(request.form['albumin']),
            float(request.form['sugar']),
            float(request.form['rbg']),
            float(request.form['bmi']),
            float(request.form['activity']),
            float(request.form['diabetes_duration']),
            float(request.form['hypertension_duration']),
            float(request.form['cystatin_c']),
            float(request.form['crp']),
            float(request.form['il6']),
            float(request.form['rbc']),
            float(request.form['pus_cells']),
            float(request.form['pus_clumps']),
            float(request.form['bacteria']),
            float(request.form['pedal_edema'])
        ]

          # Print raw features from frontend
        print("✅ Raw features from frontend:", features)
        
        # Convert to numpy array and reshape for prediction
        final_features = [np.array(features)]

        print("🔹 Processed features sent to model (numpy array):", final_features)
        
        # Make prediction
        prediction_proba = model.predict_proba(final_features)
        prediction_class = np.argmax(prediction_proba, axis=1)[0]
        prediction_probability = np.max(prediction_proba) * 100
        
        # Format the output
        output_label = class_mapping.get(prediction_class, "Unknown Class")
        output_text = f'Prediction: {output_label} (Confidence: {prediction_probability:.2f}%)'

           # Print prediction details
        print("🎯 Model predicted class:", prediction_class)
        print("📊 Model probabilities:", prediction_proba)
        print("💡 Final output text:", output_text)
        
        # Render the result back to the user
        return render_template('index.html', prediction_text=output_text)
    
    except Exception as e:
        # Handle any errors
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)


