

# Medical Prediction System

A Flask-based web application that predicts whether a patient requires dialysis based on various medical parameters using a machine learning model.

## Project Overview

This application provides a user-friendly interface for medical professionals to input patient data and receive predictions about dialysis requirements. The system uses a pre-trained machine learning model (central_model.pkl) to analyze 17 different medical features and provide predictions with confidence scores.

## Prerequisites

- Python 3.7 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd medical-prediction-system
```

2. Create and activate a virtual environment:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install flask numpy joblib scikit-learn
```

4. Ensure the model file `central_model.pkl` is in the project root directory.

## Running the Application

1. Make sure you're in the project directory with the virtual environment activated.

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## Project Structure

```
medical-prediction-system/
├── app.py                  # Main Flask application
├── central_model.pkl       # Trained machine learning model
├── templates/
│   └── index.html          # Frontend HTML template
└── README.md               # This file
```

## Features Used for Prediction

The model uses the following 18 medical features:

1. Age of the patient
2. Blood pressure (mm/Hg)
3. Albumin in urine
4. Sugar in urine
5. Random blood glucose level (mg/dl)
6. Body Mass Index (BMI)
7. Physical activity level
8. Duration of diabetes mellitus (years)
9. Duration of hypertension (years)
10. Cystatin C level
11. C-reactive protein (CRP) level
12. Interleukin-6 (IL-6) level
13. Red blood cells in urine
14. Pus cells in urine
15. Pus cell clumps in urine (binary: 0 or 1)
16. Bacteria in urine (binary: 0 or 1)
17. Pedal edema (binary: 0 or 1)

## API Endpoints

- `GET /`: Renders the home page with the prediction form.
- `POST /predict`: Processes the form data and returns a prediction.

## Prediction Output

The application provides one of two possible predictions:
- "No Dialysis Required"
- "Dialysis Required"

Each prediction includes a confidence percentage indicating the model's certainty.

## Troubleshooting

If you encounter a "Model file not found" error:
1. Ensure the `central_model.pkl` file is in the same directory as `app.py`.
2. If the model is in a different location, update the `MODEL_PATH` variable in `app.py`.

If you encounter any other errors:
1. Check that all required packages are installed.
2. Verify that the input values match the expected format and range.
3. Check the console output for detailed error messages.

## Model Information

The model used in this application is a pre-trained classifier that has been saved using joblib. It predicts the likelihood of a patient requiring dialysis based on the input medical parameters.

## License

This project is licensed under the MIT License.
