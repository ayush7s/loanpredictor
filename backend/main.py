from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy.orm import Session
from database import get_db, Prediction, Base, engine
import joblib
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

app = Flask(__name__)
CORS(app)  # allows frontend to call this API

# Load model once when server starts
model = joblib.load("model.pkl")

@app.route("/")
def root():
    return jsonify({"message": "LoanIQ API is running!"})
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "loaded"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validate required fields
    required = [
        "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"
    ]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Prepare features in exact same order as training
    features = np.array([[
        float(data["Gender"]),
        float(data["Married"]),
        float(data["Dependents"]),
        float(data["Education"]),
        float(data["Self_Employed"]),
        float(data["ApplicantIncome"]),
        float(data["CoapplicantIncome"]),
        float(data["LoanAmount"]),
        float(data["Loan_Amount_Term"]),
        float(data["Credit_History"]),
        float(data["Property_Area"])
    ]])

    # Predict
    prediction = model.predict(features)[0]
    confidence = round(float(model.predict_proba(features).max()) * 100, 2)
    result     = "Approved ✅" if prediction == 1 else "Rejected ❌"

    # Save to Supabase
    db = next(get_db())
    try:
        record = Prediction(
            gender             = str(data["Gender"]),
            married            = str(data["Married"]),
            dependents         = str(data["Dependents"]),
            education          = str(data["Education"]),
            self_employed      = str(data["Self_Employed"]),
            applicant_income   = float(data["ApplicantIncome"]),
            coapplicant_income = float(data["CoapplicantIncome"]),
            loan_amount        = float(data["LoanAmount"]),
            loan_amount_term   = float(data["Loan_Amount_Term"]),
            credit_history     = float(data["Credit_History"]),
            property_area      = str(data["Property_Area"]),
            result             = result,
            confidence         = confidence
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

    return jsonify({"result": result, "confidence": confidence})

@app.route("/history")
def history():
    db = next(get_db())
    try:
        records = db.query(Prediction).order_by(
            Prediction.created_at.desc()
        ).limit(10).all()

        return jsonify([
            {
                "id":               r.id,
                "result":           r.result,
                "confidence":       r.confidence,
                "applicant_income": r.applicant_income,
                "loan_amount":      r.loan_amount,
                "credit_history":   r.credit_history,
                "created_at":       str(r.created_at)
            }
            for r in records
        ])
    finally:
        db.close()
    

if __name__ == "__main__":
    app.run(debug=True)