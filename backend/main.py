import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy.orm import Session
from database import get_db, Prediction, Base, engine
import joblib
import numpy as np
from dotenv import load_dotenv

# ── Load env variables ─────────────────────────────────────────────
load_dotenv()

# ── Initialize Flask App ───────────────────────────────────────────
app = Flask(__name__)

# Enable CORS (VERY IMPORTANT for Vercel ↔ Render)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Create DB Tables ───────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── Load ML Model ──────────────────────────────────────────────────
model = joblib.load("model.pkl")

# ── Root Route ─────────────────────────────────────────────────────
@app.route("/")
def root():
    return jsonify({"message": "LoanIQ API is running!"})

# ── Health Check ───────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "loaded"})

# ── Prediction Endpoint ────────────────────────────────────────────
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle preflight request (CORS)
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Required fields
        required = [
            "Gender", "Married", "Dependents", "Education",
            "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
            "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]

        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Prepare features
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

        # Prediction
        prediction = model.predict(features)[0]
        confidence = round(float(model.predict_proba(features).max()) * 100, 2)
        result = "Approved ✅" if prediction == 1 else "Rejected ❌"

        # Save to database
        db = next(get_db())
        try:
            record = Prediction(
                gender=str(data["Gender"]),
                married=str(data["Married"]),
                dependents=str(data["Dependents"]),
                education=str(data["Education"]),
                self_employed=str(data["Self_Employed"]),
                applicant_income=float(data["ApplicantIncome"]),
                coapplicant_income=float(data["CoapplicantIncome"]),
                loan_amount=float(data["LoanAmount"]),
                loan_amount_term=float(data["Loan_Amount_Term"]),
                credit_history=float(data["Credit_History"]),
                property_area=str(data["Property_Area"]),
                result=result,
                confidence=confidence
            )
            db.add(record)
            db.commit()
        finally:
            db.close()

        return jsonify({
            "result": result,
            "confidence": confidence
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ── History Endpoint ───────────────────────────────────────────────
@app.route("/history")
def history():
    db = next(get_db())
    try:
        records = db.query(Prediction).order_by(
            Prediction.created_at.desc()
        ).limit(10).all()

        return jsonify([
            {
                "id": r.id,
                "result": r.result,
                "confidence": r.confidence,
                "applicant_income": r.applicant_income,
                "loan_amount": r.loan_amount,
                "credit_history": r.credit_history,
                "created_at": str(r.created_at)
            }
            for r in records
        ])
    finally:
        db.close()


# ── Run Server ─────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)