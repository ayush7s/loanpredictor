import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import Prediction, Base, engine, SessionLocal
import joblib
import numpy as np
from dotenv import load_dotenv

# ── Load environment variables ─────────────────────────────────────
load_dotenv()

# ── Initialize Flask App ───────────────────────────────────────────
app = Flask(__name__)

# ── Enable CORS ────────────────────────────────────────────────────
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Create DB Tables ───────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── Load BOTH models ───────────────────────────────────────────────
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")

# ── Root Route ─────────────────────────────────────────────────────
@app.route("/")
def root():
    return jsonify({"message": "LoanIQ API is running!"})

# ── Health Check ───────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": ["rf", "lr"]})

# ── Prediction Endpoint ────────────────────────────────────────────
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # ── Required fields ─────────────────────────────────────────
        required = [
            "Gender", "Married", "Dependents", "Education",
            "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
            "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]

        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # ── Model selection ─────────────────────────────────────────
        model_type = data.get("model_type", "rf")

        if model_type == "rf":
            model = rf_model
        elif model_type == "lr":
            model = lr_model
        else:
            return jsonify({"error": "Invalid model type"}), 400

        # ── Feature Engineering (IMPORTANT) ─────────────────────────
        applicant_income = float(data["ApplicantIncome"])
        coapplicant_income = float(data["CoapplicantIncome"])
        loan_amount = float(data["LoanAmount"])
        loan_term = float(data["Loan_Amount_Term"])

        total_income = applicant_income + coapplicant_income
        income_loan_ratio = total_income / (loan_amount + 1)
        emi = loan_amount / (loan_term + 1)

        # ── Prepare features (MATCH TRAINING ORDER) ─────────────────
        features = np.array([[ 
            float(data["Gender"]),
            float(data["Married"]),
            float(data["Dependents"]),
            float(data["Education"]),
            float(data["Self_Employed"]),
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            float(data["Credit_History"]),
            float(data["Property_Area"]),
            total_income,
            income_loan_ratio,
            emi
        ]])

        # ── Prediction ──────────────────────────────────────────────
        prediction = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            confidence = round(float(model.predict_proba(features).max()) * 100, 2)
        else:
            confidence = 100.0

        result = "Approved ✅" if prediction == 1 else "Rejected ❌"

        # ── Save to Database ────────────────────────────────────────
        db = SessionLocal()
        try:
            record = Prediction(
                gender=str(data["Gender"]),
                married=str(data["Married"]),
                dependents=str(data["Dependents"]),
                education=str(data["Education"]),
                self_employed=str(data["Self_Employed"]),
                applicant_income=applicant_income,
                coapplicant_income=coapplicant_income,
                loan_amount=loan_amount,
                loan_amount_term=loan_term,
                credit_history=float(data["Credit_History"]),
                property_area=str(data["Property_Area"]),
                result=result,
                confidence=confidence
            )
            db.add(record)
            db.commit()
        except Exception as db_error:
            print("DB ERROR:", db_error)
        finally:
            db.close()

        return jsonify({
            "result": result,
            "confidence": confidence,
            "model_used": model_type
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ── History Endpoint ───────────────────────────────────────────────
@app.route("/history")
def history():
    db = SessionLocal()
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