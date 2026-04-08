import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import Prediction, Base, engine, SessionLocal
import joblib
import numpy as np
from dotenv import load_dotenv

# ── Load env ─────────────────────────────────────────────
load_dotenv()

# ── App init ────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── DB init ─────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── Load models ─────────────────────────────────────────
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")

print("RF:", type(rf_model))
print("LR:", type(lr_model))


# ── Root Route ──────────────────────────────────────────
@app.route("/")
def root():
    return jsonify({"message": "LoanIQ API is running!"})


# ── Prediction Route ────────────────────────────────────
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    try:
        data = request.get_json()

        # ── Model selection ──────────────────────────────
        model_type = data.get("model_type", "rf")

        if model_type == "rf":
            model = rf_model
        elif model_type == "lr":
            model = lr_model
        else:
            return jsonify({"error": "Invalid model type"}), 400

        # ── Extract inputs ───────────────────────────────
        ai = float(data["ApplicantIncome"])
        cai = float(data["CoapplicantIncome"])
        la = float(data["LoanAmount"])
        lt = float(data["Loan_Amount_Term"])

        # ── Feature vector (11 features ONLY) ────────────
        features = np.array([[ 
            float(data["Gender"]),
            float(data["Married"]),
            float(data["Dependents"]),
            float(data["Education"]),
            float(data["Self_Employed"]),
            ai,
            cai,
            la,
            lt,
            float(data["Credit_History"]),
            float(data["Property_Area"])
        ]])

        print("MODEL:", model_type)
        print("FEATURES:", features)

        # ── Prediction ───────────────────────────────────
        prediction = model.predict(features)[0]

        confidence = (
            round(float(model.predict_proba(features).max()) * 100, 2)
            if hasattr(model, "predict_proba") else 100
        )

        result = "Approved ✅" if prediction == 1 else "Rejected ❌"

        # ── Save to DB ───────────────────────────────────
        db = SessionLocal()
        try:
            record = Prediction(
                gender=str(data["Gender"]),
                married=str(data["Married"]),
                dependents=str(data["Dependents"]),
                education=str(data["Education"]),
                self_employed=str(data["Self_Employed"]),
                applicant_income=ai,
                coapplicant_income=cai,
                loan_amount=la,
                loan_amount_term=lt,
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
            "confidence": confidence,
            "model_used": model_type
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ── History Route ───────────────────────────────────────
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


# ── Run Server ──────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)