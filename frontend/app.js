// ── Config ────────────────────────────────────────────────────────
const API_URL = "https://loanpredictor-60xb.onrender.com";// Change to your Render URL when deployed

// ── State ─────────────────────────────────────────────────────────
const formData = {
  Gender: 1,
  Married: 1,
  Dependents: 0,
  Education: 0,
  Self_Employed: 0,
  Credit_History: 1,
  Property_Area: 1
};

// ── Toggle Buttons ────────────────────────────────────────────────
document.querySelectorAll(".toggle-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const field = btn.dataset.field;
    const value = parseFloat(btn.dataset.value);

    // Deactivate siblings
    btn.closest(".toggle-group").querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    formData[field] = value;
  });
});

// ── Predict ───────────────────────────────────────────────────────
async function predict() {
  const btn = document.getElementById("predictBtn");

  // Collect numeric inputs
  const applicantIncome   = parseFloat(document.getElementById("ApplicantIncome").value);
  const coapplicantIncome = parseFloat(document.getElementById("CoapplicantIncome").value) || 0;
  const loanAmount        = parseFloat(document.getElementById("LoanAmount").value);
  const loanTerm          = parseFloat(document.getElementById("Loan_Amount_Term").value);

  // Validate
  if (!applicantIncome || applicantIncome <= 0) return showError("Please enter a valid Applicant Income.");
  if (!loanAmount || loanAmount <= 0)           return showError("Please enter a valid Loan Amount.");

  // Build payload
const payload = {
  ...formData,
  model_type: document.getElementById("modelType").value, // ✅ ADD THIS LINE
  ApplicantIncome: applicantIncome,
  CoapplicantIncome: coapplicantIncome,
  LoanAmount: loanAmount,
  Loan_Amount_Term: loanTerm
};

  // Loading state
  btn.classList.add("loading");
  btn.querySelector(".btn-text").textContent = "Predicting...";
  btn.querySelector(".btn-arrow").textContent = "⟳";

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    showResult(data);
    loadHistory();

  } catch (err) {
    showError("Could not connect to the API. Make sure your backend is running.");
    console.error(err);
  } finally {
    btn.classList.remove("loading");
    btn.querySelector(".btn-text").textContent = "Predict My Approval";
    btn.querySelector(".btn-arrow").textContent = "→";
  }
}

// ── Show Result ───────────────────────────────────────────────────
function showResult(data) {
  const card       = document.getElementById("resultCard");
  const icon       = document.getElementById("resultIcon");
  const text       = document.getElementById("resultText");
  const fill       = document.getElementById("confidenceFill");
  const pct        = document.getElementById("confidencePct");

  const isApproved = data.result.includes("Approved");

  card.className = `result-card visible ${isApproved ? "approved" : "rejected"}`;
  icon.textContent = isApproved ? "✅" : "❌";
  text.textContent = isApproved ? "Loan Approved!" : "Loan Rejected";
  const modelUsed = document.getElementById("modelUsed");
if (modelUsed && data.model_used) {
  let modelName = "Unknown";

  if (data.model_used === "rf") modelName = "Random Forest";
  else if (data.model_used === "lr") modelName = "Logistic Regression";

  modelUsed.textContent = "Model Used: " + modelName;
}
  pct.textContent  = `${data.confidence}%`;

  // Animate bar
  setTimeout(() => { fill.style.width = `${data.confidence}%`; }, 100);

  // Scroll to result
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Show Error ────────────────────────────────────────────────────
function showError(msg) {
  const card = document.getElementById("resultCard");
  card.className = "result-card visible rejected";
  document.getElementById("resultIcon").textContent = "⚠️";
  document.getElementById("resultText").textContent = msg;
  document.getElementById("confidenceFill").style.width = "0%";
  document.getElementById("confidencePct").textContent = "";
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Load History ──────────────────────────────────────────────────
async function loadHistory() {
  const tbody = document.getElementById("historyBody");

  try {
    const res  = await fetch(`${API_URL}/history`);
    if (!res.ok) throw new Error("Failed to fetch history");
    const data = await res.json();

    if (!data || data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="empty-row">No predictions yet. Make your first prediction above!</td></tr>`;
      return;
    }

    tbody.innerHTML = data.map((row, i) => {
      const isApproved = row.result && row.result.includes("Approved");
      const time       = row.created_at ? new Date(row.created_at).toLocaleString() : "—";
      const income     = row.applicant_income ? `₹${Number(row.applicant_income).toLocaleString()}` : "—";
      const loan       = row.loan_amount ? `₹${row.loan_amount}` : "—";
      const credit     = row.credit_history == 1 ? "Good" : "Poor";

      return `
        <tr>
          <td>${i + 1}</td>
          <td><span class="badge ${isApproved ? "approved" : "rejected"}">${isApproved ? "Approved" : "Rejected"}</span></td>
          <td>${row.confidence ? row.confidence + "%" : "—"}</td>
          <td>${income}</td>
          <td>${loan}</td>
          <td>${credit}</td>
          <td>${time}</td>
        </tr>
      `;
    }).join("");

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-row">Could not load history. Is your backend running?</td></tr>`;
    console.error(err);
  }
}

// ── Load history on page load ─────────────────────────────────────
window.addEventListener("DOMContentLoaded", loadHistory);