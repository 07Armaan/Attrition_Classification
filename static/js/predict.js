// ==========================================================================
// Prediction form: validation, progress, API call, result rendering
// ==========================================================================
// >>> CONFIGURE THIS: point to your running FastAPI backend <
const API_BASE_URL = (() => {
  // If the page is served from the same host as the API (e.g. FastAPI
  // serving these static files itself), same-origin "" works automatically.
  // Otherwise, change this to e.g. "http://localhost:8000"
  if (window.location.protocol === 'file:') {
    return 'http://localhost:8000';
  }
  return ''; // same-origin
})();
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictForm');
  if (!form) return; // not on the predict page

  const submitBtn = document.getElementById('submitBtn');
  const progressFill = document.getElementById('progressFill');
  const progressLabel = document.getElementById('progressLabel');
  const fillSampleBtn = document.getElementById('fillSampleBtn');
  const resetBtn = document.getElementById('resetBtn');
  const predictAgainBtn = document.getElementById('predictAgainBtn');
  const retryBtn = document.getElementById('retryBtn');
  const apiEndpointLabel = document.getElementById('apiEndpointLabel');

  const stateIdle = document.getElementById('resultIdle');
  const stateLoading = document.getElementById('resultLoading');
  const stateDone = document.getElementById('resultDone');
  const stateError = document.getElementById('resultError');

  if (apiEndpointLabel) apiEndpointLabel.textContent = PREDICT_ENDPOINT;

  const allFields = Array.from(form.querySelectorAll('input[required], select[required]'));
  const totalFields = allFields.length;

  /* ---------------- Progress tracking ---------------- */
  function updateProgress() {
    const filled = allFields.filter(f => f.value !== '' && f.value !== null).length;
    const pct = totalFields ? Math.round((filled / totalFields) * 100) : 0;
    if (progressFill) progressFill.style.width = pct + '%';
    if (progressLabel) progressLabel.textContent = `${filled} / ${totalFields} fields completed`;
  }
  allFields.forEach(f => {
    f.addEventListener('input', updateProgress);
    f.addEventListener('change', updateProgress);
  });
  updateProgress();

  /* ---------------- Validation ---------------- */
  function validateField(field) {
    const wrapper = field.closest('.field');
    if (!wrapper) return true;
    let valid = field.checkValidity();
    if (field.type === 'number' && field.value !== '') {
      const num = Number(field.value);
      if (Number.isNaN(num) || num < 0) valid = false;
    }
    wrapper.classList.toggle('invalid', !valid);
    return valid;
  }

  allFields.forEach(f => {
    f.addEventListener('blur', () => validateField(f));
    f.addEventListener('change', () => validateField(f));
  });

  function validateAll() {
    let allValid = true;
    allFields.forEach(f => {
      if (!validateField(f)) allValid = false;
    });
    return allValid;
  }

  /* ---------------- Result state helpers ---------------- */
  function showState(state) {
    [stateIdle, stateLoading, stateDone, stateError].forEach(el => el && el.classList.remove('active'));
    if (state) state.classList.add('active');
  }
  function setIdle() {
    stateIdle.style.display = 'flex';
    showState(null);
  }
  function setLoading() {
    stateIdle.style.display = 'none';
    showState(stateLoading);
  }
  function setDone() {
    stateIdle.style.display = 'none';
    showState(stateDone);
  }
  function setError(message) {
    stateIdle.style.display = 'none';
    showState(stateError);
    const msgEl = document.getElementById('errorMessage');
    if (msgEl && message) msgEl.textContent = message;
  }

  /* ---------------- Toast ---------------- */
  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toastMsg');
  function showToast(message, success = false) {
    if (!toast) return;
    toastMsg.textContent = message;
    toast.classList.toggle('success', success);
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3200);
  }

  /* ---------------- Gauge rendering ---------------- */
  const gaugeFill = document.getElementById('gaugeFill');
  const gaugePct = document.getElementById('gaugePct');
  const resultVerdict = document.getElementById('resultVerdict');
  const resultSummary = document.getElementById('resultSummary');
  const resultFactors = document.getElementById('resultFactors');
  const dynamicFactorBars = document.getElementById('dynamicFactorBars');

  const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 70; // r=70

  function renderResult(probability, label, topFactors) {
    // probability: 0..1
    const pct = Math.round(probability * 100);
    const isRisk = (label ? label.toLowerCase().includes('leav') || label.toLowerCase().includes('1') : pct >= 50);

    const offset = GAUGE_CIRCUMFERENCE * (1 - probability);
    gaugeFill.style.strokeDasharray = `${GAUGE_CIRCUMFERENCE}`;
    gaugeFill.style.strokeDashoffset = `${GAUGE_CIRCUMFERENCE}`;
    gaugeFill.style.stroke = isRisk ? 'var(--amber-500)' : 'var(--teal-500)';

    requestAnimationFrame(() => {
      gaugeFill.style.strokeDashoffset = `${offset}`;
    });

    animateNumber(gaugePct, 0, pct, 900, v => `${v}%`);

    resultVerdict.textContent = isRisk ? '● Likely to leave' : '● Likely to stay';
    resultVerdict.className = 'result-verdict ' + (isRisk ? 'risk' : 'safe');

    resultSummary.innerHTML = isRisk
      ? `The model estimates a <strong>${pct}% probability</strong> of attrition for this profile — high enough to warrant a retention conversation.`
      : `The model estimates a <strong>${pct}% probability</strong> of attrition for this profile — this employee currently looks stable.`;

    if (topFactors && Array.isArray(topFactors) && topFactors.length) {
      resultFactors.style.display = 'block';
      dynamicFactorBars.innerHTML = '';
      topFactors.slice(0, 5).forEach(f => {
        const row = document.createElement('div');
        row.className = 'factor-row';
        row.innerHTML = `
          <span class="name">${f.name}</span>
          <div class="factor-track"><div class="factor-fill" style="width:0%"></div></div>
          <span class="pct">${Math.round(f.weight)}%</span>
        `;
        dynamicFactorBars.appendChild(row);
        requestAnimationFrame(() => {
          row.querySelector('.factor-fill').style.width = Math.round(f.weight) + '%';
        });
      });
    } else {
      resultFactors.style.display = 'none';
    }
  }

  function animateNumber(el, from, to, duration, formatFn) {
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const val = Math.round(from + (to - from) * eased);
      el.textContent = formatFn(val);
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ---------------- Build payload ---------------- */
  function buildPayload() {
    const get = id => document.getElementById(id).value;
    return {
      "Age": Number(get('Age')),
      "Gender": get('Gender'),
      "Years at Company": Number(get('YearsAtCompany')),
      "Job Role": get('JobRole'),
      "Monthly Income": Number(get('MonthlyIncome')),
      "Work-Life Balance": get('WorkLifeBalance'),
      "Job Satisfaction": get('JobSatisfaction'),
      "Performance Rating": get('PerformanceRating'),
      "Number of Promotions": Number(get('NumberOfPromotions')),
      "Overtime": get('Overtime'),
      "Distance from Home": Number(get('DistanceFromHome')),
      "Education Level": get('EducationLevel'),
      "Marital Status": get('MaritalStatus'),
      "Number of Dependents": Number(get('NumberOfDependents')),
      "Job Level": get('JobLevel'),
      "Company Size": get('CompanySize'),
      "Company Tenure": Number(get('CompanyTenure')),
      "Remote Work": get('RemoteWork'),
      "Leadership Opportunities": get('LeadershipOpportunities'),
      "Innovation Opportunities": get('InnovationOpportunities'),
      "Company Reputation": get('CompanyReputation'),
      "Employee Recognition": get('EmployeeRecognition')
    };
  }

  /* ---------------- Submit handler ---------------- */
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!validateAll()) {
      showToast('Please fill in all required fields correctly.', false);
      const firstInvalid = form.querySelector('.field.invalid input, .field.invalid select');
      if (firstInvalid) firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }

    const payload = buildPayload();
    submitBtn.disabled = true;
    submitBtn.style.opacity = '0.7';
    setLoading();

    try {
      const response = await fetch(PREDICT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(`Server responded ${response.status}. ${text}`.trim());
      }

      const data = await response.json();

      // Expected response shape:
      // { "prediction": "Stayed" | "Left" (or 0/1), "probability": 0.0-1.0, "top_factors": [{name, weight}] }
      let probability = data.probability;
      if (probability === undefined && data.attrition_probability !== undefined) {
        probability = data.attrition_probability;
      }
      if (probability === undefined) probability = 0.5;
      if (probability > 1) probability = probability / 100;

      const label = data.prediction !== undefined ? String(data.prediction) : (probability >= 0.5 ? 'Left' : 'Stayed');

      renderResult(probability, label, data.top_factors);
      setDone();
      showToast('Prediction received.', true);

    } catch (err) {
      console.error(err);
      setError(`Couldn't reach the prediction API at ${PREDICT_ENDPOINT}. Make sure the FastAPI backend is running, CORS is enabled, and the URL is correct. (${err.message})`);
      showToast('Prediction failed — check the backend connection.', false);
    } finally {
      submitBtn.disabled = false;
      submitBtn.style.opacity = '1';
    }
  });

  /* ---------------- Fill sample data ---------------- */
  const SAMPLE = {
    Age: 22,
    Gender: 'Female',
    YearsAtCompany: 3,
    JobRole: 'Technology',
    MonthlyIncome: 4200,
    WorkLifeBalance: 'Fair',
    JobSatisfaction: 'Low',
    PerformanceRating: 'Average',
    NumberOfPromotions: 0,
    Overtime: 'Yes',
    DistanceFromHome: 28,
    EducationLevel: "Bachelor's Degree",
    MaritalStatus: 'Single',
    NumberOfDependents: 0,
    JobLevel: 'Entry',
    CompanySize: 'Large',
    CompanyTenure: 3,
    RemoteWork: 'No',
    LeadershipOpportunities: 'No',
    InnovationOpportunities: 'No',
    CompanyReputation: 'Fair',
    EmployeeRecognition: 'Low'
  };

  fillSampleBtn?.addEventListener('click', () => {
    Object.entries(SAMPLE).forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (el) {
        el.value = val;
        validateField(el);
      }
    });
    updateProgress();
    showToast('Sample profile filled in.', true);
  });

  /* ---------------- Reset / retry ---------------- */
  function resetForm() {
    form.reset();
    allFields.forEach(f => f.closest('.field')?.classList.remove('invalid'));
    updateProgress();
    setIdle();
  }

  resetBtn?.addEventListener('click', resetForm);
  predictAgainBtn?.addEventListener('click', () => {
    setIdle();
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  retryBtn?.addEventListener('click', () => {
    form.dispatchEvent(new Event('submit', { cancelable: true }));
  });

  // initialize
  setIdle();
});