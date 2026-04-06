# 📊 PICO-SR: AI-Powered Automated Systematic Review & Meta-Analysis

**PICO-SR** is an end-to-end AI pipeline that automates the gold-standard research method: the Systematic Review. By combining **Groq-accelerated LLMs** for rapid screening and **Robust Statistics** for evidence pooling, it transforms 18 months of manual research into a near-instant automated workflow.

---

## 🚀 The Problem

Systematic reviews are the backbone of evidence-based medicine, yet they are currently:

- **Slow:** Taking 6–18 months to complete manually  
- **Manual:** Researchers spend hundreds of hours screening abstracts and extracting data from dense PDF tables  
- **Static:** Reviews are outdated the moment a new study is published, leading to "evidence lag"  

---

## ✨ Our Solution: The PICO-SR Pipeline

We built a five-stage automated pipeline that fetches, screens, extracts, and analyzes academic evidence in real-time.

---

### 🔹 1. Multi-Source Ingestion

The system takes a natural language research question and expands it into technical boolean search strings. It queries **PubMed** and **OpenAlex** in parallel, deduplicating results via DOI and fuzzy title matching.

---

### 🔹 2. Intelligent Screening

Using **Llama 3.1 70B via Groq**, the system screens papers against strict **PICO** (Population, Intervention, Comparator, Outcome) criteria.

- **Human-in-the-Loop (HITL):** Papers with low confidence scores are flagged for manual verification in a dedicated review queue, ensuring clinical safety.

---

### 🔹 3. Vision-Enhanced Extraction

To solve the "Table Trap," we use **pdfplumber** to preserve the visual structure of clinical tables. The LLM extracts 12 structured fields, including effect sizes and 95% Confidence Intervals.

---

### 🔹 4. Robust Statistical Meta-Analysis

The system standardizes all data into **Cohen’s d** and uses the **DerSimonian-Laird random-effects model** to calculate a pooled estimate.

It generates:

- **Forest Plots:** A visual summary of all included evidence  
- **Heterogeneity (I²):** A measure of how much studies disagree  
- **Automated Conclusion:** A plain-English recommendation for clinicians  

---

### 🔹 5. Living Review Mode

Our "Living Review" background scheduler polls **PubMed RSS feeds** every 24 hours. When new evidence is found, the pipeline re-runs automatically and generates a **Diff Report** showing how the pooled conclusion has shifted.

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit (Python) for an interactive dashboard  
- **Backend:** FastAPI (Python) for REST APIs  
- **LLM:** Llama 3.1 70B (via Groq API)  
- **Statistics:** SciPy, Statsmodels, Matplotlib  
- **Database:** SQLite  

---

## 📦 Quick Start

Got it 👍 — here is your **FULL exact copy-paste block (no breaks, properly closed, ready for README)**:

### 1️⃣ Environment Setup
````markdown


Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
LLM_PROVIDER=groq
PICO_SR_API=http://127.0.0.1:8000
````

---

### 2️⃣ Install Dependencies

```bash
pip install fastapi uvicorn pdfplumber scipy statsmodels matplotlib streamlit groq rapidfuzz sqlalchemy requests
```

---

### 3️⃣ Run the System

Start the backend:

```bash
uvicorn api.main:app --reload --port 8000
```

Then start the frontend:

```bash
streamlit run ui/app.py
```

---

## 🎯 Recommended Demo Case

### Research Query:

```
(physical exercise OR aerobic training) AND depression AND "randomized controlled trial"
```

### PICO Criteria:

```
Include RCTs on exercise vs control for depression reporting quantitative outcomes.
```

---

## 📈 Output Highlights

* High heterogeneity (I²) handling
* Significant pooled effect detection
* Automated conclusion: Supports Intervention

```
```
