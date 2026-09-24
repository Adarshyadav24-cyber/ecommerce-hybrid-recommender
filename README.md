# 🛍️ ShopSense AI — E-Commerce Hybrid Recommender

A machine-learning based e-commerce recommendation system that combines **Collaborative Filtering** and **Content-Based Filtering** to generate personalized product recommendations.

The project includes data generation/preprocessing, recommendation models, evaluation, alpha tuning, visualization, and an interactive **Streamlit dashboard**.

---

## 🚀 Project Features

- Personalized recommendations for individual users
- Collaborative Filtering
- Content-Based Filtering
- Hybrid Recommendation Model
- Adjustable collaborative/content weighting using **Alpha**
- User interaction history
- Recommendation explanations
- Recommendation score breakdown
- Precision@10, Recall@10, MAP@10 and NDCG@10 evaluation
- Model comparison
- Alpha tuning analysis
- Interactive Streamlit dashboard
- Visual reports and charts

---

## 🧠 Recommendation Approach

### 1. Collaborative Filtering

Collaborative filtering uses historical user-item interactions.

It identifies products that are related to products the user has previously interacted with.

### 2. Content-Based Filtering

Content-based filtering uses product information such as:

- Product title
- Category
- Brand
- Description

TF-IDF and cosine similarity are used to calculate product-content similarity.

### 3. Hybrid Recommendation

The system combines both approaches using an adjustable weight:

```text
Hybrid Score =
Alpha × Collaborative Score
+
(1 - Alpha) × Content Score
```

Where:

- `Alpha = 0` → Content-Based
- `Alpha = 1` → Collaborative
- `0 < Alpha < 1` → Hybrid

---

## 📊 Dataset

The current generated dataset contains approximately:

- **1,000 users**
- **500 products**
- **23,637 interaction events**
- **18,521 training interactions**
- **5,116 test interactions**

The preprocessing pipeline also creates a sparse user-item matrix.

---

## 📈 Evaluation Metrics

The recommender is evaluated using:

### Precision@10

Measures how many of the top 10 recommended products are relevant.

### Recall@10

Measures how many relevant products are successfully retrieved in the top 10.

### MAP@10

Measures ranking quality across the top 10 recommendations.

### NDCG@10

Measures ranking quality while giving higher importance to recommendations appearing earlier in the list.

---

## 🏗️ Project Architecture

```text
                ┌──────────────────────┐
                │   Raw User/Product   │
                │      Data            │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Data Loading &       │
                │ Preprocessing        │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Train/Test Split     │
                │ + Sparse Matrix      │
                └──────────┬───────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   ┌──────────────────┐       ┌──────────────────┐
   │ Collaborative    │       │ Content-Based    │
   │ Filtering        │       │ Filtering        │
   └────────┬─────────┘       └────────┬─────────┘
            │                          │
            └────────────┬─────────────┘
                         ▼
              ┌────────────────────┐
              │ Hybrid Recommender │
              │   Alpha Weighting  │
              └──────────┬─────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ Evaluation &       │
              │ Visualization      │
              └──────────┬─────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ Streamlit          │
              │ Dashboard          │
              └────────────────────┘
```

---

## 📁 Project Structure

```text
ecommerce-hybrid-recommender/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   └── alpha_tuning.csv
│
└── src/
    ├── data_loader.py
    ├── baseline_recommenders.py
    ├── hybrid_recommender.py
    └── visualization.py
```

---

## ⚙️ Installation

Use **Python 3.11**.

Open PowerShell inside the project directory.

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If multiple Python versions are installed, use:

```powershell
& "C:\Users\ay351\AppData\Local\Programs\Python\Python311\python.exe" -m pip install -r requirements.txt
```

---

## ▶️ Run the Data Pipeline

```powershell
& "C:\Users\ay351\AppData\Local\Programs\Python\Python311\python.exe" src\data_loader.py
```

This generates the raw and processed datasets.

---

## ▶️ Run Recommendation Models

Example:

```powershell
& "C:\Users\ay351\AppData\Local\Programs\Python\Python311\python.exe" src\baseline_recommenders.py
```

Run the hybrid recommender:

```powershell
& "C:\Users\ay351\AppData\Local\Programs\Python\Python311\python.exe" src\hybrid_recommender.py
```

---

## 📊 Generate Visualizations

```powershell
& "C:\Users\ay351\AppData\Local\Programs\Python\Python311\python.exe" src\visualization.py
```

Generated figures are stored in:

```text
reports/figures/
```

---

## 🌐 Run Streamlit Dashboard

From the project root:

```powershell
& "C:\Users\ay351\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
```

The dashboard provides:

- User selection
- Recommendation generation
- User history
- Recommendation explanation
- Score breakdown
- Model comparison
- Alpha analysis
- Dataset summary
- System architecture

---

## 🎛️ Alpha Tuning

Alpha controls the contribution of the collaborative model.

Example:

```text
Alpha = 0.00
Content = 100%
Collaborative = 0%

Alpha = 0.50
Content = 50%
Collaborative = 50%

Alpha = 1.00
Content = 0%
Collaborative = 100%
```

The project stores alpha tuning results in:

```text
reports/alpha_tuning.csv
```

---

## 📋 Model Comparison

Model evaluation results are stored in:

```text
reports/model_comparison.csv
```

The dashboard visualizes:

- Precision@10
- Recall@10
- MAP@10
- NDCG@10

---

## 🖥️ Dashboard

The Streamlit application provides an interactive interface where a user can:

1. Select a user
2. View previous interactions
3. Choose the collaborative/content weighting
4. Generate personalized recommendations
5. Inspect recommendation scores
6. Open the explanation for each recommendation
7. Compare model performance

---

## 🔍 Recommendation Explanation

For each recommended product, the dashboard can show:

- Content Score
- Collaborative Score
- Hybrid Score
- Content similarity explanation
- Collaborative similarity explanation
- Recommendation mode

This makes the recommendation system easier to demonstrate and understand.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming |
| Pandas | Data processing |
| NumPy | Numerical computation |
| SciPy | Sparse matrices |
| Scikit-learn | TF-IDF and cosine similarity |
| Matplotlib | Visualization |
| Streamlit | Web dashboard |
| CSV / JSON / NPZ | Data storage |

---

## 🔮 Future Scope

Possible future improvements include:

- Real-time product recommendations
- Login/authentication
- Product images
- Product prices
- Ratings and reviews
- Deep-learning recommendation models
- Neural collaborative filtering
- More advanced ranking models
- Cold-start handling
- Online deployment
- Recommendation feedback tracking
- A/B testing
- Real e-commerce API integration

---

## 🎓 Academic / Placement Use

This project demonstrates practical knowledge of:

- Python
- Data preprocessing
- Machine Learning
- Recommendation Systems
- Collaborative Filtering
- Content-Based Filtering
- Sparse Matrices
- Cosine Similarity
- Model Evaluation
- Data Visualization
- Streamlit
- Software project organization

---

## 👨‍💻 Project Status

**Status:** Functional ML recommendation system with interactive Streamlit dashboard.

The project currently includes data generation, preprocessing, recommendation models, evaluation, alpha analysis, visualization, recommendation explanations, and user-history features.
