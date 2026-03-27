# 🌍 Quakex: Earthquake Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

> A machine learning-powered web application that predicts earthquake magnitude and classifies severity from seismic parameters with interactive visualizations and educational insights.

---

## Features

- **Magnitude Prediction** – Predicts earthquake magnitude using a trained LightGBM model
- **Severity Classification** – Categorizes predictions as Minor, Moderate, Major, or Extreme
- **Feature Importance Chart** – Visualizes which parameters most influence the prediction
- **Depth vs Magnitude Curve** – Explores how depth affects predicted magnitude
- **Global Geographic Map** – Plots the earthquake location on an interactive world map
- **Magnitude Gauge** – Displays predicted magnitude on a Richter scale gauge
- **Educational Insights** – Richter scale reference, depth classification, and safety tips

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Custom CSS |
| ML Model | LightGBM (via `joblib`) |
| Data Processing | Pandas, NumPy |
| Visualizations | Plotly Express, Plotly Graph Objects |
| Language | Python 3.8+ |

---

## Project Structure

```
quakex/
├── app.py                          # Main Streamlit application
├── lightgbm_earthquake_model.pkl   # Trained LightGBM model
├── requirements.txt                # Dependencies
├── notebook/
│   └── model_training.ipynb        # Model training notebook
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/quakex.git
cd quakex
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## Requirements

```
streamlit
pandas
numpy
joblib
plotly
lightgbm
scikit-learn
```

---

## Model Details

| Property | Detail |
|----------|--------|
| Algorithm | LightGBM Regressor |
| Target | Earthquake Magnitude |
| Input Features | Latitude, Longitude, Depth, Year, Month, Day, Hour, Weekday |
| Severity Levels | Minor (< 4.0), Moderate (4.0–5.5), Major (5.5–7.0), Extreme (≥ 7.0) |

The model was trained on historical seismic data. Features were engineered from geographic coordinates and temporal parameters to capture spatial and time-based seismic patterns.

---

## Input Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| Latitude | Geographic latitude | -90° to 90° |
| Longitude | Geographic longitude | -180° to 180° |
| Depth (km) | Depth below Earth's surface | 0 – 500 km |
| Year | Year of event | 1900 – 2100 |
| Month | Month of event | 1 – 12 |
| Day | Day of event | 1 – 31 |
| Hour (UTC) | Hour of event | 0 – 23 |
| Weekday | Day of the week | Mon – Sun |

---

## Richter Scale Reference

| Magnitude | Classification | Effects |
|-----------|---------------|---------|
| < 2.0 | Micro | Not felt, detected by instruments only |
| 2.0 – 3.9 | Minor | Often felt, rarely causes damage |
| 4.0 – 4.9 | Light | Noticeable shaking, minimal damage |
| 5.0 – 5.9 | Moderate | Can damage poorly constructed buildings |
| 6.0 – 6.9 | Strong | Destructive in populated areas |
| 7.0 – 7.9 | Major | Serious damage over large areas |
| 8.0+ | Great | Devastating damage over vast areas |

---

## Author

**Arpita Prasad**
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/arpita-prasad)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/arpita-prasad)
