# ✈️ AeroWeather — Airport Flight Delay Predictor & ML Simulator

An interactive machine learning web app that simulates how weather and operational
conditions drive flight delays. Built with Gradio, scikit-learn, and NumPy, it walks
users through the full ML pipeline: dataset generation → model training → live prediction.

---

## Features

| Tab | What it does |
|-----|-------------|
| **Step 1 – Data Generator** | Synthesize up to 2 000 flight records with 13 weather & operational inputs |
| **Step 2 – Train & Analyze** | Train a Decision Tree (classifier) and Neural Network (regressor) and plot results |
| **Step 3 – Live Predictor** | Enter any weather scenario and get an instant delay prediction from the trained models |
| **Step 4 – ML Academy** | Explanations of ML theory, feature definitions, and curated learning resources |

### Dataset attributes generated (13 inputs + 2 targets)

**Core weather**
- `WindSpeed_Knots` — headwind / tailwind
- `Visibility_Miles` — fog / mist reduction
- `Temp_F` — heat / freezing effects
- `Precip_Inches` — rain or snow accumulation

**Extended atmospheric**
- `Humidity_Pct` — elevated fog and icing risk
- `Pressure_inHg` — barometric pressure (low = storms)
- `CloudCeiling_Ft` — height of cloud base; below ~500 ft forces IFR approaches
- `Crosswind_Knots` — perpendicular wind component versus runway heading

**Operational & scheduling**
- `DepartureHour` — rush-hour cascade effect
- `Season` — winter ice events / summer convective storms
- `FlightDistance_Miles` — en-route weather exposure
- `AirportCongestion` — gate / taxiway backlog (0 = low, 1 = high)
- `RunwayCondition` — derived from temp + precip (0 = Dry, 1 = Wet, 2 = Icy)

**Targets**
- `Delayed` — binary classification label (0 = On-Time, 1 = Delayed)
- `DelayMinutes` — regression target (continuous minutes of delay)

---

## Requirements

- Python **3.10** or later
- pip

---

## Installation

```bash
# 1. Clone or download this repository
git clone https://github.com/your-username/AeroWeather.git
cd AeroWeather

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
python app.py
```

Gradio will start a local web server and print a URL such as:

```
Running on local URL:  http://127.0.0.1:7860
```

Open that URL in your browser. To share the app publicly (useful for demos), launch with:

```python
demo.launch(share=True)
```

---

## Workflow — follow the three steps in order

1. **Generate a dataset** (Tab 1)
   - Adjust the number of flight records (100 – 2 000).
   - Set the *Weather Severity Factor* (0.5 = mild, 2.0 = severe).
   - Set the *Airport Congestion Level* (0.0 = empty, 1.0 = gridlocked).
   - Click **Generate Dataset** — a preview of the first 5 rows appears.

2. **Train the models** (Tab 2)
   - Tune the Decision Tree depth and Neural Network hidden-layer size.
   - Click **Train Models & Plot Results** to see accuracy, R² score, and visualizations.

3. **Test a custom scenario** (Tab 3)
   - Drag all 13 input sliders to your chosen weather conditions.
   - Click **Evaluate Flight Status** to get an instant prediction.

---

## Project Structure

```
AeroWeather/
├── app.py            # Main application (data generation, ML, Gradio UI)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Tech Stack

| Library | Role |
|---------|------|
| [Gradio](https://www.gradio.app) | Interactive web UI |
| [scikit-learn](https://scikit-learn.org) | Decision Tree & Neural Network models |
| [NumPy](https://numpy.org) | Synthetic data generation |
| [pandas](https://pandas.pydata.org) | DataFrame handling |
| [Matplotlib](https://matplotlib.org) | Tree and regression plots |
| [Pillow](https://python-pillow.org) | Image rendering for Gradio |

---

## Learning Resources

- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Kaggle](https://www.kaggle.com) — real flight/weather datasets and competitions
- [Coursera: ML Specialization (Andrew Ng)](https://www.coursera.org/specializations/machine-learning-introduction)
- [Fast.ai](https://www.fast.ai) — code-first deep learning course
