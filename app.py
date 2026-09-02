import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
import io
from PIL import Image
# make `train.py` callable from the UI
import subprocess, sys
import torch
from torch.utils.data import DataLoader
from data import load_csv_time_series, TimeSeriesDataset
from models import LSTMForecaster, CNN1DForecaster
# Global variables to store the generated data and models
data_store = {"df": None, "clf": None, "nn": None, "features": None}

# ==========================================
# FUNCTION 1: GENERATE SAMPLE DATA
# ==========================================
def generate_data(num_flights, weather_severity, congestion_level):
    np.random.seed(42)
    n_samples = int(num_flights)

    # --- Core Weather Features ---
    wind_speed   = np.random.uniform(0, 45, n_samples)
    visibility   = np.random.uniform(0.5, 10, n_samples)
    temp         = np.random.uniform(10, 95, n_samples)
    precip       = np.random.uniform(0, 2.5, n_samples)

    # --- Extended Atmospheric Features ---
    humidity      = np.random.uniform(20, 100, n_samples)           # % relative humidity
    pressure      = np.random.uniform(28.5, 31.0, n_samples)        # inHg barometric pressure
    cloud_ceiling = np.random.uniform(500, 15000, n_samples)        # feet above ground level
    crosswind     = wind_speed * np.random.uniform(0.2, 0.8, n_samples)  # runway crosswind component

    # --- Operational & Scheduling Features ---
    departure_hour  = np.random.randint(0, 24, n_samples)           # 0-23 hour of day
    season          = np.random.randint(1, 5, n_samples)            # 1=Spring 2=Summer 3=Fall 4=Winter
    flight_distance = np.random.uniform(100, 3000, n_samples)       # statute miles

    # --- Airport Congestion (centered around user-defined level) ---
    congestion = np.clip(np.random.normal(congestion_level, 0.15, n_samples), 0, 1)

    # --- Derived: Runway Condition (0=Dry, 1=Wet, 2=Icy) ---
    runway_condition = np.zeros(n_samples, dtype=int)
    runway_condition[precip > 0.1] = 1                               # Wet
    runway_condition[(precip > 0.1) & (temp < 32)] = 2              # Icy

    # --- Composite Delay Score ---
    hour_factor = 1 + 0.3 * np.sin(np.pi * departure_hour / 12)    # peaks at noon / midnight

    delay_score = (
        (wind_speed * 1.2 * weather_severity) +
        ((10 - visibility) * 2.5 * weather_severity) +
        (precip * 12 * weather_severity) +
        (humidity * 0.08 * weather_severity) +                       # high humidity -> fog risk
        ((30.5 - pressure) * 8 * weather_severity) +                # low pressure -> storms
        ((15000 - cloud_ceiling) / 15000 * 10 * weather_severity) + # low ceiling -> IFR approach
        (crosswind * 0.8 * weather_severity) +                      # crosswind limits
        (runway_condition * 8 * weather_severity) +                 # surface condition
        (congestion * 20) +                                         # gate / taxiway backlog
        (hour_factor * 3)                                           # peak-hour congestion
    )

    delayed = (delay_score > (40 / weather_severity)).astype(int)

    delay_minutes = np.where(
        delayed == 1,
        (delay_score * 1.8) + np.random.normal(0, 8, n_samples),
        0
    )
    delay_minutes = np.clip(delay_minutes, 0, 180).astype(int)

    df = pd.DataFrame({
        'WindSpeed_Knots':      np.round(wind_speed, 1),
        'Visibility_Miles':     np.round(visibility, 1),
        'Temp_F':               np.round(temp, 1),
        'Precip_Inches':        np.round(precip, 2),
        'Humidity_Pct':         np.round(humidity, 1),
        'Pressure_inHg':        np.round(pressure, 2),
        'CloudCeiling_Ft':      np.round(cloud_ceiling, 0).astype(int),
        'Crosswind_Knots':      np.round(crosswind, 1),
        'DepartureHour':        departure_hour,
        'Season':               season,
        'FlightDistance_Miles': np.round(flight_distance, 0).astype(int),
        'AirportCongestion':    np.round(congestion, 2),
        'RunwayCondition':      runway_condition,
        'Delayed':              delayed,
        'DelayMinutes':         delay_minutes,
    })

    data_store["df"] = df

    total_delayed = int(df['Delayed'].sum())
    avg_delay     = float(df[df['Delayed'] == 1]['DelayMinutes'].mean()) if total_delayed > 0 else 0
    icy_count     = int((df['RunwayCondition'] == 2).sum())
    wet_count     = int((df['RunwayCondition'] == 1).sum())

    summary = (
        f"✅ Dataset Successfully Generated!\n"
        f"• Total Flight Records: {n_samples}\n"
        f"• Delayed Flights: {total_delayed} ({total_delayed/n_samples*100:.1f}%)\n"
        f"• Average Delay Duration: {avg_delay:.1f} minutes\n"
        f"• Icy Runway Events: {icy_count}  |  Wet Runway Events: {wet_count}\n"
        f"• Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns (13 inputs + 2 targets)"
    )

    return summary, df.head(5)

# ==========================================
# FUNCTION 2: RUN MACHINE LEARNING ANALYSIS
# ==========================================
def run_analysis(tree_depth, nn_neurons):
    if data_store["df"] is None:
        return "⚠️ Please generate sample data first in Tab 1!", None, None
    
    df = data_store["df"]
    X = df[['WindSpeed_Knots', 'Visibility_Miles', 'Temp_F', 'Precip_Inches',
            'Humidity_Pct', 'Pressure_inHg', 'CloudCeiling_Ft', 'Crosswind_Knots',
            'DepartureHour', 'Season', 'FlightDistance_Miles', 'AirportCongestion', 'RunwayCondition']]
    
    # 1. Classification
    y_class = df['Delayed']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    clf = DecisionTreeClassifier(max_depth=int(tree_depth), random_state=42)
    clf.fit(X_train_c, y_train_c)
    acc = accuracy_score(y_test_c, clf.predict(X_test_c))
    
    plt.figure(figsize=(10, 6))
    plot_tree(clf, feature_names=X.columns, class_names=['On-Time', 'Delayed'], filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Logic (Classification)", fontsize=12, fontweight='bold')
    
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight')
    buf1.seek(0)
    img_tree = Image.open(buf1)
    plt.close()
    
    # 2. Regression
    y_reg = df['DelayMinutes']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    nn = MLPRegressor(hidden_layer_sizes=(int(nn_neurons),), max_iter=800, random_state=42)
    nn.fit(X_train_r, y_train_r)
    y_pred_r = nn.predict(X_test_r)
    r2 = r2_score(y_test_r, y_pred_r)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test_r, y_pred_r, alpha=0.6, color='teal', edgecolor='k', label='Flights')
    plt.plot([0, 180], [0, 180], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.xlabel("Actual Delay (Minutes)")
    plt.ylabel("NN Predicted Delay (Minutes)")
    plt.title("Neural Network Regression Accuracy", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight')
    buf2.seek(0)
    img_nn = Image.open(buf2)
    plt.close()
    
    data_store["clf"] = clf
    data_store["nn"] = nn
    data_store["features"] = X.columns.tolist()
    
    results_summary = (
        f"🤖 Machine Learning Run Complete!\n\n"
        f"1. Classification Model (Decision Tree):\n"
        f"   • Test Accuracy: {acc * 100:.2f}%\n"
        f"   • Max Tree Depth: {tree_depth}\n"
        f"   • Features Used: 13 weather & operational inputs\n\n"
        f"2. Regression Model (Neural Network):\n"
        f"   • Hidden Layer Neurons: {nn_neurons}\n"
        f"   • R² Score: {r2:.4f}\n"
        f"   • Learned from weather, congestion & scheduling patterns."
    )
    
    return results_summary, img_tree, img_nn

# ==========================================
# FUNCTION 3: LIVE PREDICTOR (DEPLOYMENT)
# ==========================================
def predict_live(wind, vis, temp, precip, humidity, pressure, cloud_ceiling,
                 crosswind, hour, season, distance, congestion, runway_cond):
    if data_store["clf"] is None or data_store["nn"] is None:
        return "⚠️ You must train the models in Tab 2 before using live prediction."

    user_input = pd.DataFrame(
        [[wind, vis, temp, precip, humidity, pressure, cloud_ceiling,
          crosswind, int(hour), int(season), distance, congestion, int(runway_cond)]],
        columns=data_store["features"]
    )

    is_delayed     = data_store["clf"].predict(user_input)[0]
    predicted_mins = data_store["nn"].predict(user_input)[0]

    status = "🚨 DELAYED" if is_delayed == 1 else "🟢 ON-TIME"
    mins   = max(0, int(round(predicted_mins))) if is_delayed == 1 else 0

    runway_labels = {0: "Dry", 1: "Wet", 2: "Icy"}
    season_labels  = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}

    return (
        f"🔮 Live ML Model Prediction:\n"
        f"-----------------------------------------\n"
        f"• Status: {status}\n"
        f"• Predicted Delay Time: {mins} minutes\n"
        f"-----------------------------------------\n"
        f"Inputs — Wind: {wind}kts  Vis: {vis}mi  Temp: {temp}°F  Precip: {precip}\"\n"
        f"         Humidity: {humidity}%  Pressure: {pressure}inHg  Ceiling: {int(cloud_ceiling)}ft\n"
        f"         Crosswind: {crosswind}kts  Hour: {int(hour):02d}:00  Season: {season_labels.get(int(season), season)}\n"
        f"         Distance: {int(distance)}mi  Congestion: {congestion:.2f}  Runway: {runway_labels.get(int(runway_cond), runway_cond)}"
    )

# ==========================================
# GRADIO INTERFACE SETUP
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✈️ Airport Flight Delay Predictor & ML Simulator")
    gr.Markdown(
        "Welcome! This app is an interactive machine learning workspace designed to showcase how weather patterns "
        "affect flight delays. Follow the tabs below to step through the entire ML pipeline."
    )
    
    # TAB 1: DATA CREATION
    with gr.Tab("Step 1: Data Generator"):
        gr.Markdown("### Create Simulated Flight & Weather Records")
        with gr.Row():
            num_flights = gr.Slider(minimum=100, maximum=2000, value=500, step=50, label="Number of Flight Records")
            weather_severity = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Weather Severity Factor")
            congestion_level = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="Airport Congestion Level")

        gen_btn = gr.Button("Generate Dataset", variant="primary")
        data_summary = gr.Textbox(label="Dataset Metadata Summary", lines=4)
        data_preview = gr.DataFrame(label="Dataset Preview (First 5 Rows)")
        
        gen_btn.click(generate_data, inputs=[num_flights, weather_severity, congestion_level], outputs=[data_summary, data_preview])
        
    # TAB 2: TRAINING MODELS & VISUALIZING
    with gr.Tab("Step 2: Train & Analyze ML Models"):
        gr.Markdown("### Tune the Machine Learning Algorithms")
        with gr.Row():
            tree_depth = gr.Slider(minimum=2, maximum=6, value=3, step=1, label="Decision Tree Depth (Classification Limit)")
            nn_neurons = gr.Slider(minimum=2, maximum=50, value=10, step=2, label="Neural Network Neurons (Regression Size)")
            
        train_btn = gr.Button("Train Models & Plot Results", variant="primary")
        
        with gr.Row():
            ml_summary = gr.Textbox(label="Model Performance Metrics", lines=7)
            
        with gr.Row():
            tree_plot = gr.Image(label="Decision Tree Structure (How Decisions are Split)", type="pil")
            nn_plot = gr.Image(label="Neural Network Performance Graph", type="pil")
            
        train_btn.click(run_analysis, inputs=[tree_depth, nn_neurons], outputs=[ml_summary, tree_plot, nn_plot])

    # TAB 3: LIVE PREDICTION
    with gr.Tab("Step 3: Test Real-Time Weather"):
        gr.Markdown("### Input your own weather scenario to test the trained model!")
        with gr.Row():
            in_wind   = gr.Slider(0, 50, value=15, label="Wind Speed (Knots)")
            in_vis    = gr.Slider(0.1, 10.0, value=8.0, label="Visibility (Miles)")
            in_temp   = gr.Slider(0, 100, value=65, label="Temperature (\u00b0F)")
            in_precip = gr.Slider(0.0, 3.0, value=0.0, label="Precipitation (Inches)")
        with gr.Row():
            in_humidity  = gr.Slider(20, 100, value=60, step=1, label="Humidity (%)")
            in_pressure  = gr.Slider(28.5, 31.0, value=29.92, step=0.01, label="Barometric Pressure (inHg)")
            in_ceiling   = gr.Slider(500, 15000, value=8000, step=500, label="Cloud Ceiling (ft)")
            in_crosswind = gr.Slider(0, 35, value=5, step=1, label="Crosswind Component (Knots)")
        with gr.Row():
            in_hour       = gr.Slider(0, 23, value=10, step=1, label="Departure Hour (0–23)")
            in_season     = gr.Slider(1, 4, value=2, step=1, label="Season (1=Spring 2=Summer 3=Fall 4=Winter)")
            in_distance   = gr.Slider(100, 3000, value=800, step=50, label="Flight Distance (Miles)")
            in_congestion = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Airport Congestion (0=Low → 1=High)")
            in_runway     = gr.Slider(0, 2, value=0, step=1, label="Runway Condition (0=Dry 1=Wet 2=Icy)")

        predict_btn = gr.Button("Evaluate Flight Status", variant="stop")
        prediction_output = gr.Textbox(label="Live Model Prediction Output", lines=8)

        predict_btn.click(
            predict_live,
            inputs=[in_wind, in_vis, in_temp, in_precip, in_humidity, in_pressure,
                    in_ceiling, in_crosswind, in_hour, in_season, in_distance,
                    in_congestion, in_runway],
            outputs=prediction_output
        )

    # NEW TAB 4: EDUCATIONAL MATERIALS & HUB
    with gr.Tab("Step 4: ML Academy"):
        gr.Markdown("## 🎓 Machine Learning Theory & Learning Hub")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🧠 What is Machine Learning (ML)?")
                gr.Markdown(
                    "Instead of writing explicit rules (like *'if wind > 30 and rain > 1, delay flight'*), "
                    "we give a computer **historical data** (inputs and outcomes) and let it **discover the rules itself**.\n\n"
                    "There are two main tasks we ran in this simulator:\n"
                    "*   **Classification:** Grouping data into categories (e.g., Is a flight *Delayed* or *On-Time*?).\n"
                    "*   **Regression:** Predicting a continuous number (e.g., How many *minutes* of delay?)."
                )
                
                gr.Markdown("### 🕸️ What is an Artificial Neural Network (ANN)?")
                gr.Markdown(
                    "Inspired by the biological human brain, an ANN passes inputs through layers of interconnected **artificial neurons** (nodes).\n\n"
                    "*   Each connection has a **weight** (importance) and a **bias** (offset).\n"
                    "*   As the model trains, it calculates its error and sends it backward through the network (a process called **Backpropagation**), adjusting the weights until it gets the right answers!"
                )
            
            with gr.Column():
                gr.Markdown("### 📋 Understanding our Simulation Fields")
                gr.Markdown(
                    "**Core Weather (original)**\n"
                    "*   **WindSpeed_Knots:** Strong headwinds/tailwinds affect takeoff roll distance.\n"
                    "*   **Visibility_Miles:** Low visibility (fog/mist) increases runway spacing requirements.\n"
                    "*   **Temp_F:** Freezing temps trigger de-icing holds; extreme heat reduces lift.\n"
                    "*   **Precip_Inches:** Heavy rain/snow creates slick runways and lowers ceilings.\n\n"
                    "**Extended Atmospheric (new)**\n"
                    "*   **Humidity_Pct:** High humidity elevates fog and icing risk.\n"
                    "*   **Pressure_inHg:** Low barometric pressure signals approaching storm systems.\n"
                    "*   **CloudCeiling_Ft:** Below ~500 ft forces Instrument (IFR) approaches, increasing separation.\n"
                    "*   **Crosswind_Knots:** Perpendicular wind component — airports publish hard crosswind limits per aircraft type.\n\n"
                    "**Operational & Scheduling (new)**\n"
                    "*   **DepartureHour:** Morning/evening rush hours cascade delays across gates.\n"
                    "*   **Season:** Winter brings ice events; summer brings convective storms.\n"
                    "*   **FlightDistance_Miles:** Longer routes have more exposure to en-route weather diversions.\n"
                    "*   **AirportCongestion:** High traffic density amplifies any single delay (domino effect).\n"
                    "*   **RunwayCondition:** 0=Dry, 1=Wet, 2=Icy — directly impacts braking action reports.\n\n"
                    "**Targets**\n"
                    "*   **Delayed (Output/Class):** Binary label (0 = On-Time, 1 = Delayed).\n"
                    "*   **DelayMinutes (Output/Reg):** Continuous delay duration in minutes."
                )
                
                gr.Markdown("### 🏆 Popular Models Used Today")
                gr.Markdown(
                    "*   **Decision Trees:** Easy to read, maps data using yes/no questions (highly interpretable).\n"
                    "*   **Random Forests:** An 'ensemble' (group) of hundreds of decision trees voting on the answer.\n"
                    "*   **XGBoost / Gradient Boosting:** Highly accurate algorithm for structured table data (like Excel sheets).\n"
                    "*   **Neural Networks / Deep Learning:** Excellent for unstructured data like images, voice, or complex regression."
                )

        gr.HTML("<hr>")
        gr.Markdown("## 🚀 Advice and Curated Resources for Aspiring College Students")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💡 Why Learn Machine Learning?")
                gr.Markdown(
                    "Learning ML early teaches you **systemic thinking**. In college, you'll see ML applied to "
                    "medicine, space exploration, finance, and climate science. Building projects like this "
                    "demonstrates initiative, problem-solving, and a transition from a 'consumer' of technology to a 'builder'."
                )
            
            with gr.Column():
                gr.Markdown("### 🔗 Top Learning & Practice Platforms")
                gr.Markdown(
                    "*   **[Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course):** Superb, fast-paced, and highly interactive resource directly from Google.\n"
                    "*   **[Kaggle](https://www.kaggle.com):** A playground for data science. Find real flight/weather datasets, write code in the browser, and compete in ML challenges.\n"
                    "*   **[Coursera: Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction):** Taught by Andrew Ng, this is widely considered the absolute gold-standard starting point for beginner developers.\n"
                    "*   **[Fast.ai](https://www.fast.ai):** A free 'code-first' course where you build working models immediately before learning the heavy math."
                )

# TAB 5: Run training from the UI
    with gr.Tab("Step 5: Train Models (UI)"):
        gr.Markdown("### Run a short training job from the UI (uses your venv Python).")
        model_choice = gr.Dropdown(choices=["lstm", "cnn", "ssm", "transformer"], value="lstm", label="Model")
        epochs_slider = gr.Slider(1, 20, value=3, step=1, label="Epochs")
        run_btn = gr.Button("Run Training")
        train_log = gr.Textbox(label="Training Log", lines=20)

        def run_training_ui(model, epochs):
            cmd = [sys.executable, "train.py", "--model", model, "--epochs", str(int(epochs))]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                out = proc.stdout or ""
                if proc.stderr:
                    out += "\n\nSTDERR:\n" + proc.stderr
            except Exception as e:
                out = f"Failed to run training: {e}"
            return out

        run_btn.click(run_training_ui, inputs=[model_choice, epochs_slider], outputs=[train_log])

        gr.Markdown("---")
        gr.Markdown("### Train Deep Time-Series Models (LSTM / 1D-CNN) on Real CSV Data")
        csv_file = gr.File(label="Upload CSV (time-ordered)")
        csv_url = gr.Textbox(label="Or provide CSV URL (optional)")
        feat_cols = gr.Textbox(label="Feature columns (comma-separated)")
        target_col = gr.Textbox(label="Target column (single)")
        seq_len_slider = gr.Slider(10, 200, value=50, step=10, label="Sequence Length")
        deep_model_choice = gr.Dropdown(choices=["lstm", "cnn"], value="lstm", label="Model")
        deep_epochs = gr.Slider(1, 50, value=5, step=1, label="Epochs")
        run_deep = gr.Button("Train Deep Model")
        deep_log = gr.Textbox(label="Deep Training Log", lines=20)

        def run_training_deep(uploaded_file, url, feature_cols_str, target_col_str, seq_len, model_name, epochs):
            if uploaded_file is None and (url is None or url.strip() == ""):
                return "Please upload a CSV file or provide a CSV URL."
            path = None
            if uploaded_file is not None:
                path = uploaded_file.name
            else:
                path = url.strip()

            if not feature_cols_str or not target_col_str:
                return "Please provide comma-separated feature column names and a target column name."

            feat_cols = [c.strip() for c in feature_cols_str.split(',') if c.strip()]
            try:
                X, y = load_csv_time_series(path, feature_cols=feat_cols, target_col=target_col_str, seq_len=int(seq_len))
            except Exception as e:
                return f"Failed to load CSV and prepare sequences: {e}"

            if len(X) == 0:
                return "No training sequences produced. Check sequence length and data size."

            ds = TimeSeriesDataset(X, y)
            loader = DataLoader(ds, batch_size=32, shuffle=True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if model_name == 'lstm':
                model = LSTMForecaster(input_size=X.shape[-1], hidden_size=64, num_layers=2).to(device)
            else:
                model = CNN1DForecaster(input_channels=X.shape[-1], channels=[16, 32], kernel_size=3, seq_len=X.shape[1]).to(device)

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.MSELoss()

            logs = []
            for epoch in range(1, int(epochs) + 1):
                model.train()
                total = 0.0
                n = 0
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb).squeeze(-1)
                    loss = loss_fn(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += loss.item() * xb.size(0)
                    n += xb.size(0)
                logs.append(f"Epoch {epoch:3d} — Train MSE: {total / n:.6f}")

            return "\n".join(logs)

        run_deep.click(run_training_deep, inputs=[csv_file, csv_url, feat_cols, target_col, seq_len_slider, deep_model_choice, deep_epochs], outputs=[deep_log])


# Launch the interactive web server locally
if __name__ == "__main__":
    demo.launch(share=True)

