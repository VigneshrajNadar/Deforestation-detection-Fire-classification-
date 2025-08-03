# Fire Type Classification using MODIS Satellite Data

🚀 **Live Demo:** [deforestation-detection-fire-classification.streamlit.app](https://deforestation-detection-fire-classification.streamlit.app)

A visually stunning, interactive Streamlit app for predicting fire types and exploring fire data in India using MODIS satellite readings. Features a fully immersive fire-themed UI, advanced animated effects, and rich data visualizations.

## � Download Model & Data Files

You can directly download all required model and data files from the following Google Drive links:

- [best_fire_detection_model.pkl](https://drive.google.com/file/d/1k3NI_5b-hb-XmIgFnGwa7bLygq8F1wt8/view?usp=share_link)
- [scaler.pkl](https://drive.google.com/file/d/1K787fvWuCc-ojxMmiWtYT9vPb9AckYXs/view?usp=share_link)
- [modis_2021_India.csv](https://drive.google.com/file/d/17UZzdC-UiKiDhgDYTz-S211nJ708s_U0/view?usp=share_link)
- [modis_2022_India.csv](https://drive.google.com/file/d/1ZFMx-GieGBHP9Sabe4Nr1kz1UQzCKzY-/view?usp=share_link)
- [modis_2023_India.csv](https://drive.google.com/file/d/1xwFXLlsiDJo7ID0FUvN94tmq7hgaViDQ/view?usp=share_link)
- [Classification_of_Fire_Types_in_India_Using_MODIS_Satellite_Data.ipynb](https://drive.google.com/file/d/1jiFCYZJ-7RVk5BHad7uX-ut7gfNP53l0/view?usp=share_link)

These files will be automatically downloaded by the app on Streamlit Cloud, but you can also download them manually if running locally.

## �🚀 Features

- **Fire-Themed Animated UI**: Immersive, modern design with animated backgrounds, glowing/pulsing cards, bouncing icons, spinning fire emoji, rainbow shimmer, and floating ember particles for a lively, engaging experience.
- **Prediction Page**: Enter MODIS features to predict fire type (Vegetation, Other Static Land Source, Offshore) with animated feedback, a dynamic legend, and fire burst effect on prediction.
- **Fire Type Legend**: Multi-layered animation including: animated SVG fire wave, glowing border, bouncing icons, spinning emoji, rainbow shimmer, and 8 floating fire particles for maximum visual impact.
- **Data Visualization Page**: Explore MODIS fire data (2021-2023) with interactive, animated charts (bar, pie, scatter, heatmap, animated time series, and more). Sidebar filters and expanders make exploration seamless.
- **Sidebar Filters & Smooth UX**: Filter data by year, confidence, and fire type. Expand/collapse chart groups, and enjoy smooth animated transitions throughout the app.
- **Beginner-Friendly**: No coding required. Just follow the setup instructions and run the app locally.

## 🧭 App Structure & Pages

### 🔥 Prediction Page
- Enter six MODIS features (Brightness, Brightness T31, FRP, Scan, Track, Confidence).
- Click **Predict Fire Type** to get an instant prediction.
- Animated feedback: fire burst, pulsing button, and a dynamic, multi-animated legend.
- **Fire Type Legend** explains each fire category with icons, color codes, and animation.
- Floating fire emoji in the header for extra flair.

### 📊 Data Visualization Page
- Interactive charts for all years (2021-2023):
    - **Bar Charts**: Fire type distribution, monthly trends, top-N locations.
    - **Pie Charts**: Proportion of fire types, confidence breakdown.
    - **Stacked Bar Charts**: Fire type by month/year.
    - **Scatter Plots**: Animated fire detections over time.
    - **Heatmaps**: Correlation between features.
    - **Animated Charts**: Animated bar, line, and scatter plots by year/time.
- Sidebar filters: Filter by year, confidence, and fire type.
- Expanders group related charts for easy navigation.

## ✨ Animation Details
- **Fire Wave Background**: SVG at the bottom of the legend, animated to simulate fire movement.
- **Legend Card**: Glowing, pulsing border and background.
- **Icons**: Fire type icons bounce gently.
- **Fire Emoji**: Spins and glows in the legend and header.
- **Legend Header**: Rainbow shimmer animation for visual emphasis.
- **Floating Embers**: 8 particles with unique trajectories, sizes, and speeds.
- **Predict Button**: Pulses and glows, intensifies on hover.
- **Fire Burst**: Animated effect on prediction result.
- **Smooth Transitions**: All chart and UI transitions are animated for a premium feel.

## 📊 Interactive Chart Types
- Bar, Pie, Stacked Bar, Scatter, Animated Scatter, Line, Heatmap.
- All charts are interactive: hover for tooltips, zoom, pan, filter.
- Sidebar lets you filter data by year, confidence, fire type.

## 🛠️ Troubleshooting
- **Model File Error**: Ensure `best_fire_detection_model.pkl` and `scaler.pkl` are in the app directory.
- **Dependency Error**: Use the provided `requirements.txt` and recommended Python 3.10+ (Apple Silicon preferred).
- **TensorFlow/NumPy Error**: Make sure you use the exact versions in `requirements.txt` for Mac M1/M2.
- **Browser Issues**: Use Chrome, Edge, or Firefox for best animation support.
- **Data Files Missing**: Ensure all CSVs are in the directory.
- **Still stuck?**: Delete `.pyc` files, restart Streamlit, or reinstall dependencies.

## 🎨 Customization
- **Theme**: Edit the CSS in `app.py` to change colors, animation speed, or add your own effects.
- **Animations**: Adjust keyframes and style blocks in the HTML/CSS for more/less animation.
- **Charts**: Add new chart types in the Data Visualization section using Plotly.

## 🧪 Sample Prediction
**Input:**
- Brightness: 320.0
- Brightness T31: 295.0
- FRP: 18.0
- Scan: 1.1
- Track: 1.0
- Confidence: high

**Output:**
- Predicted Fire Type: Vegetation Fire
- Animated fire burst and legend highlight

## ❓ FAQ
**Q: Why do I see a model version warning?**
A: Make sure you use the provided `.pkl` files and not older versions.

**Q: Can I run this on Windows/Linux?**
A: Yes, but you may need to adjust TensorFlow and NumPy versions for your architecture.

**Q: How do I turn off animations?**
A: Comment out or remove the CSS animation blocks in `app.py`.

**Q: Can I use my own fire data?**
A: Yes! Replace the CSVs with your own, keeping the same column structure.

**Q: Who made this?**
A: See Credits below!

## 🖥️ Setup Instructions

1. **Clone this repo** and navigate to the `Fire_Classification` directory.
2. **Install dependencies** (see requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```
3. **Download model files** (required):
   - [`best_fire_detection_model.pkl`](https://drive.google.com/file/d/1eKzjHkWz5n1lI5lKq8nQw5l7XNiK9EJQ/view?usp=sharing)
   - [`scaler.pkl`](https://drive.google.com/file/d/1eKzjHkWz5n1lI5lKq8nQw5l7XNiK9EJQ/view?usp=sharing)
   Place both files in the `Fire_Classification` directory.
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## 📊 Data Files
- `modis_2021_India.csv`, `modis_2022_India.csv`, `modis_2023_India.csv` (in project directory)

## 📝 Notes
- Designed for Apple Silicon (arm64) and MacOS. All dependencies set for ML/visualization compatibility.
- For best experience, use latest Chrome/Edge/Firefox.
- If you see a model version warning, ensure you are using the provided `.pkl` files.

## 🗂️ Project Structure

```
Fire_Classification/
├── app.py                  # Main Streamlit app (UI, prediction, visualization, animation)
├── requirements.txt        # All dependencies for Apple Silicon/arm64
├── README.md               # This documentation
├── best_fire_detection_model.pkl  # Trained ML model (download separately)
├── scaler.pkl              # Scaler for input features (download separately)
├── modis_2021_India.csv    # Fire data for 2021
├── modis_2022_India.csv    # Fire data for 2022
├── modis_2023_India.csv    # Fire data for 2023
```

- **app.py**: The heart of the project. Handles UI, user input, prediction logic, data loading, all advanced animation, and data visualization.
- **requirements.txt**: Ensures you have all compatible libraries (especially for Mac M1/M2).
- **README.md**: Complete guide and documentation.
- **best_fire_detection_model.pkl**: Pre-trained ML model for fire type classification (download link above).
- **scaler.pkl**: Feature scaler used to preprocess input for the model (download link above).
- **modis_*.csv**: Yearly MODIS fire data for interactive visualizations.

## 🤖 Model Overview & Usage

This app uses a **scikit-learn machine learning model** trained to predict fire type based on MODIS satellite data.

### Model Training Details
- **Data Source:** NASA MODIS Active Fire Detections (India, 2021-2023). [MODIS C6.1 Documentation](https://modis.gsfc.nasa.gov/data/dataprod/mod14.php)
- **Preprocessing:**
    - Data cleaned for missing/invalid values.
    - Dates parsed, outliers handled.
    - Features selected for predictive power.
    - Confidence mapped from string to ordinal.
- **Feature Engineering:**
    - Used: Brightness, Brightness T31, FRP, Scan, Track, Confidence.
    - Confidence encoded as: low=0, nominal=1, high=2.
    - All features scaled (StandardScaler).
- **Algorithm:**
    - RandomForestClassifier (sklearn), tuned via grid search.
    - Model trained on 2021-2022 data, validated on 2023.
- **Evaluation:**
    - Accuracy: ~93% on validation set.
    - Confusion matrix and feature importance available in notebook.
    - Model and scaler exported as .pkl files for use in the app.

### Prediction Workflow
1. **Input Features:**
    - Brightness
    - Brightness T31
    - Fire Radiative Power (FRP)
    - Scan
    - Track
    - Confidence (categorical: low, nominal, high)
2. **Scaling:**
    - Input features are scaled using the provided `scaler.pkl` to match the model's training distribution.
3. **Prediction:**
    - The scaled input is passed to `best_fire_detection_model.pkl` (a scikit-learn classifier) to predict the fire type.
4. **Output:**
    - The prediction is mapped to one of three types: Vegetation Fire, Other Static Land Source, Offshore Fire.
    - The result is shown with animated feedback and a legend for context.

**Integration in app.py:**
- Both `scaler.pkl` and `best_fire_detection_model.pkl` are loaded at app startup.
- User input is collected via Streamlit widgets, scaled, and fed into the model.
- The prediction result triggers UI animations and updates the legend.
- All logic is contained in `app.py` for simplicity and ease of customization.

### 📁 CSV Data Usage
- Each `modis_*.csv` contains fire detection records for a year.
- Used for all data visualizations (charts, trends, maps).
- Charts update interactively based on sidebar filters.
- Data is never sent outside your machine—privacy is preserved.

## 🛠️ Extending the App
- **Add Features:**
    - Add new input fields or chart types in `app.py`.
    - Use Plotly for custom visualizations.
- **Retrain Model:**
    - Use your own MODIS data, retrain in a Jupyter notebook, export `.pkl` files.
- **Deploy:**
    - Use Streamlit Cloud, Heroku, or Docker for web deployment.
    - Add authentication if deploying publicly.

## 🔒 Security & Privacy
- All computation is local—no data leaves your device.
- Model files are not shared or uploaded.
- For sensitive data, run in a secure environment.

## 🌐 Accessibility & Browser Support
- Tested on Chrome, Edge, Firefox (latest versions).
- High-contrast colors and large fonts for readability.
- Responsive layout for desktop and tablets.
- Animations can be disabled by editing CSS in `app.py`.

## 🔗 Useful Links
- [MODIS Active Fire Data](https://firms.modaps.eosdis.nasa.gov/active_fire/)
- [NASA MODIS Overview](https://modis.gsfc.nasa.gov/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [Scikit-learn](https://scikit-learn.org/stable/)

## 🚀 Future Improvements
- Add user authentication for cloud deployments.
- Support for mobile devices and offline mode.
- More advanced ML models (deep learning, time series).
- Downloadable reports and export options.
- Live data integration (e.g., NASA FIRMS API).
- Accessibility improvements (screen reader support).

## 🔄 Visual Workflow

```
User Input (UI)  --->  Feature Scaling  --->  ML Model Prediction  --->  Animated Result
      |                     |                        |                        |
      v                     v                        v                        v
Data Visualization <--- CSV Data Load <--- Data Filtering <--- Sidebar Controls
```

## ✨ Credits
- UI/UX, animation, and ML engineering: [Your Name]
- Data: NASA MODIS

---

Enjoy a blazing, animated experience as you classify and explore fire data! 🔥
