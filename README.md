---
title: WattWise Household Energy Forecaster
emoji: ⚡
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# WattWise — Household Energy Forecaster (Elastic Net Regression)

Day 6 of the daily ML project series, and the first one built with **Python + Gradio**
(Days 1–5 used FastAPI + Streamlit).

Predicts a household's **daily electricity use (kWh)** and shows which factors drive it.

## What's in the dashboard
- **Forecast tab**: live prediction as you move any input, one-click presets, a gauge, monthly kWh,
  a bill estimate (with your own tariff), an approximate CO₂ figure, and a per-feature
  "what's moving this forecast" breakdown.
- **Model insights tab**: test metrics, a cross-validation comparison of Elastic Net vs Lasso, and an
  **alpha slider that refits the model live** so you can watch coefficients shrink.
- **About the data tab**: how the dataset was built and the honest caveats.

## Algorithm
**Elastic Net** = a linear model with a blend of L1 (Lasso) and L2 (Ridge) penalties.
`alpha` sets the overall strength, `l1_ratio` sets the mix (1.0 = pure Lasso).

- Features are standardised (`StandardScaler`) inside a single scikit-learn `Pipeline`.
- `l1_ratio` is **fixed at 0.5**; `alpha` is chosen by 5-fold cross-validation (`ElasticNetCV`).

## Dataset (synthetic, generated for this project)
`data/energy_usage.csv`: 260 households, 13 features, target `Daily_Energy_kWh`.
Built by `generate_dataset.py` so the true drivers are known:

| Group | Features |
|---|---|
| Climate (correlated) | Outdoor_Temp_C, Humidity_Pct, AC_Hours, Heating_Hours |
| Household (strongly correlated) | Occupants, Home_Area_sqft, Num_Rooms, Appliance_Count |
| Unrelated (random noise) | Neighbourhood_Noise_Index, Window_Color_Code, Pet_Count, Roof_Age_Years, Solar_Panel_Score |

## Results (20% held-out test split)
- R²: **0.812**, MAE: **1.84 kWh**, alpha ≈ 0.053

## Honest notes
- **Elastic Net and Lasso predict equally well here.** Cross-validated R² is 0.848 ± 0.024 vs 0.849 ± 0.022,
  a tie. I fixed `l1_ratio=0.5` deliberately, because the L2 half makes correlated features *share* weight.
  The visible difference is in how the weights are distributed (e.g. *Occupants* and *Home area* get
  much larger coefficients than under Lasso), not in accuracy.
- At the tuned alpha the unrelated features are shrunk to small values (|coef| < 0.15) but **not all exactly
  zero**. Raise alpha with the slider to watch more of them hit exactly zero.
- The data is synthetic, so this demonstrates the method and is not a real energy model. The bill and CO₂
  numbers are simple assumptions.

## Run locally
```bash
pip install "gradio>=6" -r requirements.txt
python app.py            # opens on http://127.0.0.1:7860
```
Optional: regenerate the data and retrain.
```bash
python generate_dataset.py
python train.py          # writes model_elasticnet.pkl and model_info.json
```
If the saved model is missing or was pickled with an incompatible scikit-learn version,
`app.py` retrains automatically on startup.

Deploy on Render
1. Push the project to a GitHub repo (same as your other days).
2. On render.com, sign in with GitHub → New → Web Service → select the repo.
3. Fill in:
      Root Directory: blank (repo root)
      Build Command: pip install -r requirements.txt && python train.py
      Start Command: python app.py
      Instance Type: Free
4. Deploy. Watch the Logs tab — first build takes a few minutes since it trains the model.
5. Your app is live at https://<your-service-name>.onrender.com

## Project structure
```
energy-usage-elasticnet/
├── app.py                  # Gradio dashboard (Blocks, tabs, custom CSS/theme, live events)
├── train.py                # trains Elastic Net (+ Lasso comparison), saves model + model_info.json
├── generate_dataset.py     # builds the synthetic dataset
├── data/energy_usage.csv
├── model_elasticnet.pkl
├── model_info.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Note on Gradio 6
In Gradio 6 `theme`, `css` and `js` are passed to `launch()`, not to `gr.Blocks()`. Older tutorials show the
old placement.
