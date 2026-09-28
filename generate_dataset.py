"""
generate_dataset.py
-------------------
Builds a synthetic household energy dataset designed to showcase Elastic Net.

Two correlated feature GROUPS drive consumption:
  Climate group   : Outdoor_Temp_C, Humidity_Pct, AC_Hours, Heating_Hours
  Household group : Occupants, Home_Area_sqft, Num_Rooms, Appliance_Count
Plus 5 pure-noise columns that have no effect on consumption.

Lasso tends to keep one feature per correlated group and drop the rest arbitrarily;
Elastic Net (L1 + L2) spreads weight across the correlated members while still
zeroing the noise columns.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 260

# ---- Climate group (strongly correlated) ----
temp = rng.uniform(2, 42, n)
humidity = (85 - 0.9 * temp + rng.normal(0, 8, n)).clip(15, 98)
ac_hours = (0.35 * (temp - 22) + rng.normal(0, 1.2, n)).clip(0, 16)
heating_hours = (0.30 * (16 - temp) + rng.normal(0, 1.0, n)).clip(0, 12)

# ---- Household size group (strongly correlated) ----
occupants = rng.integers(1, 8, n)
area = (380 * occupants + rng.normal(500, 120, n)).clip(350, 4200)
rooms = (area / 450 + rng.normal(0, 0.3, n)).round().clip(1, 10)
appliances = (4 + 2.2 * occupants + rng.normal(0, 1.0, n)).round().clip(3, 30)

# ---- Pure noise columns ----
noise_index = rng.uniform(20, 90, n)            # neighbourhood noise level
window_color = rng.integers(1, 6, n)            # window frame colour code
pet_count = rng.poisson(1.0, n)                 # pets (no real effect here)
roof_age = rng.integers(1, 40, n)               # roof age, no effect
solar_panel_score = rng.uniform(0, 10, n)       # unrelated score

# ---- True consumption process (kWh per day) ----
kwh = (
    4.0
    + 1.10 * ac_hours
    + 0.85 * heating_hours
    + 0.06 * humidity
    + 0.16 * (area / 100)
    + 0.9 * occupants
    + 0.55 * rooms
    + 0.35 * appliances
    + rng.normal(0, 2.6, n)
).clip(2, None)

df = pd.DataFrame({
    "Outdoor_Temp_C": temp.round(1),
    "Humidity_Pct": humidity.round(1),
    "AC_Hours": ac_hours.round(1),
    "Heating_Hours": heating_hours.round(1),
    "Occupants": occupants,
    "Home_Area_sqft": area.round(0).astype(int),
    "Num_Rooms": rooms.astype(int),
    "Appliance_Count": appliances.astype(int),
    "Neighbourhood_Noise_Index": noise_index.round(1),
    "Window_Color_Code": window_color,
    "Pet_Count": pet_count,
    "Roof_Age_Years": roof_age,
    "Solar_Panel_Score": solar_panel_score.round(1),
    "Daily_Energy_kWh": kwh.round(2),
})
df.to_csv("data/energy_usage.csv", index=False)
print(df.shape)
print(df.corr()["Daily_Energy_kWh"].round(2))
