import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/energy_usage.csv')
print(df.head())
print(df.describe().T[['mean', 'std', 'min', 'max']])

x = df.drop('Daily_Energy_kWh', axis=1)
y = df['Daily_Energy_kWh']
feature_names = list(x.columns)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Elastic Net = alpha * (l1_ratio * L1 + (1 - l1_ratio)/2 * L2)
# l1_ratio=1 is pure Lasso, l1_ratio->0 is Ridge.
# On this dataset cross-validated R2 is statistically identical for every l1_ratio
# (differences are far smaller than the fold-to-fold std), so accuracy alone can't pick one.
# l1_ratio is therefore FIXED at 0.5 on purpose: the L2 half makes correlated features
# share weight instead of one being chosen arbitrarily. Only alpha is tuned by CV.
L1_RATIO = 0.5
model = make_pipeline(
    StandardScaler(),
    ElasticNetCV(l1_ratio=L1_RATIO, cv=5, random_state=42, max_iter=20000),
)
model.fit(x_train, y_train)
enet = model.named_steps['elasticnetcv']
pred = model.predict(x_test)

metrics = {
    "mae": round(mean_absolute_error(y_test, pred), 3),
    "rmse": round(mean_squared_error(y_test, pred) ** 0.5, 3),
    "r2": round(r2_score(y_test, pred), 4),
    "alpha": round(float(enet.alpha_), 5),
    "l1_ratio": float(enet.l1_ratio_),
    "n_train": len(x_train),
    "n_test": len(x_test),
}
print("Elastic Net:", metrics)

coefs = dict(zip(feature_names, [round(float(c), 4) for c in enet.coef_]))

# ---- Cross-validated accuracy comparison (training split only) ----
kf = KFold(5, shuffle=True, random_state=1)
cv_scores = {}
for name, ratio in [("Elastic Net (l1_ratio 0.5)", 0.5), ("Lasso (l1_ratio 1.0)", 1.0)]:
    m = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=ratio, cv=5, max_iter=20000))
    sc = cross_val_score(m, x_train, y_train, cv=kf, scoring='r2')
    cv_scores[name] = {"mean": round(float(sc.mean()), 4), "std": round(float(sc.std()), 4)}
print("CV R2:", cv_scores)

# ---- Lasso comparison (same data) ----
lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=20000))
lasso.fit(x_train, y_train)
lasso_coefs = dict(zip(feature_names, [round(float(c), 4) for c in lasso.named_steps['lassocv'].coef_]))
lasso_r2 = round(r2_score(y_test, lasso.predict(x_test)), 4)
print("Lasso R2:", lasso_r2)

print(f"\n{'feature':28s} {'ElasticNet':>11s} {'Lasso':>9s}")
for f in feature_names:
    print(f"{f:28s} {coefs[f]:11.4f} {lasso_coefs[f]:9.4f}")

joblib.dump(model, 'model_elasticnet.pkl')
with open('model_info.json', 'w') as f:
    json.dump(
        {
            "metrics": metrics,
            "lasso_r2": lasso_r2,
            "cv_scores": cv_scores,
            "elasticnet_coefs": coefs,
            "lasso_coefs": lasso_coefs,
            "feature_names": feature_names,
            "feature_ranges": {c: [float(x[c].min()), float(x[c].max())] for c in feature_names},
        },
        f,
        indent=2,
    )
print("\nModel Trained and Saved")
