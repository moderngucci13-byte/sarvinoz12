import numpy as np
import pandas as pd

# Linear Regression - noldan (Normal Equation): beta = (X^T X)^-1 X^T y
df = pd.read_csv("data/05_taxi_fare__1_.csv", encoding="utf-8-sig")
target = "fare_thousand_uzs"
features = [c for c in df.columns if c != target]
X = df[features].to_numpy(dtype=float)
y = df[target].to_numpy(dtype=float)

X_b = np.c_[np.ones((X.shape[0], 1)), X]           # intercept ustuni
beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y      # normal equation

print("=== Koeffitsiyentlar ===")
print(f"b0 (intercept) = {beta[0]:.6f}")
for f, c in zip(features, beta[1:]):
    print(f"{f:26s} = {c:.6f}")

y_pred = X_b @ beta
res = y - y_pred
SSE = np.sum(res ** 2)
MSE = np.mean(res ** 2)
R2 = 1 - SSE / np.sum((y - y.mean()) ** 2)
print("\n=== Metrikalar ===")
print(f"SSE={SSE:.6f}  MSE={MSE:.6f}  RMSE={np.sqrt(MSE):.6f}  MAE={np.mean(np.abs(res)):.6f}  R2={R2:.6f}")

new = np.array([8,20,2,3,2], dtype=float)
print("\n=== Yangi qiymat ===")
for f, v in zip(features, new):
    print(f"{f:26s}: {v}")
print(f"Prognoz: {np.r_[1.0, new] @ beta:.2f}")

print("\n=== Haqiqiy va bashorat ===")
print(pd.DataFrame({"actual": y, "predicted": y_pred, "residual": res}).round(3))
