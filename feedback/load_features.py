import pandas as pd
import torch

df = pd.read_csv("feedback/sample_features.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

print("Feature shape:", X_tensor.shape)
print("Labels shape:", y_tensor.shape)