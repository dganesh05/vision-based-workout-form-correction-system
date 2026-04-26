import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class PoseTransformerClassifier(nn.Module):
    def __init__(self, num_features=6, num_classes=2, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)

        # Mean pooling across sequence dimension
        x = x.mean(dim=1)

        return self.classifier(x)


# =========================
# Load REAL dataset
# =========================

df = pd.read_csv("feedback/final_features.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Add fake sequence dimension for transformer
X_tensor = X_tensor.unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# =========================
# Model setup
# =========================

model = PoseTransformerClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20


# =========================
# Training loop
# =========================

for epoch in range(epochs):
    total_loss = 0

    for batch_X, batch_y in loader:
        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


# =========================
# Prediction on ALL samples
# =========================

print("\n--- Predictions for All Samples ---")

feedback_map = {
    0: "Good squat form ✅",
    1: "Incorrect squat form ⚠️"
}

model.eval()

correct_count = 0

with torch.no_grad():
    for i in range(len(X_tensor)):
        sample = X_tensor[i].unsqueeze(0)

        prediction = model(sample)
        predicted_class = torch.argmax(prediction, dim=1).item()

        actual_label = y[i]

        if predicted_class == actual_label:
            correct_count += 1

        print(f"Sample {i+1}")
        print(f"Actual Label: {actual_label}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Feedback: {feedback_map[predicted_class]}")
        print()

accuracy = (correct_count / len(X_tensor)) * 100

print("===================================")
print(f"Final Accuracy: {accuracy:.2f}%")
print("===================================")