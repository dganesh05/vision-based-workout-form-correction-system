import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class PoseTransformerClassifier(nn.Module):
    def __init__(self, num_features=8, num_classes=2, d_model=64, nhead=4, num_layers=2):
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

        x = x.mean(dim=1)

        return self.classifier(x)


# Load CSV
df = pd.read_csv("feedback/sample_features.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Add fake sequence dimension for transformer
X_tensor = X_tensor.unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model
model = PoseTransformerClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20

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
    print("\n--- Predictions ---")

    model.eval()

    feedback_map = {
        0: "Good squat form ✅",
        1: "Keep knees aligned and go deeper ⚠️"
    }

    with torch.no_grad():
        for i in range(len(X_tensor)):
            sample = X_tensor[i].unsqueeze(0)

            prediction = model(sample)
            predicted_class = torch.argmax(prediction, dim=1).item()

            print(f"Sample {i + 1}: Predicted Class = {predicted_class}")
            print(f"Feedback: {feedback_map[predicted_class]}")
            print()