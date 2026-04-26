import torch
import torch.nn as nn


class PoseTransformerClassifier(nn.Module):
    def __init__(self, num_features=8, num_classes=5, d_model=64, nhead=4, num_layers=2):
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


if __name__ == "__main__":
    batch_size = 32
    sequence_length = 60
    num_features = 8
    num_classes = 5

    model = PoseTransformerClassifier(
        num_features=num_features,
        num_classes=num_classes
    )

    dummy_input = torch.randn(batch_size, sequence_length, num_features)

    output = model(dummy_input)

    print(output.shape)