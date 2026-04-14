import torch
import pandas as pd
from models.full_model import PEModel

model = PEModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

data = pd.read_csv("data/sample/sample.csv")

for epoch in range(5):
    for _, row in data.iterrows():
        img = torch.randn(1, 1, 64, 64)
        clinical = torch.tensor([[row.d_dimer, row.heart_rate, row.spo2, row.age, 1]], dtype=torch.float32)

        text_ids = torch.randint(0, 100, (1, 128))
        mask = torch.ones_like(text_ids)

        label = torch.tensor([[row.label]], dtype=torch.float32)

        pred = model(img, clinical, text_ids, mask)

        loss = torch.nn.functional.binary_cross_entropy(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss {loss.item()}")
