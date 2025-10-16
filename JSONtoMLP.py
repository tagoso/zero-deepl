import torch
import torch.nn as nn

data = [
    {"id": 1, "age": 25, "scores": [3.5, 4.0, 2.8]},
    {"id": 2, "age": 31, "scores": [2.1, 3.3, 4.8]},
    {"id": 3, "age": 22, "scores": [4.1, 3.9, 3.5]},
]

# JSON to num array
X_list = [[d["age"]] + d["scores"] for d in data]
X = torch.tensor(X_list, dtype=torch.float32)

print(X)
print(X.shape)

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

y_pred = model(X)
print(y_pred)
