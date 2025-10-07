import torch
import torch.nn as nn

# Data
X = torch.tensor([[0., 0.], [1., 1.]], requires_grad=True)
y = torch.tensor([0, 1], dtype=torch.long)

# Model
model = nn.Sequential(
    nn.Linear(2, 5),
    nn.Tanh(),
    nn.Linear(5, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.LBFGS(model.parameters())

def closure():
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    return loss

# Train
optimizer.step(closure)

# Gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} gradient:\n{param.grad}")
