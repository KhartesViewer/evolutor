import torch

# From ChatGPT 4.5

# Initialize x with an initial guess, requires_grad=True to track gradients
x = torch.tensor([0.1, 0.1], requires_grad=True)

# Use a PyTorch optimizer, e.g., LBFGS or Adam
optimizer = torch.optim.LBFGS([x], lr=0.1)

# Define your function (from Step 1)
def f_torch(xx, shift):
    xy = xx + shift
    return torch.stack([
        torch.sin(xy[0]) + xy[1]**2 - 3,
        torch.cos(xy[1]) + xy[0]**2 - 1,
        xy[0] * xy[1] - 2
    ])

# Optimization loop
def closure():
    optimizer.zero_grad()      # Clear previous gradients
    y = f_torch(x, 1.)             # Compute function output
    loss = torch.sum(y**2)     # Least squares objective
    loss.backward()            # Compute gradients
    return loss

# Run optimization for multiple iterations
for i in range(10):
    loss = optimizer.step(closure)
    print(f'Iteration {i+1}, Loss: {loss.item()}, x: {x.data.numpy()}')

