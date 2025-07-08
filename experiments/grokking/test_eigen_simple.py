import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Simple 2-layer MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=128, output_dim=97):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def compute_cos_grad_hgrad(model, x, y, criterion):
    """Compute cos(grad, H@grad)."""
    # Forward pass
    output = model(x)
    loss = criterion(output, y)
    
    # Get gradient
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Compute g^T @ g
    grad_norm_sq = sum((g * g).sum() for g in grads)
    
    # Get H @ grad by taking gradient of g^T @ g
    Hgrad = torch.autograd.grad(grad_norm_sq, params, retain_graph=True)
    
    # Compute cosine similarity
    grad_vec = torch.cat([g.flatten() for g in grads])
    Hgrad_vec = torch.cat([h.flatten() for h in Hgrad])
    
    cos_sim = torch.nn.functional.cosine_similarity(
        grad_vec.unsqueeze(0), 
        Hgrad_vec.unsqueeze(0)
    ).item()
    
    return cos_sim


# Run experiment
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

cos_sims = []
losses = []

print("Testing cos(grad, H@grad) evolution...")

# Train for 1000 steps
for i in range(1000):
    # Random data
    x = torch.randn(32, 200).to(device)
    y = torch.randint(0, 97, (32,)).to(device)
    
    # Forward pass
    output = model(x)
    loss = criterion(output, y)
    
    # Every 10 steps, compute cosine similarity
    if i % 10 == 0:
        cos_sim = compute_cos_grad_hgrad(model, x, y, criterion)
        cos_sims.append(cos_sim)
        losses.append(loss.item())
        
        if i % 100 == 0:
            print(f"Step {i}: cos(g,Hg) = {cos_sim:+.4f}, loss = {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot
plt.figure(figsize=(10, 6))
steps = list(range(0, 1000, 10))

plt.subplot(2, 1, 1)
plt.plot(steps, cos_sims, 'r-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axhline(y=1, color='g', linestyle='--', alpha=0.3)
plt.axhline(y=-1, color='b', linestyle='--', alpha=0.3)
plt.ylabel('cos(grad, H@grad)')
plt.title('Gradient-Hessian Eigenvector Alignment')
plt.grid(True, alpha=0.3)
plt.ylim(-1.1, 1.1)

plt.subplot(2, 1, 2)
plt.plot(steps, losses, 'b-', linewidth=2)
plt.ylabel('Loss')
plt.xlabel('Training Steps')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_cosine_test.png', dpi=150)
plt.show()

# Analysis
cos_array = np.array(cos_sims)
print(f"\n=== Results ===")
print(f"Range: [{cos_array.min():.4f}, {cos_array.max():.4f}]")
print(f"Mean: {cos_array.mean():.4f}")
print(f"Negative values: {(cos_array < 0).sum()} out of {len(cos_array)}")

if (cos_array < 0).any():
    print(f"Most negative: {cos_array.min():.4f}")
    print(f"First negative at step: {np.where(cos_array < 0)[0][0] * 10}")