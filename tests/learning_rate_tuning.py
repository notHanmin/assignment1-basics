from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e1)

for t in range(10):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.

# Test different learning rates and plot results
learning_rates = [1e-1, 1e-2, 1e-3]
colors = ['red', 'blue', 'green']
plt.figure(figsize=(10, 6))

print("Starting learning rate comparison...")

for lr, color in zip(learning_rates, colors):
    print(f"\nTesting learning rate: {lr}")
    # Initialize weights with the same random seed for fair comparison
    torch.manual_seed(42)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    
    losses = []
    
    for t in range(50):  # Run for more iterations to see convergence
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.cpu().item())
        loss.backward()
        opt.step()
        
        # Print progress every 10 iterations
        if t % 10 == 0:
            print(f"  Iteration {t}: Loss = {loss.cpu().item():.6f}")
    
    plt.plot(losses, label=f'lr={lr}', color=color, linewidth=2)
    print(f"  Final loss: {losses[-1]:.6f}")

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration for Different Learning Rates')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Use log scale for better visualization
plt.tight_layout()
plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'learning_rate_comparison.png'")
plt.close()

# Print final losses for comparison
print("\nFinal losses after 50 iterations:")
for lr, color in zip(learning_rates, colors):
    torch.manual_seed(42)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    
    for t in range(50):
        opt.zero_grad()
        loss = (weights**2).mean()
        loss.backward()
        opt.step()
    
    print(f"lr={lr}: {loss.cpu().item():.6f}")