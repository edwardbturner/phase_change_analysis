import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModularAdditionMLP(nn.Module):
    """Simple MLP for modular addition - no attention layers."""
    def __init__(self, p, hidden_size=256):
        super().__init__()
        self.p = p
        
        # Embedding for inputs
        self.embed = nn.Embedding(p, hidden_size)
        
        # MLP layers
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, p)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # x shape: (batch_size, 2) containing [a, b]
        # Embed inputs
        embedded = self.embed(x)  # (batch_size, 2, hidden_size)
        
        # Flatten
        x = embedded.view(embedded.size(0), -1)  # (batch_size, 2 * hidden_size)
        
        # MLP forward
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x