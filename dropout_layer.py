import torch
from torch import nn

class DropoutLayer(nn.Module):
    """
    Dropout Layer: A tool to prevent 'overfitting' in neural networks.
    
    WHY RANDOMNESS?
    Think of a group project: if one person does all the work, the others 
    become 'lazy.' By randomly 'dropping' students (neurons) during practice, 
    we force every student to learn the material independently. This prevents 
    'co-adaptation,' where neurons rely too heavily on each other.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        
        # 1. RANGE VALIDATION
        # p=1.0 is invalid because if we drop 100% of data, the scaling 
        # formula (1 / (1 - 1.0)) would result in division by zero.
        if not (0.0 <= p < 1.0):
            raise ValueError("Drop probability p must be in the range [0, 1).")
        
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2. EVALUATION MODE (The Test)
        # If self.training is False, we turn dropout OFF.
        # We want the 'full team' active to get the most accurate result.
        if not self.training:
            return x
        
        # 3. HANDLE p=0 CASE
        # If p is 0, no neurons are dropped; return input as-is.
        if self.p == 0.0:
            return x

        # 4. THE RANDOM MASK (The 'Plug Your Ears' Logic)
        # We generate a random number [0, 1) for every single element in x.
        # If the random number > p, the neuron 'stays' (1).
        # If the random number <= p, the neuron is 'dropped' (0).
        # This randomness ensures the network can't predict who will be missing.
        mask = (torch.rand_like(x) > self.p).to(x.dtype)

        # 5. INVERTED DROPOUT SCALING (The 'Volume' Fix)
        # Since we are removing 'p' percent of the signal, the total 'energy' 
        # drops. To fix this, we boost the survivors by 1/(1-p).
        # Example: If p=0.5, we divide by 0.5 (which is multiplying by 2).
        # This keeps the 'average volume' the same as in Evaluation mode.
        scale_factor = 1.0 / (1.0 - self.p)
        
        # 6. COMBINE
        # Input * Mask (0s and 1s) * Scale (the boost)
        return x * mask * scale_factor

# --- Summary of Behavior ---
# Train Mode: Randomly kills neurons and boosts survivors.
# Eval Mode:  Does nothing (Identity function).

# mask = (torch.rand_like(x) > self.p).to(x.dtype) 
# we are creating a boolean mask (True/False). Because we are using the "Greater Than" () operator, everything else (the "Less Than or Equal To" part) automatically becomes False
# scale_factor = 1.0 / (1.0 - self.p)
