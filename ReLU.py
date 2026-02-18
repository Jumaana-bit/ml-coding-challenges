from typing import Any
import torch
from torch import nn


class ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. DATA TYPE CHECK
        # The specification requires us to reject non-floating point inputs.
        # This ensures we don't accidentally try to perform ReLU on integers.
        if not torch.is_floating_point(x):
            raise TypeError(f"Expected floating point tensor, but got {x.dtype}")

        # 2. CREATE A REFERENCE "FLOOR"
        # We create a tensor filled with 0.0 that matches 'x' exactly in size.
        # Think of this as a 'mask' or a 'reference sheet' of zeros.
        # .zeros_like automatically copies the device (CPU/GPU) and dtype from x.
        zero_tensor = torch.zeros_like(x)
        
        # 3. ELEMENT-WISE COMPARISON
        # torch.maximum looks at every index (e.g., Row 1, Col 1).
        # It compares the value in 'x' with the value in 'zero_tensor' (which is 0).
        # It returns the larger of the two.
        #
        # Example:
        # x:           [-1.5,  2.0,  0.0]
        # zero_tensor: [ 0.0,  0.0,  0.0]
        # Result:      [ 0.0,  2.0,  0.0]
        #
        # This effectively 'filters' out all negative numbers.
        output = torch.maximum(x, zero_tensor)

        # 4. RETURN RESULT
        # We return the new tensor, leaving the original 'x' unmodified.
        return output
        
       #PyTorch functions learnt
       #torch.is_floating_point(x)
       #torch.zeros_like(x)
       #torch.maximum(x, zero_tensor)
