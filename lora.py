import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        
        self.weight = nn.Parameter(original_linear.weight.data, requires_grad=False)
        self.bias = nn.Parameter(original_linear.bias.data, requires_grad=False) if original_linear.bias is not None else None
        self.lora_A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check the dtype of x and ensure all tensors are the same type
        dtype = x.dtype
        device = x.device
        
        # Ensure that LoRA parameters are on the same device and dtype as x
        lora_A = self.lora_A.to(device).to(dtype)
        lora_B = self.lora_B.to(device).to(dtype)
        x = x.to(dtype)  # Ensure input x is of the correct dtype
        
        result = x @ self.weight.T
        if self.bias is not None:
            result += self.bias
        
        lora_out = x @ lora_A.T
        lora_out = lora_out @ lora_B.T
        result += self.scale * lora_out
        
        return result


def replace_linear_with_lora(module: nn.Module, r: int = 4, alpha: float = 1.0):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha))
        else:
            replace_linear_with_lora(child, r, alpha)

def apply_lora_replacement(models, lora_rank, lora_alpha):
    """
    Applies LoRA replacements to all nn.Linear layers in the given models.

    Args:
        models: A list of models to process (e.g., unet, vae, text_encoder).
        lora_rank: The rank for the LoRA layers.
        lora_alpha: The scaling factor for the LoRA layers.
    """
    for model in models:
        replace_linear_with_lora(model, lora_rank, lora_alpha)
        for name, module_ in model.named_modules():
            if isinstance(module_, LoRALinear):
                print(f"LoRA-change: {name}")
