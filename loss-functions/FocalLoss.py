import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive vs negative examples
               or a list of weights for each class. Default: 0.25
        gamma: Exponent of the modulating factor (1 - p_t)^gamma to balance easy vs hard examples.
               Default: 2.0
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where N = batch size, C = number of classes
            targets: (N) where each value is 0 <= targets[i] <= C-1
        
        Returns:
            Focal loss value (scalar)
        """
        ce_loss = self.ce_loss(inputs, targets)
        
        # Get softmax probabilities
        p = torch.exp(-ce_loss)
        
        # Apply focal term: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha if self.alpha >= 0 else 1.0
        
        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        return focal_loss.mean()


# Testing code

if __name__ == "__main__":
    # Example usage
    batch_size = 32
    num_classes = 10
    
    # Create sample inputs and targets
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Initialize loss function
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Compute loss
    loss = loss_fn(inputs, targets)
    
    print(f"Focal Loss: {loss.item():.4f}")
