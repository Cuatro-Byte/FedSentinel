import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """
    Structure representing the result of evaluating a model.
    This strictly measures model performance and does not contain 
    any threat scores, attack-success rates, or Sentinel metadata.
    """
    loss: float
    accuracy: float
    sample_count: int

class Evaluator:
    """
    Infrastructure component responsible for evaluating a PyTorch model 
    against a dataset to produce standard performance metrics (loss, accuracy).
    
    It performs inference only, without accumulating gradients or modifying 
    model state.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the evaluator.
        
        Args:
            device: Device to use for evaluation ("cpu" or "cuda").
        """
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> EvaluationResult:
        """
        Evaluate the model on the provided dataloader.
        
        Args:
            model: PyTorch neural network to evaluate.
            dataloader: DataLoader containing the evaluation dataset.
            
        Returns:
            An EvaluationResult containing aggregate loss, accuracy, and sample count.
            
        Raises:
            ValueError: If the model is invalid, the dataset is empty, or a batch is malformed.
        """
        if model is None or not isinstance(model, nn.Module):
            raise ValueError("A valid PyTorch model must be provided.")
            
        if dataloader is None:
            raise ValueError("A valid DataLoader must be provided.")
            
        try:
            dataset_size = len(dataloader.dataset) # type: ignore
        except TypeError:
            raise ValueError("Dataset must implement __len__.")
            
        if dataset_size == 0:
            raise ValueError("Cannot evaluate an empty dataset.")

        # Remember the original mode so we can restore it exactly
        was_training = model.training

        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Perform inference only
        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                    raise ValueError(f"Malformed batch: expected (data, target), got {type(batch).__name__}.")
                
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                
                batch_size = data.size(0)
                
                output = model(data)
                loss = self.criterion(output, target)
                
                # Aggregate total loss across all samples to prevent batch-size weighting bias
                total_loss += loss.item() * batch_size
                
                # Calculate correct classifications
                _, predicted = torch.max(output.data, 1)
                correct_predictions += (predicted == target).sum().item()
                
                total_samples += batch_size

        # Restore original mode
        model.train(mode=was_training)

        # Calculate dataset-wide averages
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        return EvaluationResult(
            loss=avg_loss,
            accuracy=avg_accuracy,
            sample_count=total_samples
        )
