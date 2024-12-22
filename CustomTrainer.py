from transformers import Trainer
import json
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
class CustomTrainer(Trainer):
    def __init__(self, *args, prediction_writer=None, **kwargs):
        """
        Initializes the CustomTrainer.

        Args:
            prediction_writer (PredictionWriter): Instance of PredictionWriter for saving predictions.
        """
        super().__init__(*args, **kwargs)
        self.prediction_writer = prediction_writer

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Overrides the training step to include saving predictions.

        Args:
            model (PreTrainedModel): The model to use for predictions.
            inputs (dict): The inputs to the model.

        Returns:
            torch.Tensor: The loss value for backpropagation.
        """
        model.train()  # Ensure model is in training mode initially
        
        # Save predictions only when needed
        if self.prediction_writer:
            # Switch model to evaluation mode temporarily for logging
            model.eval()

            # Prepare inputs for the forward pass
            inputs = self._prepare_inputs(inputs)
            
            # Forward pass for logging
            with torch.no_grad():  # No need to track gradients during logging
                outputs = model(**inputs)
                logits = outputs.logits if isinstance(outputs, dict) else outputs[1]
                labels = inputs.get("labels")
                input_ids = inputs.get("input_ids")
                current_epoch = self.state.epoch + 1
                zipped = zip(logits, labels, input_ids)
                
                # Write predictions
                for logits, labels, input_ids in zipped:
                    self.prediction_writer.write(json.dumps({
                        "logits": logits.tolist(),
                        "labels": labels.tolist(),
                        "input_ids": input_ids.tolist(),
                        "epoch": current_epoch
                    }) + "\n")
            
            # Return model to training mode after logging
            model.train()

        # Call the original training step (with optimizations)
        loss = super().training_step(model, inputs)
        return loss
