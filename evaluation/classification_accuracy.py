import torch

from eeg_emotion_classification.models import load_dgcnn_from_checkpoint
from dataset_utils.transforms.de import DifferentialEntropy


class EvaluateEEGEmotionClassificationAccuracy:
    def __init__(self, dataset_name: str, config: dict, use_custom_dataset: bool = False):
        self.dataset_name = dataset_name
        self.config = config
        self.use_custom_dataset = use_custom_dataset
        self.transform = DifferentialEntropy(
            fs=self.config['eeg_specific']['sampling_rate']
        )
            
        self.classifier = load_dgcnn_from_checkpoint(
            checkpoint_path=f"saved_models/dgcnn_classifier-{dataset_name}.ckpt"
        )
    
    def compute(self, x_gen, true_labels):
        # Load pre-trained DGCNN model
        self.classifier.eval()

        # Apply DE transform if required
        if self.use_custom_dataset:
            x_gen_transformed = self.transform(x_gen)
        else:
            x_gen_transformed = x_gen

        # Get predictions
        with torch.no_grad():
            predictions = self.classifier(x_gen_transformed)
            predicted_labels = torch.argmax(predictions, dim=1)
        # Calculate accuracy
        accuracy = (predicted_labels == true_labels).float().mean().item()
        return accuracy
