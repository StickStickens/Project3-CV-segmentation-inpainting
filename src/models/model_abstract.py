import torch

class ModelAbstract(torch.nn.Module):
    """ Abstract base class for all models.
        All models should inherit from this class and implement the required methods.
    """
    def __init__(self):
        super(ModelAbstract, self).__init__()

    def forward(self, x: torch.Tensor):
        """ Forward pass of the model.
            Args:
                x: Input tensor.
            Returns:
                Output tensor after passing through the model.
        """
        pass # To be implemented by subclasses

    def predict(self, x: torch.Tensor):
        """ Prediction method for the model.
            Args:
                x: Input tensor.
            Returns:
                Output tensor after prediction.
        """
        pass # To be implemented by subclasses
    
    def fit_to_size(self, x: torch.Tensor) -> torch.Tensor:
        """ Fit input tensor to the size expected by the model.
            Args:
                x: Input tensor.
            Returns:
                Resized tensor.
        """
        pass # To be implemented by subclasses
    