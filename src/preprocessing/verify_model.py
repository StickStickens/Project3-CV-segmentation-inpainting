from ..train import  model_trainer
from ..utils.losses import choose_loss
from ..utils.params import Params, integrate_global_parameters
from ..dataset.dataset import get_random_subset
import os
def verify_model_file(model_path):
    """Verify that the model file exists and is in a valid format."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    
    valid_extensions = ['.pt', '.pth', '.h5', '.onnx']
    _, ext = os.path.splitext(model_path)
    ext = ext.lower()
    if ext not in valid_extensions:
        raise ValueError(f"The model file extension {ext} is not supported. Supported extensions are: {valid_extensions}")
    
    print(f"The model file at {model_path} is verified and valid.")
    return True

def verify_model_memorization(model, dataset, sample_size=10, params : Params | None = None, epochs=100):
    """Verify if the model can memorize a small subset of the dataset."""
    if params is None:
        params = Params()
    params = integrate_global_parameters(params)
    params.set('num_epochs', epochs)
    subset = get_random_subset(dataset, sample_size)
    trainer = model_trainer(model)
    trainer.train_model(subset, subset, parameters=params)
    train_loss = trainer.validate_1_epoch({'CombinedLoss': choose_loss('CombinedLoss', {})}, device=params.get('device', 'cpu'))
    
    if train_loss['CombinedLoss'] < 1e-4:
        print("Model successfully memorized the small subset of data.")
        return True
    else:
        print("Model failed to memorize the small subset of data.")
        return False
    