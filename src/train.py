import torch
from torch.utils.data import Dataset, DataLoader
import mlflow
from src.utils.losses import choose_loss
from src.utils.params import Params

from torch.amp import autocast, GradScaler

    
class model_trainer:
    """ Class to handle model training and validation. """
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def train_1_epoch(self, training_loader, optimizer, loss_fn, device='cpu', use_amp=False):
        """ Function to train the model for one epoch.
            Args:
                model: The model to be trained.
                training_loader: DataLoader for the training data.
                optimizer: The optimizer to use for training.
                loss_fn: The loss function to use.
                device: Device to train on ('cpu' or 'cuda').
                use_amp: Whether to use Automatic Mixed Precision for faster training.
            
            Returns:
                Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        scaler = GradScaler(device=device, enabled=use_amp)
        
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, annotation = data
            inputs = inputs.to(device)
            annotation = annotation.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = self.model(inputs)
                loss = loss_fn(outputs, annotation)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            num_batches += 1
        
        return running_loss / num_batches if num_batches > 0 else 0.0

    def validate_1_epoch(self, validation_loader, loss_functions : dict[str, callable], device='cpu'):
        """ Function to validate the model for one epoch.
            Args:
                model: The model to be validated.
                validation_loader: DataLoader for the validation data.
                loss_functions: Dictionary of loss functions to use.
                device: Device to validate on ('cpu' or 'cuda').
            
            Returns:
                Dictionary of average validation losses.
        """
        self.model.eval()
        loss_list = {key: 0.0 for key in loss_functions.keys()}
        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                inputs, annotation = data
                inputs = inputs.to(device)
                annotation = annotation.to(device)
                outputs = self.model(inputs)
                for key, loss_fn in loss_functions.items():
                    loss = loss_fn(outputs, annotation)
                    loss_list[key] += loss.item()
        return {key: loss_list[key] / len(validation_loader) for key in loss_list.keys()}



    def train_model(self, train_dataset : Dataset, val_dataset: Dataset, parameters : Params):
        """ Function to train the model with given parameters. 
            Args:
                model: The model to be trained.
                train_dataset: Training dataset.
                val_dataset: Validation dataset.
                parameters: A Params object with training parameters. 
                (it should contain 'learning_rate', 'num_epochs', 'batch_size', etc.)
        """
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        print(f"Training on device: {device}")
        
        # Multi-GPU support with DataParallel (single machine, multiple GPUs)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        else:
            print("Using single GPU or CPU")
            self.model = torch.compile(self.model)
        
        # Optional: Compile model for faster execution (PyTorch 2.0+)
        # try:
        #     self.model = torch.compile(self.model, mode='reduce-overhead')
        #     print("Model compiled with torch.compile()")
        # except AttributeError:
        #     print("torch.compile not available (requires PyTorch 2.0+)")
        
        # Enable CuDNN autotuner for optimal performance
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            use_amp = parameters.get('use_amp', True)  # Use mixed precision by default on CUDA
        else:
            use_amp = False

        batch_size = parameters.get('batch_size', 16)
        num_workers = parameters.get('num_workers', 8)
        prefetch_factor = parameters.get('prefetch_factor', 2)
        pin_memory = device.type == 'cuda'

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=pin_memory, 
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,       # Prepare batches ahead of time
            persistent_workers=True)
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=pin_memory, 
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,       # Prepare batches ahead of time
            persistent_workers=True)

        learning_rate = parameters.get('learning_rate', 0.001)
        num_epochs = parameters.get('num_epochs', 10)
        optimizer_name = parameters.get('optimizer', 'AdamW')
        if(optimizer_name == 'AdamW'):
            momentum_1 = parameters.get('optimizer_momentum', 0.9)
            momentum_2 = parameters.get('optimizer_momentum2', 0.999)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(momentum_1, momentum_2))
        else:
            momentum = parameters.get('optimizer_momentum', 0.9)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        
        loss_fn = choose_loss(parameters.get('train_loss_function'))
        loss_functions = {loss : choose_loss(loss) for loss in parameters.get('validation_loss_function')}
        
        scheduler = parameters.get('scheduler', None)
        if(scheduler == 'StepLR'):
            step_size = parameters.get('scheduler_step_size', 7)
            gamma = parameters.get('scheduler_gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif(scheduler == 'ReduceLROnPlateau'):
            mode = parameters.get('scheduler_mode', 'min')
            factor = parameters.get('scheduler_factor', 0.1)
            patience = parameters.get('scheduler_patience', 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        else:
            scheduler = None
        
        # Early stopping parameters
        early_stopping_patience = parameters.get('early_stopping_patience', 15)
        early_stopping_min_delta = parameters.get('early_stopping_min_delta', 0.0)
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = parameters.get('best_model_path', 'best_model.pt')
        recent_model_path = parameters.get('recent_model_path', 'recent_model.pt')
        
        with mlflow.start_run() as run:
            # Log training parameters
            mlflow.log_params(parameters.get_all())
            
            for epoch in range(num_epochs):
                # Train for one epoch
                train_loss = self.train_1_epoch(train_loader, optimizer, loss_fn, device=device, use_amp=use_amp)
                
                # Validate for one epoch
                val_losses = self.validate_1_epoch(val_loader, loss_functions, device=device)
                val_loss = val_losses[list(val_losses.keys())[0]]  # Use first loss for early stopping
                
                # Update scheduler
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Log metrics to MLflow
                mlflow.log_metrics({"train_loss": train_loss}, step=epoch)
                mlflow.log_metrics({f"val_{key}": val_losses[key] for key in val_losses.keys()}, step=epoch)
                
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Losses: {val_losses}")
                
                # Early stopping logic
                if val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"  -> Best model saved with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs (patience: {early_stopping_patience})")
                        # Load best model
                        self.model.load_state_dict(torch.load(best_model_path))
                        mlflow.log_param("early_stopping_epoch", epoch+1)
                        break

                # Save the most recent model
                torch.save(self.model.state_dict(), recent_model_path)
                print(f"Most recent model saved to {recent_model_path}")
        print("Training complete.")
    
    def get_model(self):
        """ Function to get the trained model.
            Returns:
                The trained model.
        """
        return self.model