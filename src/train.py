import platform
import optuna
import torch
from torch.utils.data import Dataset, DataLoader
import mlflow
from src.utils.losses import choose_loss
from src.utils.params import Params, integrate_global_parameters
from src.utils.params import Params, integrate_global_parameters

from torch.amp import autocast, GradScaler

    
class model_trainer:
    """ Class to handle model training and validation. """
    def __init__(self, model: torch.nn.Module, accumulation_steps: int = 1):
        self.model = model
        self.train_loader : DataLoader = None
        self.validation_loader : DataLoader = None
        self.accumulation_steps = accumulation_steps

    def train_1_epoch(self, optimizer, loss_fn, device='cpu', use_amp=False):
    def train_1_epoch(self, optimizer, loss_fn, device='cpu', use_amp=False):
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
        
        for i, data in enumerate(self.train_loader):
        for i, data in enumerate(self.train_loader):
            # Every data instance is an input + label pair
            inputs = data['image'].to(device)
            annotation = data['annotation'].to(device)

            # Mixed precision forward pass
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = self.model(inputs)
                loss = loss_fn(outputs, annotation)
                # Scale loss by accumulation steps to keep gradient magnitude consistent
                loss = loss / self.accumulation_steps
            
            running_loss += loss.item()
            # Backward pass with scaling
            scaler.scale(loss).backward()
            num_batches += 1
            
            # Optimizer step only every accumulation_steps batches
            if (i + 1) % self.accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        return running_loss / num_batches if num_batches > 0 else 0.0

    def validate_1_epoch(self, loss_functions : dict[str, callable], device='cpu', num_classes: int = 151):
    def validate_1_epoch(self, loss_functions : dict[str, callable], device='cpu', num_classes: int = 151):
        """ Function to validate the model for one epoch.
            Args:
                model: The model to be validated.
                validation_loader: DataLoader for the validation data.
                loss_functions: Dictionary of loss functions to use.
                device: Device to validate on ('cpu' or 'cuda').
                num_classes: Number of classes for mIoU computation.
                num_classes: Number of classes for mIoU computation.
            
            Returns:
                Dictionary of average validation losses and metrics.
                Dictionary of average validation losses and metrics.
        """
        from src.utils.metrics import compute_miou, compute_pixel_accuracy
        
        from src.utils.metrics import compute_miou, compute_pixel_accuracy
        
        self.model.eval()
        loss_list = {key: 0.0 for key in loss_functions.keys()}
        total_miou = 0.0
        total_pixel_acc = 0.0
        num_batches = 0
        
        total_miou = 0.0
        total_pixel_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i, data in enumerate(self.validation_loader):
                inputs = data['image'].to(device)
                annotation = data['annotation'].to(device)
            for i, data in enumerate(self.validation_loader):
                inputs = data['image'].to(device)
                annotation = data['annotation'].to(device)
                outputs = self.model(inputs)
                
                # Compute losses
                
                # Compute losses
                for key, loss_fn in loss_functions.items():
                    loss = loss_fn(outputs, annotation)
                    loss_list[key] += loss.item()
                
                # Compute metrics
                miou, _ = compute_miou(outputs, annotation, num_classes)
                pixel_acc = compute_pixel_accuracy(outputs, annotation)
                total_miou += miou
                total_pixel_acc += pixel_acc
                num_batches += 1
        
        # Average losses and metrics
        results = {key: loss_list[key] / num_batches for key in loss_list.keys()}
        results['mIoU'] = total_miou / num_batches if num_batches > 0 else 0.0
        results['pixel_accuracy'] = total_pixel_acc / num_batches if num_batches > 0 else 0.0
        
        return results



    def train_model(self, train_dataset : Dataset, val_dataset: Dataset, parameters : Params, optuna_trial=None):
        """ Function to train the model with given parameters. 
            Args:
                model: The model to be trained.
                train_dataset: Training dataset.
                val_dataset: Validation dataset.
                parameters: A Params object with training parameters. 
                (it should contain 'learning_rate', 'num_epochs', 'batch_size', etc.)
                optuna_trial: Optional Optuna trial for pruning support.
        """

        parameters = integrate_global_parameters(parameters)
        # Device setup
        device = torch.device(parameters.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        device = torch.device(parameters.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(device)
        print(f"Training on device: {device}")
        
        # Multi-GPU support with DataParallel (single machine, multiple GPUs)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        else:
            print("Using single GPU or CPU")
            # Optional: Compile model for faster execution (PyTorch 2.0+)
            if device.type == 'cuda' and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    print("Model compiled with torch.compile()")
                except Exception as e:
                    print(f"torch.compile failed: {e}")
            # Optional: Compile model for faster execution (PyTorch 2.0+)
            if device.type == 'cuda' and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    print("Model compiled with torch.compile()")
                except Exception as e:
                    print(f"torch.compile failed: {e}")
        
        #dataloader setup
        #dataloader setup
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            use_amp = parameters.get('use_amp', True)  # Use mixed precision by default on CUDA
        else:
            use_amp = False
        batch_size = parameters.get('batch_size', 16)
        num_workers = parameters.get('num_workers', 8)
        prefetch_factor = parameters.get('prefetch_factor', 2)
        pin_memory = device.type == 'cuda'
        print(f"DataLoader settings - batch_size: {batch_size}, num_workers: {num_workers}, pin_memory: {pin_memory}, prefetch_factor: {prefetch_factor}")
        # Windows often has issues with multiprocessing; fallback to 0 workers
        if platform.system() == 'Windows' and num_workers > 0:
            print("Warning: On Windows, using num_workers=0 to avoid multiprocessing issues")
            num_workers = 0
            prefetch_factor = None  # Not allowed when num_workers=0
            persistent_workers = False
        else:
            persistent_workers = True if num_workers > 0 else False

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=pin_memory, 
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,       # Prepare batches ahead of time
            persistent_workers=persistent_workers)
            persistent_workers=persistent_workers)
        
        self.validation_loader = DataLoader(
        self.validation_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=pin_memory, 
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,       # Prepare batches ahead of time
            persistent_workers=persistent_workers)
        
        # Optimizer setup
            persistent_workers=persistent_workers)
        
        # Optimizer setup
        learning_rate = parameters.get('learning_rate', 0.001)
        num_epochs = parameters.get('num_epochs', 10)
        optimizer_name = parameters.get('optimizer', 'AdamW')
        if(optimizer_name == 'AdamW'):
            momentum_1 = parameters.get('optimizer_momentum', 0.9)
            momentum_2 = parameters.get('optimizer_momentum2', 0.999)
            weight_decay = parameters.get('weight_decay', 1e-4)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(momentum_1, momentum_2), weight_decay=weight_decay)
        else:
            momentum = parameters.get('optimizer_momentum', 0.9)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        # loss function setup
        loss_fn = choose_loss(parameters.get('train_loss_function'), parameters.get('train_loss_params', {}))
        loss_functions = {loss : choose_loss(loss, parameters.get('validation_loss_params', {})) for loss in parameters.get('validation_loss_functions', ['CrossEntropyLoss'])}
        # to verify model accuracy during validation, we can use the first validation loss function
        veryfying_loss_function = parameters.get('validation_loss_functions', ['CrossEntropyLoss'])[0]
        # Learning rate scheduler setup
        # loss function setup
        loss_fn = choose_loss(parameters.get('train_loss_function'), parameters.get('train_loss_params', {}))
        loss_functions = {loss : choose_loss(loss, parameters.get('validation_loss_params', {})) for loss in parameters.get('validation_loss_functions', ['CrossEntropyLoss'])}
        # to verify model accuracy during validation, we can use the first validation loss function
        veryfying_loss_function = parameters.get('validation_loss_functions', ['CrossEntropyLoss'])[0]
        # Learning rate scheduler setup
        scheduler = parameters.get('scheduler', None)
        if(scheduler == 'StepLR'):
            step_size = parameters.get('scheduler_step', 7)
            step_size = parameters.get('scheduler_step', 7)
            gamma = parameters.get('scheduler_gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif(scheduler == 'ReduceLROnPlateau'):
            mode = parameters.get('scheduler_mode', 'min')
            factor = parameters.get('scheduler_factor', 0.1)
            patience = parameters.get('scheduler_patience', 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        elif(scheduler == 'CosineWarmup'):
            # Simple cosine schedule with linear warmup.
            # Warmup increases LR from start_factor * base_lr to base_lr over warmup epochs.
            warmup_epochs = parameters.get('scheduler_warmup_epochs', max(1, num_epochs // 10))
            warmup_start_factor = parameters.get('scheduler_warmup_start_factor', 0.1)
            eta_min = parameters.get('scheduler_eta_min', 0.0)

            # Ensure valid durations
            warmup_epochs = max(1, min(warmup_epochs, max(1, num_epochs - 1)))
            cosine_epochs = max(1, num_epochs - warmup_epochs)

            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_start_factor, total_iters=warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            scheduler = None
        
        # Early stopping parameters
        early_stopping_patience = parameters.get('early_stopping_patience', 15)
        early_stopping_min_delta = parameters.get('early_stopping_min_delta', 0.0)
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = parameters.get('best_model_path', 'best_model.pt')
        recent_model_path = parameters.get('recent_model_path', 'recent_model.pt')
        
        # Get num_classes for mIoU computation
        num_classes = parameters.get('num_classes', 151)  # ADE20K has 150 classes + background
        
        # Get num_classes for mIoU computation
        num_classes = parameters.get('num_classes', 151)  # ADE20K has 150 classes + background
        
        with mlflow.start_run() as run:
            # Log training parameters
            mlflow.log_params(parameters.get_all())
            
            for epoch in range(num_epochs):
                # Train for one epoch
                train_loss = self.train_1_epoch(optimizer, loss_fn, device=device, use_amp=use_amp)
                train_loss = self.train_1_epoch(optimizer, loss_fn, device=device, use_amp=use_amp)
                
                # Validate for one epoch (includes mIoU and pixel accuracy)
                val_metrics = self.validate_1_epoch(loss_functions, device=device, num_classes=num_classes)
                val_loss = val_metrics[veryfying_loss_function]  # Use first loss for early stopping
                # Validate for one epoch (includes mIoU and pixel accuracy)
                val_metrics = self.validate_1_epoch(loss_functions, device=device, num_classes=num_classes)
                val_loss = val_metrics[veryfying_loss_function]  # Use first loss for early stopping
                
                # Update scheduler
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Log metrics to MLflow
                mlflow.log_metrics({"train_loss": train_loss}, step=epoch)
                mlflow.log_metrics({f"val_{key}": val_metrics[key] for key in val_metrics.keys()}, step=epoch)
                mlflow.log_metrics({f"val_{key}": val_metrics[key] for key in val_metrics.keys()}, step=epoch)
                
                # Optuna pruning: report intermediate value and check if trial should be pruned
                if optuna_trial is not None:
                    optuna_trial.report(val_loss, epoch)
                    if optuna_trial.should_prune():
                        print(f"Trial pruned at epoch {epoch+1} (val_loss: {val_loss:.4f})")
                        raise optuna.TrialPruned()
                
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
                print(f"  Validation: {val_metrics}")
                
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