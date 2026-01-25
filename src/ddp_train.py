import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import mlflow
from src.utils.losses import choose_loss
from src.utils.params import Params, integrate_global_parameters

from torch.amp import autocast, GradScaler


def setup_ddp(rank: int, world_size: int):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    # Use a random-ish port to avoid conflicts; can be overridden via env var
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def _ddp_training_worker(rank: int, world_size: int, model_class, model_kwargs: dict,
                         dataset_class, train_dataset_kwargs: dict, val_dataset_kwargs: dict,
                         parameters_dict: dict, output_model_path: str):
    """
    Worker function for DDP training. Each process runs this independently.
    """
    try:
        # Setup DDP
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        
        # Create model on this GPU
        model = model_class(**model_kwargs).to(device)
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        # Recreate datasets in each worker (avoids pickling issues)
        train_dataset = dataset_class(**train_dataset_kwargs)
        val_dataset = dataset_class(**val_dataset_kwargs)
        
        # Convert parameters dict back to Params object
        parameters = Params()
        parameters.params = parameters_dict
        
        # Create samplers for distributed training
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        # Training setup from parameters
        parameters = integrate_global_parameters(parameters)
        batch_size = parameters.get('batch_size', 16)
        num_workers = parameters.get('num_workers', 4)
        prefetch_factor = parameters.get('prefetch_factor', 2)
        learning_rate = parameters.get('learning_rate', 0.001)
        num_epochs = parameters.get('num_epochs', 10)
        use_amp = parameters.get('use_amp', True)
        num_classes = parameters.get('num_classes', 151)
        early_stopping_patience = parameters.get('early_stopping_patience', 15)
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor,
            persistent_workers=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor,
            persistent_workers=False
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate,
            weight_decay=parameters.get('weight_decay', 1e-4)
        )
        
        # Loss function
        loss_fn = choose_loss(
            parameters.get('train_loss_function', 'CrossEntropyLoss'),
            parameters.get('train_loss_params', {})
        )
        
        # Scheduler
        scheduler = None
        scheduler_name = parameters.get('scheduler', None)
        if scheduler_name == 'CosineWarmup':
            warmup_epochs = max(1, num_epochs // 10)
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
        elif scheduler_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=parameters.get('scheduler_step', 7),
                gamma=parameters.get('scheduler_gamma', 0.1)
            )
        
        # Training loop
        scaler = GradScaler(enabled=use_amp)
        best_val_loss = float('inf')
        patience_counter = 0
        torch.backends.cudnn.benchmark = True
        
        # Start MLflow run on rank 0 only
        mlflow_run = None
        if rank == 0:
            mlflow_run = mlflow.start_run()
            mlflow.log_params(parameters_dict)
        
        try:
            for epoch in range(num_epochs):
                train_sampler.set_epoch(epoch)
                
                model.train()
                train_loss_sum = torch.tensor(0.0, device=device)
                train_batches = torch.tensor(0, device=device)
                
                for batch in train_loader:
                    images = batch['image'].to(device, non_blocking=True)
                    masks = batch['annotation'].to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    with autocast(device_type='cuda', enabled=use_amp):
                        outputs = model(images)
                        loss = loss_fn(outputs, masks)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss_sum += loss.detach()
                    train_batches += 1
                
                # Average train loss across all GPUs
                dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_batches, op=dist.ReduceOp.SUM)
                train_loss = (train_loss_sum / train_batches).item()
                
                # validation
                model.eval()
                val_loss_sum = torch.tensor(0.0, device=device)
                val_batches = torch.tensor(0, device=device)
                
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['image'].to(device, non_blocking=True)
                        masks = batch['annotation'].to(device, non_blocking=True)
                        with autocast(device_type='cuda', enabled=use_amp):
                            outputs = model(images)
                            loss = loss_fn(outputs, masks)
                        val_loss_sum += loss.detach()
                        val_batches += 1
                
                # Average validation loss across all GPUs
                dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_batches, op=dist.ReduceOp.SUM)
                val_loss = (val_loss_sum / val_batches).item()
                
                # Scheduler step
                if scheduler is not None:
                    scheduler.step()
                
                # Logging and saving (rank 0 only)
                if rank == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    }, step=epoch)
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.module.state_dict(), output_model_path)
                        print(f"  -> Best model saved (val_loss: {val_loss:.4f})")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                            mlflow.log_param("early_stopping_epoch", epoch + 1)
                            break
                
                # Broadcast early stopping decision to all ranks
                should_stop = torch.tensor([patience_counter >= early_stopping_patience], device=device)
                dist.broadcast(should_stop, src=0)
                if should_stop.item():
                    break
            
            if rank == 0:
                mlflow.log_metric("best_val_loss", best_val_loss)
                print("DDP Training complete!")
        
        finally:
            # End MLflow run
            if mlflow_run is not None:
                mlflow.end_run()
            
    finally:
        cleanup_ddp()


def train_with_ddp_kaggle(model_class, model_kwargs: dict,
                          dataset_class, train_dataset_kwargs: dict, val_dataset_kwargs: dict,
                          parameters: Params, num_gpus: int = None,
                          output_model_path: str = './models/best_ddp_model.pt') -> str:
    """
    Kaggle-compatible DDP training using mp.spawn optimized for speed and stability.
    
    This function spawns multiple processes (one per GPU) and coordinates
    distributed training.
    
    Args:
        model_class: Model class (e.g., UNet)
        model_kwargs: Dict of kwargs for model constructor
        dataset_class: Dataset class (e.g., ADE20KDataset)
        train_dataset_kwargs: Dict of kwargs for training dataset
        val_dataset_kwargs: Dict of kwargs for validation dataset
        parameters: Params object with training config
        num_gpus: Number of GPUs (default: all available)
        output_model_path: Where to save the trained model
    
    Returns:
        Path to the saved model weights
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)
    
    if num_gpus < 2:
        print(f"Only {num_gpus} GPU(s) available")
        raise RuntimeError("DDP training requires at least 2 GPUs.")
    
    print(f"Starting DDP training on {num_gpus} GPUs...")
    
    # Convert Params to dict for pickling
    parameters_dict = parameters.get_all()
    
    # Set multiprocessing start method (required for CUDA)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Spawn workers
    mp.spawn(
        _ddp_training_worker,
        args=(num_gpus, model_class, model_kwargs,
              dataset_class, train_dataset_kwargs, val_dataset_kwargs,
              parameters_dict, output_model_path),
        nprocs=num_gpus,
        join=True
    )
    
    print(f"DDP training complete. Model saved to: {output_model_path}")
    return output_model_path