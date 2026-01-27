import copy
import torch
from torch.utils.data import Subset
import optuna

from ..dataset.dataset import get_random_subset
from ..utils.params import Params
from ..utils.losses import choose_loss
from ..train import model_trainer


def _params_from_dict(d: dict) -> Params:
    p = Params()
    p.params = dict(d)
    return p


def estimate_hyperparameters(model, model_params_path, dataset, n_trials=20, sample_size=100,
                             epochs = None):
    """Estimate hyperparameters using Optuna on a random subset of the dataset.

    - Properly split subset using Subset indices.
    - Reset model weights each trial to avoid leakage.
    - Tune valid loss hyperparameters (ce/dice weights, label smoothing) and weight decay.
    """

    subset = get_random_subset(dataset, sample_size)
    idx = torch.randperm(len(subset))
    train_split = int(0.8 * len(subset))
    train_subset = Subset(subset, idx[:train_split].tolist())
    val_subset = Subset(subset, idx[train_split:].tolist())

    base_params = Params(model_params_path).get_all()

    # Snapshot initial weights so each trial starts fresh
    initial_state = copy.deepcopy(model.state_dict())

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.1)

        trial_params = dict(base_params)
        trial_params.setdefault('train_loss_function', 'CrossEntropyLoss')
        trial_params.setdefault('validation_loss_functions', ['CrossEntropyLoss'])
        trial_params.update({
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'train_loss_params': {

                'label_smoothing': label_smoothing,
            },
            'validation_loss_params': {
                'label_smoothing': label_smoothing,
            },
        })
        if(epochs is not None):
            trial_params.update({'num_epochs': epochs})  # Keep trials short

        trial_model = copy.deepcopy(model)
        trial_model.load_state_dict(initial_state)

        params_obj = _params_from_dict(trial_params)
        trainer = model_trainer(trial_model)
        trainer.train_model(train_subset, val_subset, params_obj, optuna_trial=trial, save_model=False)

        loss_fn = {'CrossEntropyLoss': choose_loss('CrossEntropyLoss', {
            'label_smoothing': label_smoothing,
        })}
        device = torch.device(params_obj.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        val_metrics = trainer.validate_1_epoch(loss_fn, device=device, num_classes=params_obj.get('num_classes', 151))
        return val_metrics['CrossEntropyLoss']

    # Use MedianPruner: stops trials performing worse than median of previous trials
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Don't prune first 5 trials (need baseline)
        n_warmup_steps=3,    # Wait 3 epochs before pruning each trial
        interval_steps=1      # Check pruning every epoch
    )
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best hyperparameters: ", study.best_params)

    updated = dict(base_params)
    updated.update({
        'learning_rate': study.best_params['lr'],
        'weight_decay': study.best_params['weight_decay'],
        'train_loss_params': {
            'label_smoothing': study.best_params['label_smoothing'],
        },
        'validation_loss_params': {
            'label_smoothing': study.best_params['label_smoothing'],
        },
    })
    _params_from_dict(updated).save(model_params_path)
    return study.best_params