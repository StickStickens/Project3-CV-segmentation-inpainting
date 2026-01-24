from ..dataset.dataset import get_random_subset
from ..utils.params import Params
from ..utils.losses import choose_loss
from ..train import model_trainer
import optuna


def estimate_hyperparameters(model, model_params_path,dataset, n_trials=20, sample_size=100):
    """Estimate hyperparameters using Optuna on a random subset of the dataset."""
    subset = get_random_subset(dataset, sample_size)
    train_split = int(0.8 * len(subset))
    train_subset = subset[:train_split]
    val_subset = subset[train_split:]
    known_params = Params(model_params_path)
    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        loss_gamma = trial.suggest_uniform('loss_gamma', 0.2, 1.0)
        # Update known params with trial suggestions
        trial_params : dict = known_params.to_dict()
        trial_params.update({
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'train_loss_params': {'gamma': loss_gamma},
            'validation_loss_params': {'gamma': loss_gamma}
        })
        # Here you would train your model on the subset and return validation loss
        trainer = model_trainer(model)
        trainer.train_model(train_subset, val_subset, Params(trial_params))
        val_loss = trainer.validate_1_epoch(val_subset, {'CombinedLoss': choose_loss('CombinedLoss', {'gamma': loss_gamma})})
        return val_loss['CombinedLoss']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters: ", study.best_params)
    known_params.update_from_dict(study.best_params)
    known_params.save(model_params_path)
    return study.best_params