import json
from pathlib import Path


def _project_root() -> Path:
    """Return the absolute path to the project root (two levels above utils)."""
    return Path(__file__).resolve().parents[2]


def _resolve_param_path(filepath: str | Path) -> Path:
    """Resolve a params file to an absolute path anchored at the project root."""
    path = Path(filepath)
    if path.is_absolute():
        return path

    root = _project_root()
    candidates = [root / path, root / 'src' / 'config' / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fall back to the first candidate even if missing; let caller raise.
    return candidates[0]

class Params:
    def __init__(self, filepath = None):
        if filepath is not None:
            resolved_path = _resolve_param_path(filepath)
            with open(resolved_path, 'r') as f:
                params = json.load(f)
        else:
            params = {}
        self.params = params
    def get(self, key, default=None):
        return self.params.get(key, default)
    
    def set(self, key, value):
        self.params[key] = value

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.params, f, indent=4)

    def get_all(self):
        return self.params
    
    @staticmethod
    def overwrite_params(overwritten: 'Params', overwrite: 'Params') -> None:
        """Overwrite keys in 'overwritten' with values from 'overwrite'.
        Updates in place; does not return a new Params.
        """
        overwritten.params.update(overwrite.get_all())

def merge_params(params1: Params, params2: Params) -> Params:
    merged_params = Params.__new__(Params)
    merged_params.params = {**params1.params, **params2.params}
    return merged_params

def integrate_global_parameters(params : Params):
    merged_params = Params.__new__(Params)
    global_params = Params(_resolve_param_path('global_params.json'))
    merged_params.params = {**params.params, **global_params.params}
    return merged_params