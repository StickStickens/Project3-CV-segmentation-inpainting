import json

class Params:
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
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

def merge_params(params1: Params, params2: Params) -> Params:
    merged_params = Params.__new__(Params)
    merged_params.params = {**params1.params, **params2.params}
    return merged_params