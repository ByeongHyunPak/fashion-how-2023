import copy

networks = {}

def register(name):
    def decorator(cls):
        networks[name] = cls
        return cls
    return decorator

def make(model_spec, args=None):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    return networks[model_spec['name']](**model_args)