class JassorDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for arg in [a for a in args if hasattr(a, 'keys')]:
            super().update(arg)
        super().update([a for a in args if not hasattr(a, 'keys')])
        super().update(kwargs)

    def __getattr__(self, arg_name):
        return self[arg_name]

    def __getitem__(self, item):
        return super().__getitem__(item) if item in self else None

    def __setattr__(self, arg_name, arg_value):
        self[arg_name] = arg_value

    def __setitem__(self, item, value):
        return super().__setitem__(item, value)
