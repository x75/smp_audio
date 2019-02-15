
# from util
def args_to_dict(args):
    cls=dict
    # l.info('geconf.from_args(args = {0})'.format(args))
    args_attrs = [attr_ for attr_ in dir(args) if not attr_.startswith('_')]
    args_attr_vals = dict([(attr_, getattr(args, attr_)) for attr_ in dir(args) if not attr_.startswith('_')])
    # l.info('geconf.from_args: dir(args) = {0})'.format(args_attr_vals))
    return cls(**args_attr_vals)

