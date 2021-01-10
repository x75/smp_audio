"""smp_audio.utils

Utility lib
"""
import argparse
import json
import numpy as np

# from smp_base util
def args_to_dict(args):
    cls=dict
    # l.info('geconf.from_args(args = {0})'.format(args))
    # args_attrs = [attr_ for attr_ in dir(args) if not attr_.startswith('_')]
    args_attr_vals = dict([(attr_, getattr(args, attr_)) for attr_ in dir(args) if not attr_.startswith('_')])
    # l.info('geconf.from_args: dir(args) = {0})'.format(args_attr_vals))
    return cls(**args_attr_vals)

# convert from argparse.Namespace to kwargs dictionary
def ns2kw(ns):
    kw = dict([(_, getattr(ns, _)) for _ in dir(ns) if not _.startswith('_')])
    return kw

# convert from kwargs dictionary to argparse.Namespace
def kw2ns(kw):
    ns = argparse.Namespace()
    for k in kw:
        setattr(ns, k, kw[k])
    return ns

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

