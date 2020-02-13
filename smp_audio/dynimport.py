"""dynamic module import

.. note:: see `smp_base.impl.smpi` which already does this

goal is to have

- dynamic imports
- support a stack of equivalent libraries with similar function sets
- load them according to availability and preference

approaches

- importlib
- try / except / HAVE_THATLIB
- imp, deprecated in favor of importlib
- __import__
"""

import os
import sys
import inspect
import pkgutil
import importlib
from importlib import import_module
from pathlib import Path

def load_from_file(filepath):
    import imp
    class_inst = None
    expected_class = 'MyClass'

    mod_name,file_ext = os.path.splitext(os.path.split(filepath)[-1])

    if file_ext.lower() == '.py':
        py_mod = imp.load_source(mod_name, filepath)

    elif file_ext.lower() == '.pyc':
        py_mod = imp.load_compiled(mod_name, filepath)

    if hasattr(py_mod, expected_class):
        class_inst = getattr(py_mod, expected_class)()

    return class_inst

def load_lib_simple(action):
    action_ = importlib.import_module('actions.'+ action)
    return action_

def load_lib_simple_2():
    for (_, name, _) in pkgutil.iter_modules([os.path.dirname(__file__)]):
        imported_module = import_module('.' + name, package='animals.fish')

        class_name = list(filter(lambda x: x != 'AnimalBaseClass' and not x.startswith('__'), 
    			     dir(imported_module)))

        fish_class = getattr(imported_module, class_name[0])

        if issubclass(fish_class, AnimalBaseClass):
            setattr(sys.modules[__name__], name, fish_class)

def load_lib_from_file():
    """load lib from file

    from https://docs.python.org/3/library/importlib.html#examples
    """
    import importlib.util
    import sys

    # For illustrative purposes.
    import tokenize
    file_path = tokenize.__file__
    module_name = tokenize.__name__
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Optional; only necessary if you want to be able to import the module
    # by name later.
    sys.modules[module_name] = module

def load_libs(**kwargs):
    """load_libs

    Load libraries for project
    - iter list of strings
    - return dict of names
    """
    if 'libs' in kwargs:
        libs_to_load = kwargs['libs']
    else:
        # default libs to load
        libs_to_load = {
            'librosa': None, 'aubio': None, 'essentia': None,
            'essentia.standard': None, 'madmom': None,
            'blub': None,
        }
        
    # libs loaded successfully
    libs_loaded = {}

    # iterate libs to load
    for lib_to_load in libs_to_load:
        print('Trying to load {0}'.format(lib_to_load))
        # try to load lib
        try:
            lib_loaded_ = import_module(lib_to_load)
            print('Loaded {0} from {1}'.format(lib_loaded_, lib_to_load))
        # catch fail non-fatal
        except ImportError as err:
            print('Failed to load {0} from {1}'.format(lib_loaded_, lib_to_load))
            # option: at lib_to_load insert None 
            # option: don't insert lib_to_load key

        # debug
        print('Namespace    dir() = {0}'.format(dir()))

        # insert name into libs_loaded dict
        print('Inserting {0} into locals'.format(lib_to_load))
        # locals()[lib_to_load] = lib_loaded_
        libs_loaded[lib_to_load] = lib_loaded_
        print('Namespace locals() = {0}'.format(dir()))

    # return dict libname:loaded_module
    return libs_loaded
        
if __name__ == '__main__':

    # load_lib_simple()
    # load_lib_simple_2()
    
    locals().update(load_libs())

    print('__main__.Namespace    dir() = {0}'.format(dir()))
    print('__main__.Namespace locals() = {0}'.format(dir()))

    if 'aubio' in locals():
        print('aubio', aubio)
