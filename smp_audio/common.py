"""smp_audio.common

common data, embedding, transform functions for offline music and
multivariate timeseries processing

Oswald Berthold, 2018
"""

import logging

DEBUG=True

def myprint(*args, **kwargs):
    if not DEBUG: return
    print(*args, **kwargs)

def get_module_logger(modulename = 'experiment', loglevel = logging.INFO):
    """get a logging.logger instance with reasonable defaults

    Create a new logger and configure its name, loglevel, formatter
    and output stream handling.
    1. initialize a logger with name from arg 'modulename'
    2. set loglevel from arg 'loglevel'
    3. configure matching streamhandler
    4. set formatting swag
    5. return the logger
    """
    loglevels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warn': logging.WARNING}
    if type(loglevel) is str:
        try:
            loglevel = loglevels[loglevel]
        except:
            loglevel = logging.INFO
            
    if modulename.startswith('smp_graphs'):
        modulename = '.'.join(modulename.split('.')[1:])
        
    if len(modulename) > 20:
        modulename = modulename[-20:]
    
    # create logger
    logger = logging.getLogger(modulename)
    logger.setLevel(loglevel)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)

    # create formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(levelname)8s: %(name)20s: %(message)s')
    # formatter = logging.Formatter('{levelname:8}s: %(name)20s: %(message)s')
    # formatter = logging.Formatter('%(name)s: %(levelname)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    # suppress double log output 
    logger.propagate = False
    return logger
