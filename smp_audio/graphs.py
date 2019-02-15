from collections import OrderedDict
from pprint import pformat

def graph_walk_dict_flat(indict, pre=None):
    """walk nested dictionary return flattended
    """
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, [key] + pre):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, [key] + pre):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield indict

def graph_walk_collection_flat(indict, pre=None):
    """walk nested collections return flattended
    """
    pre = pre[:] if pre else []
    
    # if isinstance(indict, dict):
    if type(indict) in [dict, OrderedDict]:
        for key, value in indict.items():
            # if isinstance(value, dict):
            if type(value) in [dict, OrderedDict]:
                for d in graph_walk_collection_flat(value, [key] + pre):
                    yield d
            # elif isinstance(value, list) or isinstance(value, tuple):
            elif type(value) in [list, tuple]:
                for v in value:
                    for d in graph_walk_collection_flat(v, [key] + pre):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield indict

def cb_graph_walk_print(key, item, level, **cb_args):
    # print('{1}{0}'.format(key, ' ' * (level * 4)))
    print('{2}{0}: {1}'.format(key, item, ' ' * (level * 4)))
        
def cb_graph_walk_build_graph(key, item, level, **cb_args):
    print('{2}{0}: {1}'.format(key, item, ' ' * (level * 4)))
    G = cb_args['G']
    # P = cb_args['P']

    # # if cb_args['P'] is None:
    # if level == 0:
    #     # root node
    #     # G.add_node('root')
    #     cb_args['P'] = 'root'

    # if type(key) is function:
    if hasattr(key, '__call__'):
        key = key.__name__
    if hasattr(item, '__call__'):
        item = item.__name__
    
    G.add_node(key)
    # link current node with parent unless root
    G.add_edge(cb_args['P'], key, key='P')
        
    if item is not None:
        # cb_args['P'] = key
        # item_str = str(item)[:20] #.replace('[', 'arr')
        # item_str = repr(item)[:20] #.replace('[', 'arr')
        item_str = "blub"
        G.add_node(item_str)
        G.add_edge(key, item_str, key='P')
        # G.add_node('{0}-{1}'.format(key, 'G'))
    else:
        cb_args['P'] = key
        
    # if cb_args['P'] is not None:

    print('cb_args', pformat(cb_args))
    return cb_args

def graph_walk_collection(item, level=0, cb=None, **cb_args):
    if cb is None:
        cb = cb_graph_walk_print
        
    # if isinstance(item, collections.Container):
    assert type(item) in [dict, OrderedDict], "item needs to be of type dict"
    for key_, item_ in item.items():
        if type(item_) in [dict, OrderedDict]:
            cb_args_ = cb(key_, None, level, **cb_args)
            graph_walk_collection(item_, level=level+1, cb=cb, **cb_args_)
        else:
            cb(key_, item_, level, **cb_args)
    # print('cb_args', pformat(cb_args))
    return cb_args
