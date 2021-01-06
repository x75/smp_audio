# caching joblib
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)
