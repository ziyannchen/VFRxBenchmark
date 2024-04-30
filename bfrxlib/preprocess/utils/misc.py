import shutil
import os

def makedir(dir_path, rebuild: bool =True):
    '''
    To make a directory of the given path.

    dir_path: tuple or str
    rebuild: delete the old directory and create a new one if rebuild is True
    '''
    if isinstance(dir_path, tuple):
        dir_path = os.path.join(*dir_path)

    if rebuild and os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def append_list_to_dict(dict, key, value):
    if key not in dict.keys():
        dict[key] = []
    dict[key].append(value)
    return dict

class obj(object):
    '''
    objectify a dict into a class.
    '''
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)