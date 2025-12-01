from yaml import load, dump
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import sys

def getParams():
    cfgfilename = sys.argv[1]
    cfgfile = open(cfgfilename, 'r')

    params = load(cfgfile, Loader=Loader)

    return params


def saveParams(params, file):
    with open(file, 'w') as yamlfile:
        data = dump(params, yamlfile)
