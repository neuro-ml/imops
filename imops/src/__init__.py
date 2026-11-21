import sys
from importlib import import_module
from pkgutil import iter_modules

from ._utils import supports_x86_64_v3


# `from .src._zoom import ...` then reads the `_zoom_v3` build, which only runs on x86-64-v3 CPUs
if supports_x86_64_v3():
    for module in iter_modules(__path__):
        if module.name.endswith('_v3'):
            sys.modules[f'{__name__}.{module.name[:-3]}'] = import_module(f'{__name__}.{module.name}')
