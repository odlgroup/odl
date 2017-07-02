# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import odl
import inspect
import importlib


__all__ = ('make_interface',)

module_string = """
.. rubric:: Modules

.. toctree::
   :maxdepth: 2

   {}
"""

fun_string = """
.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   {}
"""

class_string = """
.. rubric:: Classes

.. autosummary::
   :toctree: generated/

   {}
"""

string = """{shortname}
{line}

{docstring}

.. currentmodule:: {name}

{module_string}
{class_string}
{fun_string}
"""


def import_submodules(package, name=None, recursive=True):
    """Import all submodules of ``package``.

    Parameters
    ----------
    package : `module` or string
        Package whose submodules to import.
    name : string, optional
        Override the package name with this value in the full
        submodule names. By default, ``package`` is used.
    recursive : bool, optional
        If ``True``, recursively import all submodules of ``package``.
        Otherwise, import only the modules at the top level.

    Returns
    -------
    pkg_dict : dict
        Dictionary where keys are the full submodule names and values
        are the corresponding module objects.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)

    if name is None:
        name = package.__name__

    submodules = [m[0] for m in inspect.getmembers(package, inspect.ismodule)
                  if m[1].__name__.startswith('odl')]

    results = {}
    for pkgname in submodules:
        full_name = name + '.' + pkgname
        try:
            results[full_name] = importlib.import_module(full_name)
        except ImportError:
            pass
        else:
            if recursive:
                results.update(import_submodules(full_name, full_name))
    return results


def make_interface():
    """Generate the RST files for the API doc of ODL."""
    modnames = ['odl'] + list(import_submodules(odl).keys())

    for modname in modnames:
        if not modname.startswith('odl'):
            modname = 'odl.' + modname

        shortmodname = modname.split('.')[-1]
        print('{: <25} : generated {}.rst'.format(shortmodname, modname))

        line = '=' * len(shortmodname)

        module = importlib.import_module(modname)

        docstring = module.__doc__
        submodules = [m[0] for m in inspect.getmembers(
            module, inspect.ismodule) if m[1].__name__.startswith('odl')]
        functions = [m[0] for m in inspect.getmembers(
            module, inspect.isfunction) if m[1].__module__ == modname]
        classes = [m[0] for m in inspect.getmembers(
            module, inspect.isclass) if m[1].__module__ == modname]

        docstring = '' if docstring is None else docstring

        submodules = [modname + '.' + mod for mod in submodules]
        functions = ['~' + modname + '.' + fun
                     for fun in functions if not fun.startswith('_')]
        classes = ['~' + modname + '.' + cls
                   for cls in classes if not cls.startswith('_')]

        if len(submodules) > 0:
            this_mod_string = module_string.format('\n   '.join(submodules))
        else:
            this_mod_string = ''

        if len(functions) > 0:
            this_fun_string = fun_string.format('\n   '.join(functions))
        else:
            this_fun_string = ''

        if len(classes) > 0:
            this_class_string = class_string.format('\n   '.join(classes))
        else:
            this_class_string = ''

        with open(modname + '.rst', 'w') as text_file:
            text_file.write(string.format(shortname=shortmodname,
                                          name=modname,
                                          line=line,
                                          docstring=docstring,
                                          module_string=this_mod_string,
                                          fun_string=this_fun_string,
                                          class_string=this_class_string))


if __name__ == '__main__':
    make_interface()
