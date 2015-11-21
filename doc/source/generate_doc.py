import os
import pkgutil
import odl
import inspect
import importlib


module_string = """
Sub-modules
-----------

.. autosummary::
   :toctree: generated/

   {}
"""

fun_string = """
Functions
---------

.. autosummary::
   :toctree: generated/

   {}
"""

class_string = """
Classes
-------

.. autosummary::
   :toctree: generated/

   {}
"""

string = """{name} package
{line}

.. currentmodule:: {name}

{module_string}
{fun_string}
{class_string}
"""


def make_interface():
    if not os.path.exists('odl_interface'):
        os.makedirs('odl_interface')

    for importer, modname, ispkg in pkgutil.walk_packages(path=odl.__path__,
                                                          prefix=odl.__name__+'.',
                                                          onerror=lambda x: None):
        print(modname)

        line = '=' * (len(modname) + 8)

        module = importlib.import_module(modname)

        submodules = [m[0] for m in inspect.getmembers(module, inspect.ismodule) if m[1].__name__.startswith('odl')]
        functions = [m[0] for m in inspect.getmembers(module, inspect.isfunction) if m[1].__module__ == modname]
        classes = [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == modname]

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

        text_file = open('odl_interface/' + modname + '.rst', "w")
        text_file.write(string.format(name=modname,
                                      line=line,
                                      module_string=this_mod_string,
                                      fun_string=this_fun_string,
                                      class_string=this_class_string))

        text_file.close()

if __name__ == '__main__':
    make_interface()
