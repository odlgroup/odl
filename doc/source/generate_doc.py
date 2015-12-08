import pkgutil
import odl
import inspect
import importlib

__all__ = ('make_interface',)

module_string = """
.. toctree::
   :maxdepth: 2

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

string = """{shortname}
{line}

{docstring}

.. currentmodule:: {name}

{module_string}
{class_string}
{fun_string}
"""


def make_interface():

    modnames = [modname for _, modname, _ in pkgutil.walk_packages(path=odl.__path__,
                                                                   prefix=odl.__name__+'.',
                                                                   onerror=lambda x: None)]

    modnames += ['odl']

    for modname in modnames:
        shortmodname = modname.split('.')[-1]
        print('generated: ' + modname + '.rst')

        line = '=' * len(shortmodname)

        module = importlib.import_module(modname)

        docstring = module.__doc__
        submodules = [m[0] for m in inspect.getmembers(module, inspect.ismodule) if m[1].__name__.startswith('odl')]
        functions = [m[0] for m in inspect.getmembers(module, inspect.isfunction) if m[1].__module__ == modname]
        classes = [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == modname]

        docstring = '' if docstring is None else docstring

        submodules = [modname + '.' + mod for mod in submodules]
        functions = ['~' + modname + '.' + fun for fun in functions if not fun.startswith('_')]
        classes = ['~' + modname + '.' + cls for cls in classes if not cls.startswith('_')]

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


        text_file = open(modname + '.rst', "w")
        text_file.write(string.format(shortname=shortmodname,
                                      name=modname,
                                      line=line,
                                      docstring=docstring,
                                      module_string=this_mod_string,
                                      fun_string=this_fun_string,
                                      class_string=this_class_string))

        text_file.close()

if __name__ == '__main__':
    make_interface()
