import os
import types
import pkg_resources
import scvelo as scv

def _get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]
            
        rename_packages = {"PIL": "Pillow",
                            "sklearn": "scikit-learn"}

        if name in rename_packages.keys():
            name = rename_packages[name]

        yield name

def print_imports():
    """Prints import statements of the loaded in Python modules.

    Parameters
    ----------

    Returns
    -------
    """
    imports = list(set(_get_imports()))
    requirements = []
    for m in pkg_resources.working_set:
        if m.project_name in imports and m.project_name!="pip":
            requirements.append((m.project_name, m.version))

    for r in requirements:
        print("{}=={}".format(*r))

    scv.logging.print_version()

def make_directory(directory: str = None):
    """Creates a directory at the specified path if one doesn't exist.

    Parameters
    ----------
    directory : str
        A string specifying the directory path.

    Returns
    -------
    """
    if not os.path.exists(directory):
        os.makedirs(directory)