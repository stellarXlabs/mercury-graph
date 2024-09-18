import importlib.util

spec = importlib.util.find_spec('numba')


if spec is None:
    def njit(f):
        """
        This decorator does nothing, it is a fallback in case numba is not installed.
        """

        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper

    graph_i4 = int
    numba_installed = False


else:
    from numba import njit
    from numba.types import i4 as graph_i4
    numba_installed = True
