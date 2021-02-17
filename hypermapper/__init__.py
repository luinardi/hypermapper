import sys

# check if module called as command line script, otherwise import
if "-m" not in sys.argv:
    from . import bo  # noqa
    from . import compute_pareto  # noqa
    from . import evolution  # noqa
    from . import local_search  # noqa
    from . import models  # noqa
    from . import optimizer  # noqa
    from . import plot_pareto  # noqa
    from . import plot_hvi  # noqa
    from . import plot_optimization_results  # noqa
    from . import prior_optimization  # noqa
    from . import random_scalarizations  # noqa
    from . import space  # noqa
    from . import utility_functions  # noqa
    from . import profiling  # noqa
