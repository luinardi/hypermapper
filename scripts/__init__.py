import warnings

warnings.warn(
    "Imports from 'scripts' are deprecated and will be removed in the future. "
    "Use 'hypermapper' instead.",
    DeprecationWarning,
    2,
)

from hypermapper import bo  # noqa
from hypermapper import compute_pareto  # noqa
from hypermapper import evolution  # noqa
from hypermapper import local_search  # noqa
from hypermapper import models  # noqa
from hypermapper import optimizer as hypermapper  # noqa
from hypermapper import plot_pareto  # noqa
from hypermapper import plot_hvi  # noqa
from hypermapper import plot_optimization_results  # noqa
from hypermapper import prior_optimization  # noqa
from hypermapper import random_scalarizations  # noqa
from hypermapper import space  # noqa
from hypermapper import utility_functions  # noqa
