from ctypes import *
import time
import site
import os

site_packages = site.getusersitepackages()
libfile = site_packages + '/' + \
          [x for x in os.listdir(site_packages) if x.find('hypermapper_c_bayesian.') >= 0][0]

bayesian_c = cdll.LoadLibrary(libfile)

bayesian_c.random_init.restype = None
bayesian_c.random_init(int(time.time()))

bayesian_c.bayesian_optimization.restype = c_double

OptFuncType = CFUNCTYPE(c_double, POINTER(c_double))

class BayesianCWrapper:

    def __init__(self, blackbox, config):
        self.f = blackbox
        self.n_doe = config['design_of_experiment']['number_of_samples']
        self.max_iter = config['optimization_iterations'] + self.n_doe
        self.y = config['optimization_objectives'][0]
        self.vars = []
        self.lower = []
        self.upper = []
        for name, prop in config['input_parameters'].items():
            self.vars.append(name)
            bounds = prop['values']
            self.lower.append(bounds[0])
            self.upper.append(bounds[1])
        self.dims = len(self.vars)

    def f_wrapper(self, x):
        x_dict = {self.vars[i]: x[i] for i in range(self.dims)}
        return self.f(x_dict)

    def run(self):
        x = (self.dims * c_double)(0.0)
        y = bayesian_c.bayesian_optimization(OptFuncType(self.f_wrapper),
                                           (self.dims * c_double)(*self.lower),
                                           (self.dims * c_double)(*self.upper),
                                           x,
                                           self.dims, self.n_doe, self.max_iter)
        out = {self.vars[i]: [x[i]] for i in range(self.dims)}
        out[self.y] = [y]
        return out
