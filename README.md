## Hypermapper

The Bayesian Optimization framework (Hypermapper) is a flexible out-of-the-box compiler optimizer. The aim is to develop a flexible tool that could be applied in a wide range of optimization settings. It has successfully been used in for example hardware design and compiler tuning.

This is a new version that is being introduced and so will be subject to change and improvements in the near future.  

## Installation
To install the code, clone the repo and run the setup files.
```
git clone https://github.com/luinardi/hypermapper.git
cd hypermapper
git checkout hypermapper-v3
pip install -e .
```

## Use
There are two main ways to interact with Hypermapper: either by using calling it through python

```
import hypermapper
hypermapper.optimize("json_name.py", blackbox-function)
```
if you have a python interface to your application, or through the client-server functionality that interacts with your application through the terminal. In either case, a .json scenario file is required that sets up the optimization. This is where the input parameters are given as well as other run settings. Examples of scenario files can be found in tests/aux and the full template is found in hypermapper/schema.json.

# Running Hypermapper with a black-box function
To run it with a blackbox function, simply call the optimize() routine with a callable python-function and the name of the scenario file.

# Running Hypermapper client-server
In the client-server mode, the compiler framework calls Hypermapper on demand asking for recommended settings.

The two parties communicate via a client (the third-party software) and server (Hypermapper) protocol defined by Hypermapper. The general idea of the protocol is that the client asks the server the configurations to run. So, there is the first design of experiments phase where samples are drawn and a second phase that is about the (Bayesian) optimization.

To enable the Client-Server mode add this line to the json file:

```
“hypermapper_mode”: {
       “mode”: “client-server”
}
```

The client and server communicate following a csv-like protocol via the standard output. The client calls Hypermapper to start the optimization process. When called by the client, Hypermapper will reply by requesting a number of function evaluations and wait for a reply. As an example, Hypermapper will reply with:

```
Request 3
x1,x2
-10,12
1,6
-8,20
```

Note that Hypermapper starts the protocol by stating how many evaluations it is requesting, followed by the input parameters (x1 and x2 int this case) and a series of parameter values.

Upon receiving this message, the client must compute the function values at the requested points and reply with the input parameters and the function values:

```
x1,x2,value,Valid
-10,12,267,False
1,6,28,True
-8,20,463,False
```

This protocol continues for the number of iterations specified by the client in the scenario file and after all iterations are done, Hypermapper will save all explored points to a csv file and end its execution.

A more detailed description will be uploaded soon. To just try to run the code, simply run tests/test_all from the Hypermapper root folder.

# BaCO
This code was in large developed for our work on Compiler Optimization. We would recommend having a look at our paper
https://arxiv.org/abs/2212.11142.
We would further appreciate if you cited this work if you use HyperMapper in your academic publications.
