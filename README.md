# Dynamic collective model with limited commitment


This repository contains accompanying code for solution and simulation of limited commitment dynamic bargaining models.

The skeleton of this repository comes from Thomas Jorgensen's [Household Bargaining Guide](https://github.com/ThomasHJorgensen/HouseholdBargainingGuide), which is extended
by including income shocks and taste shocks for participation (soon enough I will add human capital accumulation).

The idea is to have a compact and fast (numba is used almost everywhere) basic dynamic collective model with limited commitment. A list of files follow:

- model.ipynb: describe the baseline model
- Bargaining_numba.py: solve and simulate the model using the method of endogenous gridpoints
- UserFunctions_numba.py: utility functions, budget constraints and other utilities used in Bargaining_numba.py
- time.py: time the solution and simulation of the model
- vfi.py: contains the codes for solving the model using a minimization routine for finding optimal savings. This is pretty slow but also robust, can be
  used for checking the accuracy iof EGM (NB: for the comparison set the number of gridpoints for assets to a large number, like 300...). You can activate
  this method by setting par.EGM to False in Bargaining_numba.py.
- setup.py: switch on/off numba and parallelization. Useful for debugging. Setting parallilization to True implies very long compilation times...

