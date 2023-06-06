import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import time
import Bargaining as brg
import Bargaining_numba2 as brgj

# plot style
linestyles = ['-','--','-.',':',':']
markers = ['o','s','D','*','P']
linewidth = 2
font_size = 17
font = {'size':font_size}
matplotlib.rc('font', **font)

plt.rcParams.update({'figure.max_open_warning': 0,'text.usetex': False})


# settings for models to solve
T = 4
specs = {
    'model 1':{'latexname':'$\sigma_{\psi}=0$', 'par':{'sigma_love':0.1,'T':T,'num_love':15}}
}

# solve different models (takes several minutes)
model  = brg.HouseholdModelClass(name='model_1',par=specs['model 1']['par'])
modelj = brgj.HouseholdModelClass(name='model_1',par=specs['model 1']['par'])


# solve
tic=time.time()
modelj.solve()
toc=time.time()
print('Time elapsed is {}'.format(toc-tic))


tic=time.time()
modelj.solve()
toc=time.time()
print('Time elapsed is {}'.format(toc-tic))


tic=time.time()
modelj.solve()
toc=time.time()
print('Time elapsed is {}'.format(toc-tic))


# tic=time.time()
# model.solve()
# toc=time.time()
# print('Time elapsed is {}'.format(toc-tic))


# assert np.allclose(model.sol.C_pub_couple,modelj.sol.C_pub_couple)