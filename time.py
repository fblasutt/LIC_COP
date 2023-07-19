import matplotlib
import matplotlib.pyplot as plt
import time
import Bargaining as brg
import Bargaining_numba as brgj
import numpy as np

# plot style
linestyles = ['-','--','-.',':',':']
markers = ['o','s','D','*','P']
linewidth = 2
font_size = 17
font = {'size':font_size}
matplotlib.rc('font', **font)

plt.rcParams.update({'figure.max_open_warning': 0,'text.usetex': False})


# settings for models to solve
T = 20
specs = {
    'model 1':{'latexname':'$\sigma_{\psi}=0$', 'par':{'sigma_love':0.1,'T':T,'Tr':2*T//3,'num_love':15}}
}

# solve different models (takes several minutes)
#model  = brg.HouseholdModelClass(name='model_1',par=specs['model 1']['par'])
modelj = brgj.HouseholdModelClass(name='model_1',par=specs['model 1']['par'])


#solve
tic=time.time()
modelj.solve()
toc=time.time()
print('Time elapsed is {}'.format(toc-tic))


tic=time.time()
modelj.simulate() 
toc=time.time()
print('Time elapsed is {}'.format(toc-tic))


# tic=time.time()
# model.solve()
# toc=time.time()
# print('Time elapsed is {}'.format(toc-tic))

# tic=time.time()
# model.simulate()
# toc=time.time()
# print('Time elapsed is {}'.format(toc-tic))

# print('Differences is {}'.format(np.min(model.sol.C_pub_couple[0,:,:,:]-modelj.sol.C_pub_couple[0,4,:,:,:])))

# print('Differences is {}'.format(np.max(model.sol.Vw_remain_couple[0,:,:,:]-modelj.sol.Vw_remain_couple[0,4,:,:,:])))

# print('Differences is {}'.format((modelj.sim.Cw_tot-model.sim.Cw_tot).max()))

import UserFunctions_numba as usr
par=modelj.par
sol=modelj.sol
sim=modelj.sim
agg_cons,home_good=np.zeros((2,*sim.Cw_pub.shape))
for t in range(par.simT):
    for i in range(par.simN):
        home_good[i,t]=usr.home_good(sim.Cw_pub[i,t],par.θ,par.λ,par.tb,couple=1.0,ishom=1.0-sim.WLP[i,t])
        agg_cons[i,t]= (par.α1*sim.Cw_priv[i,t]**par.ϕ1 + par.α2*home_good[i,t]**par.ϕ1)**(1/par.ϕ1)
        


#Graph the mean path of assets, income and consumption
base=np.cumsum(np.ones(modelj.par.T))
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(base, modelj.sim.incw.mean(axis=0)+modelj.sim.incm.mean(axis=0), label="Mean income path") 
ax.plot(base, modelj.sim.C_tot.mean(axis=0), label="Mean consumption path")
ax.plot(base, agg_cons.mean(axis=0), label="Mean aggregage consumption path")
ax.plot(base, modelj.sim.Aw.mean(axis=0)+modelj.sim.Am.mean(axis=0), label="Mean assets path") 
ax.grid()
ax.set_xlabel('t')                        #Label of x axis
ax.set_ylabel('a, y, c')                  #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph

#Graph the cross sectional variance path of income and consumption
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(base, np.var(np.log(modelj.sim.incw+modelj.sim.incm),axis=0), label="Variance log income path") 
ax.plot(base, np.var(np.log(modelj.sim.C_tot),axis=0), label="Variance log consumption path")
ax.grid()
ax.set_xlabel('t')                        #Label of x axis
ax.set_ylabel('Var(y), Var(c)')           #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph

