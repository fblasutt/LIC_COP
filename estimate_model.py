# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:31:56 2024

@author: 32489
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Bargaining_numba as brg
import setup
import matplotlib.colors as colorss
import pybobyqa
from scipy.optimize import dual_annealing,differential_evolution
import scipy
import time
import UserFunctions_numba as usr
import dfols

mar=setup.mar;coh=setup.coh


#Function to minimize
def q(pt,model_old):

   
    model = model_old.copy(name='numba_new_copy')
    
    model.par.unil=[True,True]
    model.par.grid_title=np.array([0.5])#np.linspace(0.2,0.8,11)#
    model.par.comty_regime=[True,True]
    #model.par.grid_h = np.array([0.0,0.0])
   
    model.par.meet=pt[0]
    model.par.sep_cost_u=[0.0,0.0]
    
    model.par.λ_grid = np.ones(model.par.T)*pt[0]
    for t in range(model.par.Tr,model.par.T):model.par.λ_grid[t]=0.0
    
    model.par.γ=[0.0,pt[1]]
    
    
    model.par.grid_lovew,model.par.Πlw,model.par.Πlw0= usr.addaco_nonst(model.par.T,pt[3]*pt[2],pt[2],model.par.num_lovew)
    model.par.grid_lovem,model.par.Πlm,model.par.Πlm0= usr.addaco_nonst(model.par.T,pt[3]*pt[2],pt[2],model.par.num_lovem)
    model.par.Πl =[np.kron(model.par.Πlw[t], model.par.Πlm[t] ) for t in range(model.par.T-1)] # couples trans matrix
    model.par.Πl0=[np.kron(model.par.Πlw0[t],model.par.Πlm0[t]) for t in range(model.par.T-1)] # couples trans matrix
        
        
    
    model.par.α2=pt[4]
    model.par.α1=1.0-pt[4]
    
    print(pt)
    # solve and simulate the model
    tic=time.time()
    model.solve()
    model.simulate()
    toc=time.time()
    #print('Time elapsed for model solution is {}'.format(toc-tic))
    
    # Compute the moments
    
    #Women's labor supply
    WLP = model.sim.WLP[:,10:30][model.sim.rel[:,10:30]<2].mean()

    #Share ever married and cohabited
    share_m = np.mean(np.cumsum(model.sim.rel==0,axis=1)[:,16]>0)#np.mean(model.sim.rel[:,16]==0)#
    share_c = np.mean(np.cumsum(model.sim.rel==1,axis=1)[:,16]>0)#np.mean(model.sim.rel[:,16]==1)#

    #Share ever divorce and broke up
    share_d = np.mean(np.cumsum((model.sim.rel_lag==0) & (model.sim.rel==2),axis=1)[:,16]>0)
    share_b = np.mean(np.cumsum((model.sim.rel_lag==1) & (model.sim.rel==2),axis=1)[:,16]>0)
    
    
    print('WLP: {}, share_m: {}; share_c: {}, share_d: {}, share_b: {}'.format(WLP,share_m,share_c,share_d,share_b))
    #fit =((WLP-0.55)/0.55)**2+((share_m-0.95)/0.95)**2+((share_c-0.35)/0.35)**2+((share_d-0.23)/0.23)**2+((share_b-0.18)/0.18)**2
   
    fit =((WLP-0.55))**2+((share_m-0.947))**2+((share_c-0.378))**2+((share_d-0.312))**2+((share_b-0.106))**2
    print(fit)
    #return [((WLP-0.55)/0.55),((share_m-0.95)/0.95),(share_c-0.35)/0.35,((share_d-0.23)/0.23),((share_b-0.18)/0.18)]
    return [((WLP-0.55)),((share_m-0.947)),(share_c-0.378),((share_d-0.312)),((share_b-0.106))]
    #return fit
    
    
    
#Initialize seed
np.random.seed(10)


#Define initial point (xc) and boundaries (xl,xu)

#unilateral estimation love:3
xc=np.array([0.308232  , 0.00883859, 1.82712902, 0.33920033, 0.40122403] )
xl=np.array([0.1,0.0001,0.1,0.01,0.2])
xu=np.array([0.8,0.015  ,2.5,1.0 ,0.60])

#unilateral estimation love:5
xc=np.array([0.99,       0.00297826, 0.02161296, 0.40408969, 0.40300724])
xc=np.array([0.92574115, 0.00898869, 0.02515822, 0.56917511, 0.40189191])

xc=np.array([0.50067821, 0.01076512, 0.0503481 , 0.78203779, 0.40640988])
xl=np.array([0.1,0.0001,0.001,0.001,0.3])
xu=np.array([0.999,0.2  ,0.3,2.0 ,0.50])





#Parametrize the model
par = {'unil':[False,True],'comty_regime':[True,True],'sep_cost':[0.2,0.0],'grid_title': np.array([0.5]),
       'meet': xc[0], 'γ':[0.0,xc[1]],'σL0':xc[2],'σL':xc[3],'α2':xc[4],'α1':1.0-xc[4]}
model = brg.HouseholdModelClass(par=par)
    



#Optimization below
# res=dfols.solve(q, xc,args=(model,), rhobeg = 0.3, rhoend=1e-5, maxfun=20000, bounds=(xl,xu),
#                 npt=len(xc)+5,scaling_within_bounds=True, seek_global_minimum=True,
#                 user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,
#                               'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},
#                 objfun_has_noise=False,print_progress=True)


res=dfols.solve(q, xc,args=(model,), rhobeg = 0.3, rhoend=1e-4, maxfun=100, bounds=(xl,xu),
                npt=len(xc)+5,scaling_within_bounds=True, 
                user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,
                              'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},
                objfun_has_noise=False,print_progress=True)
 
#res = scipy.optimize.minimize(q,xc,args=(model),bounds=list(zip(list(xl), list(xu))),method='Nelder-Mead',tol=1e-3)
#res = differential_evolution(q,args=(model,),bounds=list(zip(list(xl), list(xu))),disp=True,mutation=(0.1, 0.5),recombination=0.8) 
 
