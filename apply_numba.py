import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

import Bargaining_numba as brg

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
    'model 1':{'latexname':'$\sigma_{\psi}=0$', 'par':{'sigma_love':0.0,'T':T,'num_love':5}},
    'model 2':{'latexname':'$\sigma_{\psi}=0.1$', 'par':{'sigma_love':0.1,'T':T,'num_love':5}},
}

# solve different models (takes several minutes)
models = {}
for name,spec in specs.items():
    print(f'solving {name}...')
    
    # setup model
    models[name] = brg.HouseholdModelClass(name=name,par=spec['par'])
    models[name].spec = spec
    
    # solve
    models[name].solve()
    
  
# Bargaining power updating
for model_name in ('model 1','model 2'):
    model = models[model_name]
    par = model.par
    sol = model.sol
    iz=4
    izw=iz//model.par.num_zm;izm=iz//model.par.num_zw
    for t in (par.T-1,):
        i = 0
        for iA in (5,20):
            for iL in (par.num_love//2,par.num_love//2 - 1):
                for sex in ('women','men'):
                    i += 1
                    fig, ax = plt.subplots()

                    # pick relevant values
                    if sex=='women':
                        V_single = sol.Vw_single[t,izw,iA]*np.ones(par.num_power)
                        V_remain_couple = sol.Vw_remain_couple[t,iz,:,iL,iA]
                        V_couple = sol.Vw_couple[t,iz,:,iL,iA]
                    else:
                        V_single = sol.Vm_single[t,izm,iA]*np.ones(par.num_power)
                        V_remain_couple = sol.Vm_remain_couple[t,iz,:,iL,iA]
                        V_couple = sol.Vm_couple[t,iz,:,iL,iA]

                    # plot values    
                    ax.plot(par.grid_power,V_single,linewidth=linewidth,label='value of divorce')
                    ax.plot(par.grid_power,V_remain_couple,marker='o',linewidth=linewidth,label='value of remaining couple')
                    ax.plot(par.grid_power,V_couple,linewidth=linewidth,linestyle='--',label='value of entering as couple')

                    # calculate corssing point
                    S = V_remain_couple - V_single
                    pos = np.int_(S > 0.0)
                    change = pos[1:] - pos[0:-1]
                    idx = np.argmax(np.abs(change))
                    if sex=='women':
                        idx = idx +1

                    denom = (par.grid_power[idx+1] - par.grid_power[idx]);
                    ratio = (S[idx+1] - S[idx])/denom;
                    power_zero = par.grid_power[idx] - S[idx]/ratio;
                    x = power_zero
                    
                    if x>0 and x <1:
                        ymin = -1.0
                        if t<par.T-1:
                            ymin = -5.0
                        ax.vlines(x=x, ymin=ymin, ymax=V_single[idx], color='gray',linestyle=':')

                        if t==par.T-1: 
                            if sex=='women':
                                ax.text(x,-0.6,'$\\rightarrow$',ha='left')
                            else:
                                ax.text(x,-0.6,'$\\leftarrow$',ha='right')

                    if t==par.T-1: ax.set(ylim=[-1.0, -0.3],xlabel='$\mu$ (bargaining power of women)',ylabel='value')

                    if i==1: plt.legend();
                    plt.tight_layout();   
                    
#Policy Functions
cmaps = ('viridis','gray')
model_list = ('model 1','model 2')
t = 2
iz=7
par = models['model 1'].par
X, Y = np.meshgrid(par.grid_power, par.grid_A,indexing='ij')

for iL in (par.num_love//2,): 
    for var in ('Vw_couple','Vm_couple','Cw_priv_couple','Cm_priv_couple','C_pub_couple','C_tot_couple','power','WLP'):

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        for i,name in enumerate(model_list):
            model = models[name]
        
            Z = getattr(model.sol,var)[t,iz,:,iL,:]
            
            alpha = 0.5 if name=='model 1' else 0.7
            ax.plot_surface(X, Y, Z,rstride=1,cstride=1,cmap=cmaps[i], edgecolor='none',alpha=alpha);
            
            if var == 'power': 
                
                ax.set(zlim=[0.0,1.0])
            
            ax.set(xlabel='$\mu_{t-1}$',ylabel='$A_{t-1}$',zlabel=f'{var}');
        
        plt.tight_layout();
        
        
#Policy Functions II
cmaps = ('viridis','gray')
model_list = ('model 1','model 1')
t = 1
iA=0
par = models['model 1'].par
X, Y = np.meshgrid(np.cumsum(np.ones(par.num_z)),par.grid_power,indexing='ij')

for iL in (par.num_love//2,): 
    for var in ('Vw_couple','Vm_couple','Cw_priv_couple','Cm_priv_couple','C_pub_couple','C_tot_couple','power','WLP'):

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        for i,name in enumerate(model_list):
            model = models[name]
        
            Z = getattr(model.sol,var)[t,:,:,iL,iA]
            
            alpha = 0.5 if name=='model 1' else 0.7
            ax.plot_surface(X, Y, Z,rstride=1,cstride=1,cmap=cmaps[i], edgecolor='none',alpha=alpha);
            
            if var == 'power': 
                
                ax.set(zlim=[0.0,1.0])
            
            ax.set(ylabel='$\mu_{t-1}$',xlabel='$z_{t-1}$',zlabel=f'{var}');
        
        plt.tight_layout();
        
# Simulated Path

var_list = ('Cw_priv','Cm_priv','Cw_pub','C_tot','A','power','power_idx','love','couple','WLP')
model_list = ('model 1','model 2')

for init_power_idx in (1,10):
    for init_love in (0.0,0.2): 

            for i,name in enumerate(model_list):
                model = models[name]

                # show how starting of in a low bargaining power gradually improves
                model.sim.init_power_idx[:] = init_power_idx
                model.sim.init_love[:] = init_love 
                model.simulate()
                
            for var in var_list:

                fig, ax = plt.subplots()
                
                for i,name in enumerate(model_list):
                    model = models[name]

                    # pick out couples (if not the share of couples is plotted)
                    if var == 'couple':
                        nan = 0.0
                    else:
                        I = model.sim.couple<1
                        nan = np.zeros(I.shape)
                        nan[I] = np.nan

                    # pick relevant variable for couples
                    y = getattr(model.sim,var)        
                    y = np.nanmean(y + nan,axis=0)

                    ax.plot(y,marker=markers[i],linestyle=linestyles[i],linewidth=linewidth,label=model.spec['latexname']);
                    ax.set(xlabel='age',ylabel=f'{var}');ax.set_title(f'pow_idx={init_power_idx}, init_love={init_love}')

                plt.legend()
                plt.tight_layout()
