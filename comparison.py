import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Bargaining_numba as brg

# plot style
linestyles = ['-','--','-.',':',':']
markers = ['o','s','D','*','P']
linewidth = 2
font_size = 8
font = {'size':font_size}
matplotlib.rc('font', **font)
plt.rcParams.update({'figure.max_open_warning': 0,'text.usetex': False})


# settings for models to solve
T = 4
specs = {
    'model 1':{'latexname':'EGM', 'par':{'sigma_love':0.2,'T':T,'num_A':100,'max_A':1.5,'num_love':3,"num_power":35,"EGM":True}},
    'model 2':{'latexname':'VFI', 'par':{'sigma_love':0.2,'T':T,'num_A':100,'max_A':1.5,'num_love':3,"num_power":35,"EGM":False}},
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
    
   
        
#Policy Functions
cmaps = ('viridis','gray')
model_list = ('model 1','model 2')

#Points to consider
t = 2; iz=0; ih=0

par = models['model 1'].par
for iL in (par.num_love//2,): 
    for var in ('p_Vw_remain_couple','n_C_tot_remain_couple','power','remain_WLP'):

        fig = plt.figure();ax = plt.axes(projection='3d')
                
        for i,name in enumerate(model_list):
            model = models[name]
            par = models[name].par
            X, Y = np.meshgrid(par.grid_power, par.grid_A,indexing='ij')
            
            Z = getattr(model.sol,var)[t,ih,iz,:,iL,:]
            alpha = 0.2 if name=='model 1' else 0.5
            ax.plot_surface(X, Y, Z,cmap=cmaps[i],alpha=alpha);
            if var == 'power':  ax.set(zlim=[0.0,1.0])
            ax.set(xlabel='power',ylabel='$A$');ax.set_title(f'{var}')
    
# Simulated Path
var_list = ('couple','A','power','love','WLP')
model_list = ('model 1','model 2')
init_power_idx=7;init_love=0.0
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
        if var == 'couple': nan = 0.0
        else:
            I = model.sim.couple<1
            nan = np.zeros(I.shape)
            nan[I] = np.nan

        # pick relevant variable for couples
        y = getattr(model.sim,var);y = np.nanmean(y + nan,axis=0)
        ax.plot(y,marker=markers[i],linestyle=linestyles[i],linewidth=linewidth,label=model.spec['latexname']);
        ax.set(xlabel='age',ylabel=f'{var}');ax.set_title(f'pow_idx={init_power_idx}, init_love={init_love}')


#plt.hist(par.grid_power[model.sim.power_idx_lag[model.sim.power_idx!=-1]]-par.grid_power[model.sim.power_idx[model.sim.power_idx!=-1]])
