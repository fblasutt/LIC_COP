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
    'model 2':{'latexname':'VFI', 'par':{'sigma_love':0.2,'T':T,'num_A':100,'max_A':1.5,'num_love':3,"num_power":35,"EGM":True}},
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
t = 2
iz=0


par = models['model 1'].par
for iL in (par.num_love//2,): 
    for var in ('p_Vw_remain_couple','n_C_tot_remain_couple','power','remain_WLP'):

        fig = plt.figure()
        ax = plt.axes(projection='3d')
                
        for i,name in enumerate(model_list):
            model = models[name]
            par = models[name].par
            X, Y = np.meshgrid(par.grid_power, par.grid_A,indexing='ij')
            

        
            for i,name in enumerate(model_list):
                model = models[name]
            
                Z = getattr(model.sol,var)[t,iz,:,iL,:]
                
                alpha = 0.2 if name=='model 1' else 0.5
                ax.plot_surface(X, Y, Z,rstride=1,cstride=1,cmap=cmaps[i], edgecolor='none',alpha=alpha);
                
                if var == 'power': 
                    
                    ax.set(zlim=[0.0,1.0])
                
                ax.set(xlabel='$\mu_{t-1}$',ylabel='$A_{t-1}$',zlabel=f'{var}');
        
        plt.legend()
        plt.tight_layout();
            
        
# Simulated Path
var_list = ('couple','A','power','love','WLP')#'Cw_priv','Cm_priv','Cw_pub','C_tot',
model_list = ('model 1','model 2')


init_power_idx=7
init_love=0.0
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
    plt.tight_layout();

#Two nice final checks (inidipendent on weights at divorce)
iP=0
t=0
iL=3
util_model1=model.par.grid_power[iP]*getattr(models['model 1'].sol,'p_Vw_remain_couple')[t,iz,iP,iL-2,:]+\
      (1.0-model.par.grid_power[iP])*getattr(models['model 1'].sol,'p_Vm_remain_couple')[t,iz,iP,iL-2,:]
      
util_model2=model.par.grid_power[iP]*getattr(models['model 2'].sol,'p_Vw_remain_couple')[t,iz,iP,iL-2,:]+\
      (1.0-model.par.grid_power[iP])*getattr(models['model 2'].sol,'p_Vm_remain_couple')[t,iz,iP,iL-2,:]

fig, ax = plt.subplots()
plt.plot(util_model1-util_model2)

fig, ax = plt.subplots()
plt.plot(model.par.grid_A,getattr(models['model 1'].sol,'p_C_tot_remain_couple')[t,iz,0,iL-2,:],
         model.par.grid_A,getattr(models['model 2'].sol,'p_C_tot_remain_couple')[t,iz,0,iL-2,:])

fig, ax = plt.subplots()
plt.plot(model.par.grid_power,getattr(models['model 1'].sol,'p_C_tot_remain_couple')[t,iz,:,iL-2,0],
         model.par.grid_power,getattr(models['model 2'].sol,'p_C_tot_remain_couple')[t,iz,:,iL-2,0])


fig, ax = plt.subplots()
plt.plot(model.par.grid_power,getattr(models['model 1'].sol,'power')[t,iz,:,iL-2,0],
         model.par.grid_power,getattr(models['model 2'].sol,'power')[t,iz,:,iL-2,0])


fig, ax = plt.subplots()
plt.plot(model.par.grid_A,getattr(models['model 1'].sol,'power')[t,iz,0,iL-2,:],
         model.par.grid_A,getattr(models['model 2'].sol,'power')[t,iz,0,iL-2,:])





