import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Bargaining_numba as brg
import setup
mar=setup.mar;coh=setup.coh

# plot style
linestyles = ['-','--','-.',':',':']
markers = ['o','s','D','*','P']
linewidth = 2
font_size = 8
font = {'size':font_size}
matplotlib.rc('font', **font)
plt.rcParams.update({'figure.max_open_warning': 0,'text.usetex': False})


# settings for models to solve
T = 8
specs = {
    #'model 1':{'latexname':'EGM1', 'par':{'γ':[0.0, 0.0],'sep_cost':[0.0,0.2],'grid_title':np.array([0.5]),'sigma_love':0.4,'T':T,'num_A':20,'max_A':8.0,"num_power":15}},
    'model 2':{'latexname':'EGM2', 'par':{'γ':[0.0, 0.0],'sep_cost':[0.0,0.2],'grid_title':np.array([0.5,0.4,0.6]),'sigma_love':0.4,'T':T,'num_A':20,'max_A':8.0,"num_power":15}},
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
    
   
###################        
#Policy Functions
###################
cmaps = ('viridis',)#('viridis','gray')#
model_list =('model 2',)#('model 1','model 2')# 


#Points to consider
par = models['model 2'].par
t = 4; iz=8; ih=1; iL=5#par.num_love//2
 
for var in ('p_Vw_remain_couple','p_C_tot_remain_couple','remain_WLP'):

    fig = plt.figure();ax = plt.axes(projection='3d')
            
    for i,name in enumerate(model_list):
        model = models[name]
        par = models[name].par
        X, Y = np.meshgrid(par.grid_power, par.grid_A,indexing='ij')
        
        Z = getattr(model.sol,var)[t,1,ih,iz,:,iL,:]
        alpha = 0.2 if name=='model 2' else 0.5
        ax.plot_surface(X, Y, Z,cmap=cmaps[i],alpha=alpha);
        if var == 'power':  ax.set(zlim=[0.0,1.0])
        ax.set(xlabel='power',ylabel='$A$');ax.set_title(f'{var}')
plt.show()    
####################################    
# Simulated Path - general variables
####################################
var_list = ('rel','A','power','WLP')
init_power=model.par.grid_power[7];init_love=par.num_love//2
for i,name in enumerate(model_list):
    model = models[name]

    # show how starting of in a low bargaining power gradually improves
    model.sim.init_power[:] = init_power
    model.sim.init_love[:] = init_love 
    model.simulate()
    
for var in var_list:

    fig, ax = plt.subplots()
    
    for i,name in enumerate(model_list):
        model = models[name]

        # pick out couples (if not the share of couples is plotted)
        if var == 'rel': nan = 0.0
        else:
            I = model.sim.rel>1
            nan = np.zeros(I.shape)
            nan[I] = np.nan

        # pick relevant variable for couples
        y = getattr(model.sim,var);y = np.nanmean(y + nan,axis=0)
        ax.plot(y,marker=markers[i],linestyle=linestyles[i],linewidth=linewidth,label=model.spec['latexname']);
        ax.set(xlabel='age',ylabel=f'{var}');ax.set_title(f'pow_idx={init_power}, init_love={init_love}')
plt.show()

###################################
# Simulated path - partnership type
################################### 
fig, ax = plt.subplots() 
 
colors=('blue','red','black') 
rel_list=('mar','coh','single') 
for j,var in enumerate(rel_list): 
    for i,name in enumerate(model_list): 
        model = models[name] 
     
        # pick out couples (if not the share of couples is plotted) 
        I = model.sim.rel==j 
        yt = np.zeros(I.shape) 
        yt[I] = 1.0 
     
        # pick relevant variable for couples 
        y = np.mean(yt,axis=0) 
        ax.plot(y,color=colors[j],linestyle=linestyles[i],linewidth=linewidth,label=var);ax.legend() 
ax.set(xlabel='age',ylabel=f'{var}');ax.set_title(f'pow_idx={init_power}, init_love={init_love}') 
plt.show() 

###################################
# Simulated path - love by type
###################################
fig, ax = plt.subplots()

colors=('blue','red')
rel_list=('mar','coh')
for j,var in enumerate(rel_list):
    for i,name in enumerate(model_list):
        model = models[name]
    
        # pick out couples (if not the share of couples is plotted)
        I = model.sim.rel!=j
        yw = model.par.grid_lovew[t][model.sim.love//par.num_lovem]
        ym = model.par.grid_lovem[t][model.sim.love%par.num_lovew]
        yw[I] = np.nan
        ym[I] = np.nan
    
        # pick relevant variable for couples
        y_w = np.nanmean(yw,axis=0);y_m = np.nanmean(ym,axis=0)
        ax.plot(y_w,marker=markers[0],color=colors[j],linestyle=linestyles[i],linewidth=linewidth,label=var)
        ax.plot(y_m,marker=markers[1],color=colors[j],linestyle=linestyles[i],linewidth=linewidth,label=var);ax.legend()
ax.set(xlabel='age',ylabel=f'{var}');ax.set_title(f'pow_idx={init_power}, init_love={init_love}')
plt.show()

#####################################
#Cohabitation - marriage analysis
#####################################
#model1 = models['model 1']
model2 = models['model 2']
shape_couple = (par.T,2,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)
V1,V2=np.zeros(shape_couple),np.zeros(shape_couple)
for t in range(par.T):
    for rel in (mar,coh):
        for ih in range(par.num_h):  
            for iz in range(par.num_z): 
                for iP in range(par.num_power):   
                    for iL in range(par.num_love): 
                        for iA in range(par.num_A): 
                   
                            idx=(t,rel,ih,iz,iP,iL,iA)
                            #V1[idx]=par.grid_power[iP]      *model1.sol.p_Vw_remain_couple[idx]+\
                             #       (1.0-par.grid_power[iP])*model1.sol.p_Vm_remain_couple[idx]
                                    
                            V2[idx]=par.grid_power[iP]      *model2.sol.n_Vw_remain_couple[idx]+\
                                    (1.0-par.grid_power[iP])*model2.sol.n_Vm_remain_couple[idx]
                                    
                            
t = 4; iz=8; ih=1; iL=5
plt.plot(V2[t,0,ih,iz,:,iL,:]-V2[t,1,ih,iz,:,iL,:])
plt.show()
plt.plot(V2[t,0,ih,iz,:,iL,:].T-V2[t,1,ih,iz,:,iL,:].T)
plt.show()

#print('Model 1 Vmar-Vcoh is {}'.format((V1[:model.par.Tr-1,0,:,:,:,:,:]-V1[:model.par.Tr-1,1,:,:,:,:,:]>=0.0).mean()))
print('Model 2 Vmar-Vcoh is {}'.format((V2[:model.par.Tr-3,0,:,:,:,:,:]-V2[:model.par.Tr-3,1,:,:,:,:,:]>=0.0).mean()))


