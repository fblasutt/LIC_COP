import numpy as np#import autograd.numpy as np#
from EconModel import EconModelClass, jit
from consav.grids import nonlinspace
from consav import linear_interp, linear_interp_1d,quadrature
from numba import njit,prange,config
import UserFunctions_numba as usr
import vfi as vfi
from quantecon.optimize.nelder_mead import nelder_mead
upper_envelope=usr.create(usr.value_of_choice)
# set gender indication as globals
woman = 1
man = 2
 
parallel=False
config.DISABLE_JIT = False


class HouseholdModelClass(EconModelClass):
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. cpp
        self.cpp_filename = 'cppfuncs/solve.cpp'
        self.cpp_options = {'compiler':'vs'}
        
        
    def setup(self):
        par = self.par
        
        
        par.EGM = True # Want to use the EGM method to solve the model?
        
        par.R = 1.01#3
        par.β = 1.00#/par.R # Discount factor
        
        par.div_A_share = 0.5 # divorce share of wealth to wife

        # Utility: CES aggregator or additively linear
        par.ρ = 2.0#        # CRRA      
        par.α1 = 0.55
        par.α2 = 0.0
        par.ϕ1 = 0.2
        par.ϕ2 = (1.0-par.ρ)/par.ϕ1
        
        # production of home good
        par.θ = 1.0#0.21 #weight on money vs. time to produce home good
        par.λ = 0.19 #elasticity betwen money and time in public good
        par.tb = 0.2 #time spend on public goods by singles
        
        #Taste shock
        par.σ = 0.000000001 #taste shock applied to working/not working
        
        ####################
        # state variables
        #####################
        
        par.T = 10 # terminal age
        par.Tr = 6 # age at retirement
        
        # wealth
        par.num_A = 300
        par.max_A = 6.0
        
        # bargaining power
        par.num_power = 3

        # love/match quality
        par.num_love = 41
        par.max_love = 1.0

        par.sigma_love = 0.1
        par.num_shock_love = 5
        
        # income of men and women
        par.num_zw=3
        par.num_zm=3
        par.num_z=par.num_zm*par.num_zw
        
        par.t0w=-0.5;par.t1w=0.03;par.t2w=0.0;par.σzw=0.00001;par.σ0zw=0.00005
        par.t0m=-0.5;par.t1m=0.03;par.t2m=0.0;par.σzm=0.00001;par.σ0zm=0.00005

        # pre-computation
        par.num_Ctot = 200
        par.max_Ctot = par.max_A*2

        par.num_A_pd = par.num_A * 2

        # simulation
        par.seed = 9210
        par.simT = par.T
        par.simN = 50_000
        
    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # setup grids
        par.simT = par.T
        self.setup_grids()
        
        # singles
        shape_singlew = (par.T,par.num_zw,par.num_A)
        shape_singlem = (par.T,par.num_zm,par.num_A)
        sol.Vw_single = np.nan + np.ones(shape_singlew)
        sol.Vm_single = np.nan + np.ones(shape_singlem)
        sol.Cw_tot_single = np.nan + np.ones(shape_singlew)
        sol.Cm_tot_single = np.nan + np.ones(shape_singlem)
        
        # EGM
        sol.Vw_marg_single =  np.ones(shape_singlew)
        sol.Vm_marg_single =  np.ones(shape_singlem)

        # couples
        shape_couple = (par.T,par.num_z,par.num_power,par.num_love,par.num_A)
        sol.Vw_couple = np.nan + np.ones(shape_couple)
        sol.Vm_couple = np.nan + np.ones(shape_couple)

        sol.Vw_remain_couple = np.nan + np.ones(shape_couple)
        sol.Vm_remain_couple = np.nan + np.ones(shape_couple)
        sol.p_Vw_remain_couple = np.nan + np.ones(shape_couple)
        sol.p_Vm_remain_couple = np.nan + np.ones(shape_couple)
        sol.n_Vw_remain_couple = np.nan + np.ones(shape_couple)
        sol.n_Vm_remain_couple = np.nan + np.ones(shape_couple)      
        sol.p_C_tot_remain_couple = np.nan + np.ones(shape_couple)
        sol.n_C_tot_remain_couple = np.nan + np.ones(shape_couple)
        sol.remain_WLP = np.ones(shape_couple)
        
        sol.power_idx = np.zeros(shape_couple,dtype=np.int_)
        sol.power = np.zeros(shape_couple)

        # EGM
        sol.marg_Vw_couple = np.zeros(shape_couple)
        sol.marg_Vm_couple = np.zeros(shape_couple)
        sol.marg_Vw_remain_couple = np.zeros(shape_couple)
        sol.marg_Vm_remain_couple = np.zeros(shape_couple)

        # pre-compute optimal consumption allocation
        shape_pre = (par.num_wlp,par.num_power,par.num_Ctot)
        sol.pre_Ctot_Cw_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_Cm_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_C_pub = np.nan + np.ones(shape_pre)
        
        # simulation
        shape_sim = (par.simN,par.simT)
        sim.Cw_priv = np.nan + np.ones(shape_sim)
        sim.Cm_priv = np.nan + np.ones(shape_sim)
        sim.Cw_pub = np.nan + np.ones(shape_sim)
        sim.Cm_pub = np.nan + np.ones(shape_sim)
        sim.Cw_tot = np.nan + np.ones(shape_sim)
        sim.Cm_tot = np.nan + np.ones(shape_sim)
        sim.C_tot = np.nan + np.ones(shape_sim)
        sim.iz = np.ones(shape_sim,dtype=np.int_)
        
        sim.A = np.nan + np.ones(shape_sim)
        sim.Aw = np.nan + np.ones(shape_sim)
        sim.Am = np.nan + np.ones(shape_sim)
        sim.couple = np.nan + np.ones(shape_sim)
        sim.power_idx = np.ones(shape_sim,dtype=np.int_)
        sim.power = np.nan + np.ones(shape_sim)
        sim.love = np.nan + np.ones(shape_sim)
        sim.incw = np.nan + np.ones(shape_sim)
        sim.incm = np.nan + np.ones(shape_sim)
        sim.WLP = np.ones(shape_sim,dtype=np.int_)

        # shocks
        np.random.seed(par.seed)
        sim.draw_love = np.random.normal(size=shape_sim)
        sim.shock_z=np.random.random_sample((par.simN,par.simT))
        sim.shock_z_init=np.random.random_sample((2,par.simN))
        sim.shock_taste=np.random.random_sample((par.simN,par.simT))

        # initial distribution
        sim.init_A = par.grid_A[0] + np.zeros(par.simN)
        sim.init_Aw = par.div_A_share * sim.init_A #np.zeros(par.simN)
        sim.init_Am = (1.0 - par.div_A_share) * sim.init_A #np.zeros(par.simN)
        sim.init_couple = np.ones(par.simN,dtype=bool)
        sim.init_power_idx = par.num_power//2 * np.ones(par.simN,dtype=np.int_)
        sim.init_love = np.zeros(par.simN)
        sim.init_zw = np.ones(par.simN,dtype=np.int_)*par.num_zw//2
        sim.init_zm = np.ones(par.simN,dtype=np.int_)*par.num_zm//2
        

    def setup_grids(self):
        par = self.par
        
        # wealth. Single grids are such to avoid interpolation
        par.grid_A = par.grid_A = np.linspace(0.0,par.max_A,par.num_A)#nonlinspace(0.0,par.max_A,par.num_A,1.1)
        par.grid_Aw = par.div_A_share * par.grid_A
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # power. non-linear grid with more mass in both tails.
        odd_num = np.mod(par.num_power,2)
        first_part = nonlinspace(0.01,0.5,(par.num_power+odd_num)//2,1.3)
        last_part = np.flip(.99 - nonlinspace(0.01,0.5,(par.num_power-odd_num)//2 + 1,1.3))[1:]
        par.grid_power = np.append(first_part,last_part)

        # love grid and shock
        if par.num_love>1:
            par.grid_love = np.linspace(-par.max_love,par.max_love,par.num_love)#+200.0
        else:
            par.grid_love = np.array([0.0])#+200.0

        if par.sigma_love<=1.0e-6:
            par.num_shock_love = 1
            par.grid_shock_love,par.grid_weight_love = np.array([0.0]),np.array([1.0])

        else:
            par.grid_shock_love,par.grid_weight_love = quadrature.normal_gauss_hermite(par.sigma_love,par.num_shock_love)

        # grid for women's labor supply
        par.grid_wlp=np.array([0.0,1.0])
        par.num_wlp=len(par.grid_wlp)
        
        # pre-computation
        par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)

        # EGM 
        par.grid_inv_marg_u = np.flip(par.grid_Ctot)
        par.grid_marg_u = np.nan + np.ones((par.num_wlp,par.num_power,par.num_Ctot))# couples
        par.grid_marg_uw = np.nan + np.ones((par.num_wlp,par.num_power,par.num_Ctot))# couples
        par.grid_marg_um = np.nan + np.ones((par.num_wlp,par.num_power,par.num_Ctot))# couples
        par.grid_marg_u_for_inv = np.nan + np.ones(par.grid_marg_u.shape)# couples
        par.grid_u_s =  np.nan + np.ones(par.num_Ctot)# singles
        par.grid_marg_u_s = np.nan + np.ones(par.num_Ctot)# singles
        par.grid_marg_u_for_inv_s = np.nan + np.ones(par.num_Ctot)# singles

        
        # income shocks grids: singles and couples
        par.grid_zw,par.Π_zw= usr.labor_income(par.t0w,par.t1w,par.t2w,par.T,par.Tr,par.σzw,par.σ0zw,par.num_zw)
        par.grid_zm,par.Π_zm= usr.labor_income(par.t0m,par.t1m,par.t2m,par.T,par.Tr,par.σzm,par.σ0zm,par.num_zm)
        par.Π=[np.kron(par.Π_zw[t],par.Π_zm[t]) for t in range(par.T-1)]
        for t in range(par.Tr,par.T):par.grid_zw[t][:]=1e-8
        par.matw0=usr.rouw_nonst(2,par.σ0zw,0.0,par.num_zw)[1][0]
        par.matm0=usr.rouw_nonst(2,par.σ0zw,0.0,par.num_zw)[1][0]
        
    def solve(self):

        with jit(self) as model:#This allows passing sol and par to jiit functions  
            
            #Import parameters and arrays for solution
            par = model.par; sol = model.sol
            
            # precompute the optimal intra-temporal consumption allocation given total consumpotion
            solve_intraperiod(sol,par)
            
            # loop backwards and obtain policy functions
            for t in reversed(range(par.T)):
                
                # choose EGM or vhi method to solve the single's problem
                solve_single_egm(sol,par,t) if par.EGM else vfi.solve_single(sol,par,t)
                
                # solve the couple's problem
                solve_couple(sol,par,t)
    
                     
    def simulate(self):
        
        with jit(self) as model:    
            
            #Import parameter, policy functions and simulations arrays
            par = model.par; sol = model.sol; sim = model.sim
            
            #Call routing performing the simulation
            simulate_lifecycle(sim,sol,par)
            
  
########################################
# INTRAPERIOD OPTIMIZATION FOR COUPLES #
########################################
@njit(parallel=parallel)
def solve_intraperiod(sol,par):
        
    
    # unpack to help numba (horrible)
    C_pub,  Cw_priv, Cm_priv, grid_marg_u, grid_marg_u_for_inv, grid_marg_u_s, grid_marg_u_for_inv_s, grid_u_s, grid_marg_uw, grid_marg_um =\
        sol.pre_Ctot_C_pub, sol.pre_Ctot_Cw_priv, sol.pre_Ctot_Cm_priv, par.grid_marg_u, par.grid_marg_u_for_inv, par.grid_marg_u_s,\
        par.grid_marg_u_for_inv_s, par.grid_u_s, par.grid_marg_uw, par.grid_marg_um
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb,True)
    
    ϵ = par.grid_Ctot[0]/2.0 # to compute numerical deratives
    #grid2=np.ones(grid_marg_u_s.shape)
    #Dfun=grad(lambda x,*pars: -usr.optimizer(usr.obj_s,1.0e-8, C_tot-1.0e-8,args=(x,*pars[:-1]))[1])
    ################ Singles part: only marg_util for EGM #####################
    for i,C_tot in enumerate(par.grid_Ctot):
        # get util from consumption and the numerical derivative(only needed for EGM...)
        grid_u_s[i] = -usr.optimizer(usr.obj_s,1.0e-8, C_tot-1.0e-8,args=(C_tot,*pars[:-1]))[1]
        forward  =    -usr.optimizer(usr.obj_s,1.0e-8, C_tot+ϵ-1.0e-8,args=(C_tot+ϵ,*pars[:-1]))[1]
        backward =    -usr.optimizer(usr.obj_s,1.0e-8, C_tot-ϵ-1.0e-8,args=(C_tot-ϵ,*pars[:-1]))[1]
        grid_marg_u_s[i] = (forward - backward)/(2*ϵ)
        
        #x=usr.optimizer(usr.obj_s,1.0e-8, C_tot-1.0e-8,args=(C_tot,*pars[:-1]))[1]
        #grid2[i]= Dfun(C_tot,*pars) 
    #Create grid of inverse marginal utility 
    grid_marg_u_for_inv_s=np.flip(grid_marg_u_s) 
  
      
    ################ Couples part ##########################
    bounds = np.array([[0.0,1.0],[0.0,1.0]]) # minimization bounds
    for iP in prange(par.num_power):
        
        x0 = np.array([0.33,0.33])#initial condition (to be updated)
        power=par.grid_power[iP]
        
        for iwlp,wlp in enumerate(par.grid_wlp):
            for i,C_tot in enumerate(par.grid_Ctot):
                
                # estimate
                res = nelder_mead(usr.couple_util,x0,bounds=bounds,args=(C_tot,power,1.0-wlp,*pars),tol_f=1e-10,tol_x=1e-10)

                # unpack
                resx=np.array([0.5,0.5])
                Cw_priv[iwlp,iP,i]= resx[0]*C_tot
                Cm_priv[iwlp,iP,i]= resx[1]*C_tot
                C_pub[iwlp,iP,i] = C_tot - Cw_priv[iwlp,iP,i] - Cm_priv[iwlp,iP,i]
                
                # update initial conidtion
                x0=res.x if i<par.num_Ctot-1 else np.array([0.33,0.33])
                
                # get the numerical derivative(only needed for EGM...)
                forward  = nelder_mead(usr.couple_util,res.x,bounds=bounds,args=(C_tot+ϵ,power,1.0-wlp,*pars),tol_f=1e-10,tol_x=1e-10)
                backward = nelder_mead(usr.couple_util,res.x,bounds=bounds,args=(C_tot-ϵ,power,1.0-wlp,*pars),tol_f=1e-10,tol_x=1e-10)
                grid_marg_u[iwlp,iP,i] = (forward.fun - backward.fun)/(2*ϵ)
                
                assert forward.success==True
                assert backward.success==True
                # get individual derivative
                
                forward_w,  forward_m  = usr.couple_util_ind(resx, C_tot+ϵ,power,1.0-wlp,*pars)
                backward_w, backward_m = usr.couple_util_ind(resx,C_tot-ϵ,power,1.0-wlp,*pars)
                grid_marg_uw[iwlp,iP,i] = 0.5/C_tot#(forward_w - backward_w)/(2*ϵ)
                grid_marg_um[iwlp,iP,i] = 0.5/C_tot#(forward_m - backward_m)/(2*ϵ)
                grid_marg_u[iwlp,iP,i]=power*grid_marg_uw[iwlp,iP,i]+(1.0-power)*grid_marg_um[iwlp,iP,i]
                #assert np.allclose(power*grid_marg_uw[iwlp,iP,i]+(1.0-power)*grid_marg_um[iwlp,iP,i],grid_marg_u[iwlp,iP,i])
            #Create grid of inverse marginal utility 
            grid_marg_u_for_inv[iwlp,iP,:]=np.flip(par.grid_marg_u[iwlp,iP,:])    
        

#######################
# SOLUTIONS - SINGLES #
#######################

@njit
def integrate_single(sol,par,t):
    Ew,Ewd=np.ones((2,par.num_zw,par.num_A));Em,Emd=np.ones((2,par.num_zm,par.num_A)) 
    
    for iA in range(par.num_A):Ew[:,iA]  = sol.Vw_single[t+1,:,iA].flatten() @ par.Π_zw[t] 
    for iA in range(par.num_A):Em[:,iA]  = sol.Vm_single[t+1,:,iA].flatten() @ par.Π_zm[t]
    for iA in range(par.num_A):Ewd[:,iA] = sol.Vw_marg_single[t+1,:,iA].flatten() @ par.Π_zw[t] 
    for iA in range(par.num_A):Emd[:,iA] = sol.Vm_marg_single[t+1,:,iA].flatten() @ par.Π_zm[t]
    
    return Ew,par.β*par.R*Ewd,Em,par.β*par.R*Emd
    
@njit(parallel=parallel)
def solve_single_egm(sol,par,t):

    #Integrate to get continuation value...
    if t<par.T-1:Ew, Ewd, Em, Emd = integrate_single(sol,par,t)
    else:#...unless if you are in the last period
        Ew, Ewd= np.zeros((2,par.num_zw,par.num_A));Em, Emd= np.zeros((2,par.num_zm,par.num_A))
             
    #Pre-define outcomes (if update .sol directly, parallelization go crazy)
    vw,vdw,cw,cwt,Ewt=np.ones((5,par.num_zw,par.num_A));vm,vdm,cm,cmt,Emt=np.ones((5,par.num_zm,par.num_A))
    
    # Women
    for iz in prange(par.num_zw):

        resw = (par.R*par.grid_Aw + par.grid_zw[t,iz])*0.000001
        if t==(par.T-1): cw[iz,:] = resw.copy() #consume all resources
        else: #before T-1 make consumption saving choices
            
            # first get consumption out of grid using FOCs
            linear_interp.interp_1d_vec(np.flip(par.grid_marg_u_s),par.grid_inv_marg_u,Ewd[iz,:],cwt[iz,:])
            
            # use budget constraint to get current resources
            Aw_now = (par.grid_Aw.flatten() + cwt[iz,:] - par.grid_zw[t,iz])/par.R
            
            # now interpolate onto common beginning-of-period asset grid to get consumption
            linear_interp.interp_1d_vec(Aw_now,cwt[iz,:],par.grid_Aw,cw[iz,:])
            
            # get consumption (+make sure that the no-borrowing constraint is respected)
            cw[iz,:] = np.minimum(cw[iz,:], resw.copy())
            
            
        # get utility: interpolate Exp value (line 1), current util (line 2) and add continuation (line 3)#TODO improve precision
        linear_interp.interp_1d_vec(par.grid_Aw,Ew[iz,:], resw.copy()-cw[iz,:],Ewt[iz,:])
        linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_u_s,cw[iz,:],vw[iz,:])
        vw[iz,:]+=par.β*Ewt[iz,:]
            
        # finally get marginal utility from consumption
        linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_marg_u_s,cw[iz,:],vdw[iz,:])
        
    # Men
    for iz in prange(par.num_zm):

        resm = (par.R*par.grid_Am + par.grid_zm[t,iz])*0.000001
        if t==(par.T-1): cm[iz,:] = resm.copy() #consume all resources
        else: #before T-1 make consumption saving choices
            
            # first get consumption out of grid using FOCs
            linear_interp.interp_1d_vec(np.flip(par.grid_marg_u_s),par.grid_inv_marg_u,Emd[iz,:],cmt[iz,:])
            
            # use budget constraint to get current resources
            Am_now = (par.grid_Am.flatten() + cmt[iz,:] - par.grid_zm[t,iz])/par.R
            
            # now interpolate onto common beginning-of-period asset grid to get consumption
            linear_interp.interp_1d_vec(Am_now,cwt[iz,:],par.grid_Am,cm[iz,:])
            
            # get consumption (+make sure that the no-borrowing constraint is respected)
            cm[iz,:] = np.minimum(cm[iz,:], resm.copy())
            
        # get utility: interpolate Exp value (line 1), current util (line 2) and add continuation (line 3)#TODO improve precision
        linear_interp.interp_1d_vec(par.grid_Am,Em[iz,:],resm.copy() - cm[iz,:],Emt[iz,:])
        linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_u_s,cm[iz,:],vm[iz,:])
        vm[iz,:]+=par.β*Emt[iz,:]
            
        # finally get marginal utility from consumption
        linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_marg_u_s,cm[iz,:],vdm[iz,:])         
                
    sol.Vw_single[t,:,:]=vw.copy();sol.Vm_single[t,:,:]=vm.copy()
    sol.Cw_tot_single[t,:,:] = cw.copy() ; sol.Cm_tot_single[t,:,:] = cm.copy()
    sol.Vw_marg_single[t,:,:] = vdw.copy(); sol.Vm_marg_single[t,:,:] = vdm.copy();

#################################################
# SOLUTION - COUPLES
################################################

def solve_couple(sol,par,t):#Solve the couples's problem, choose EGM of VFI techniques
 
    # solve the couple's problem: choose your fighter
    if par.EGM: tuple_with_outcomes =     solve_remain_couple_egm(par,sol,t)# EGM solution        
    else:       tuple_with_outcomes = vfi.solve_remain_couple(par,sol,t)    # vfi solution
              
    #Check participation constrints and eventually update policy function and pareto weights
    check_participation_constraints(*tuple_with_outcomes,par,sol,t)
    
    
@njit#this should be parallelized, but it's tricky... 
def integrate(par,sol,t): 
     
    EVw,EVm,pEVw,pEVm,pEVdw,pEVdm,EVdw,EVdm= np.ones((8,par.num_z,par.num_love,par.num_power,par.num_A)) 
    aw,am,adw,adm=np.ones((4,par.num_shock_love)) 
    love_next_vec=np.ones((par.num_love,par.num_shock_love)) 
     
    for iL in range(par.num_love):love_next_vec[iL,:] = par.grid_love[iL] + par.grid_shock_love 
     
    for iL in range(par.num_love): 
        for iP in range(par.num_power):     
            for iA in range(par.num_A): 
                 
                #Integrate income shocks
                pEVw[:,iL,iP,iA] =  sol.Vw_couple[t+1,:,iP,iL,iA].flatten() @ par.Π[t] 
                pEVm[:,iL,iP,iA] =  sol.Vm_couple[t+1,:,iP,iL,iA].flatten() @ par.Π[t] 
                pEVdw[:,iL,iP,iA] = sol.marg_Vw_couple[t+1,:,iP,iL,iA].flatten() @ par.Π[t]
                pEVdm[:,iL,iP,iA] = sol.marg_Vm_couple[t+1,:,iP,iL,iA].flatten() @ par.Π[t]
   
    for iL in range(par.num_love): 
        for iP in range(par.num_power):     
            for iA in range(par.num_A): 
                for iz in range(par.num_z): 
                    #Integrate love shocks
                    linear_interp.interp_1d_vec(par.grid_love,pEVw[iz,:,iP,iA],love_next_vec[iL,:],aw) 
                    linear_interp.interp_1d_vec(par.grid_love,pEVm[iz,:,iP,iA],love_next_vec[iL,:],am)
                    linear_interp.interp_1d_vec(par.grid_love,pEVdw[iz,:,iP,iA],love_next_vec[iL,:],adw)
                    linear_interp.interp_1d_vec(par.grid_love,pEVdm[iz,:,iP,iA],love_next_vec[iL,:],adm)
                     
                    EVw[iz,iL,iP,iA] =  aw.flatten() @ par.grid_weight_love 
                    EVm[iz,iL,iP,iA] =  am.flatten() @ par.grid_weight_love 
                    EVdw[iz,iL,iP,iA]= adw.flatten() @ par.grid_weight_love
                    EVdm[iz,iL,iP,iA]= adm.flatten() @ par.grid_weight_love
           
    return EVw,EVm,par.R*par.β*EVdw,par.R*par.β*EVdm


@njit(parallel=parallel) 
def solve_remain_couple_egm(par,sol,t): 
 
    #Integration    
    if t<(par.T-1): EVw, EVm, EVdw, EVdm = integrate(par,sol,t)
 
    # initialize 
    Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_C_tot,n_C_tot,wlp,Vwd,Vmd = np.zeros((11,par.num_z,par.num_love,par.num_A,par.num_power)) 

    #parameters: useful to unpack this to improve speed 
    couple=1.0;ishome=1.0-par.grid_wlp[0] 
     
    pars2=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb) 
     
      
    for iL in prange(par.num_love): 
        for iP in range(par.num_power):            
            for iz in range(par.num_z):  
                
                izw=iz//par.num_zm;izm=iz%par.num_zw# indexes 
                love=par.grid_love[iL]
                power = par.grid_power[iP]
                          
                # continuation values 
                if t==(par.T-1):#last period 
                    
                 
                    #Get consumption here
                    n_C_tot[iz,iL,:,iP] = usr.resources_couple(t,par.Tr,par.grid_A,par.grid_zw[t,izw],0.0,par.grid_zm[t,izm],par.R) #No savings, it's the last period 
                     
                    # current utility from consumption allocation 
                    Cw_priv, Cm_priv, C_pub =\
                        intraperiod_allocation_vector(n_C_tot[iz,iL,:,iP],par.num_A,par.grid_Ctot,sol.pre_Ctot_Cw_priv[0,iP],sol.pre_Ctot_Cm_priv[0,iP]) 
                    Vw[iz,iL,:,iP] = n_Vw[iz,iL,:,iP] =usr.util(Cw_priv,C_pub,*pars2,par.grid_love[iL],couple,ishome) 
                    Vm[iz,iL,:,iP] = n_Vm[iz,iL,:,iP] = usr.util(Cm_priv,C_pub,*pars2,par.grid_love[iL],couple,ishome) 
                    wlp[iz,iL,:,iP] = 0.0;p_Vm[iz,iL,:,iP]=p_Vw[iz,iL,:,iP]=-1e10
                    
                    #Now get marginal utility of consumption
                    linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_marg_uw[0,iP,:],n_C_tot[iz,iL,:,iP],Vwd[iz,iL,:,iP])
                    linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_marg_um[0,iP,:],n_C_tot[iz,iL,:,iP],Vmd[iz,iL,:,iP])
                    
                else:#periods before the last 
                             
                    EVd = power*EVdw[iz,iL,iP,:]+(1.0-power)*EVdm[iz,iL,iP,:]
                    
                    # compute optimal consumption and utility given partecipation (0/1).
                    # the last 3 arguments of fun below are the output at iz,iL,iP
                    compute_couple(par,sol,EVd,t,iz,iL,iP,EVw,EVm,1,p_C_tot,p_Vw,p_Vm) # participation
                    compute_couple(par,sol,EVd,t,iz,iL,iP,EVw,EVm,0,n_C_tot,n_Vw,n_Vm) # no participation
                    
                    # if optimal assets are crossing, use upper-envolop algorithm. any?
                    #if np.any(np.diff(n_A_now)<0): bbb=upper_envelop(par,sol,n_A_now,t,iz,iL,iP,EVw,EVm,0,n_C_tot,n_Vw,n_Vm)
                    #if np.any(np.diff(p_A_now)<0): aaa=upper_envelop(par,sol,p_A_now,t,iz,iL,iP,EVw,EVm,1,p_C_tot,p_Vw,p_Vm)
                    
                    
                    # compute participation choices below
                    p_Vwd, n_Vwd,p_Vmd, n_Vmd = np.ones((4,par.num_A))
                    p_v_couple = power*p_Vw[iz,iL,:,iP]+(1.0-power)*p_Vm[iz,iL,:,iP]
                    n_v_couple = power*n_Vw[iz,iL,:,iP]+(1.0-power)*n_Vm[iz,iL,:,iP]
                                  
                    
                    # get the probabilit of working, baed on couple utility choices
                    c=np.maximum(p_v_couple,n_v_couple)/par.σ
                    v_couple=par.σ*(c+np.log(np.exp(p_v_couple/par.σ-c)+np.exp(n_v_couple/par.σ-c)))
                    wlp[iz,iL,:,iP]=np.exp(p_v_couple/par.σ-v_couple/par.σ)#np.minimum(np.maximum(np.exp(p_v_couple/par.σ-v_couple/par.σ) ,1e-6),1.0-1e-6)# probability of optimal labor supply
                    
                    # now the value of making the choice see Shepard (2019), page 11
                    Δp=p_Vw[iz,iL,:,iP]-p_Vm[iz,iL,:,iP];Δn=n_Vw[iz,iL,:,iP]-n_Vm[iz,iL,:,iP]
                    Vw[iz,iL,:,iP]=v_couple+(1.0-par.grid_power[iP])*(wlp[iz,iL,:,iP]*Δp+(1.0-wlp[iz,iL,:,iP])*Δn)
                    Vm[iz,iL,:,iP]=v_couple+(-par.grid_power[iP])   *(wlp[iz,iL,:,iP]*Δp+(1.0-wlp[iz,iL,:,iP])*Δn)
 
                    # finally compute marginal utility of consumption
                    linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_marg_uw[1,iP,:],p_C_tot[iz,iL,:,iP],p_Vwd)
                    linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_marg_uw[0,iP,:],n_C_tot[iz,iL,:,iP],n_Vwd)
                    
                    linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_marg_um[1,iP,:],p_C_tot[iz,iL,:,iP],p_Vmd)
                    linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_marg_um[0,iP,:],n_C_tot[iz,iL,:,iP],n_Vmd)
                    
                    Vwd[iz,iL,:,iP]=wlp[iz,iL,:,iP]*p_Vwd+(1.0-wlp[iz,iL,:,iP])*n_Vwd
                    Vmd[iz,iL,:,iP]=wlp[iz,iL,:,iP]*p_Vmd+(1.0-wlp[iz,iL,:,iP])*n_Vmd
                    

                    #if t==6:Vmd[iz,iL,:,iP,0,0]=0.0
                        
 
    # return objects 
    return (Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_C_tot,n_C_tot, wlp, Vwd, Vmd)
 
@njit   
def compute_couple(par,sol,EVd,t,iz,iL,iP,EVw,EVm,part,p_C_tot,p_Vw,p_Vm):

    izw=iz//par.num_zm;izm=iz%par.num_zw# indexes 
    love=par.grid_love[iL]; power = par.grid_power[iP]
    
    p_C_pd,p_Ew,p_Em,p_Vwd,p_Vmd,_= np.ones((6,par.num_A))        
    
    p_resources = usr.resources_couple(t,par.Tr,par.grid_A,par.grid_zw[t,izw],par.grid_wlp[part],par.grid_zm[t,izm],par.R) 
    
    pars2=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb) 
    pars=(par,sol,iP,EVw[iz,iL,iP,:],EVm[iz,iL,iP,:],part,power,love,pars2)
    
    # first get consumption out of grid using FOCs  
    linear_interp.interp_1d_vec(par.grid_marg_u_for_inv[part,iP,:],par.grid_inv_marg_u,EVd,p_C_pd)
    
    # use budget constraint to get current resources
    p_A_now =  par.grid_A.flatten() + p_C_pd#(2*par.grid_A.flatten() + p_C_pd - p_resources)/par.R
    
    # if (t==12) & (iP==1):
    #     import matplotlib.pyplot as plt
    #     p_Vm[iz,iL,:,iP,0,0]=14
    
    if 1>0:#np.any(np.diff(p_A_now)<0):#apply upperenvelope + enforce no borrowing constraint

        upper_envelope(par.grid_A,p_A_now,p_C_pd,par.β*EVw[iz,iL,iP,:],par.β*EVm[iz,iL,iP,:],power,p_resources,
                       p_C_tot[iz,iL,:,iP],p_Vw[iz,iL,:,iP],p_Vm[iz,iL,:,iP],*pars)

        
    else:#upperenvelope not necessary: enforce no borrowing constraint
    
        # now interpolate onto common beginning-of-period asset grid to get consumption
        linear_interp.interp_1d_vec(p_A_now,p_C_pd,p_resources,p_C_tot[iz,iL,:,iP] )

        p_C_tot[iz,iL,:,iP] = np.minimum(p_C_tot[iz,iL,:,iP] , p_resources)
    
        # value of choices below: first participation...
        p_Cw_priv, p_Cm_priv, p_C_pub =\
            intraperiod_allocation_vector(p_C_tot[iz,iL,:,iP] ,par.num_A,par.grid_Ctot,sol.pre_Ctot_Cw_priv[part,iP],sol.pre_Ctot_Cm_priv[part,iP]) 
            
        linear_interp.interp_1d_vec(par.grid_A,EVw[iz,iL,iP,:],p_resources-p_C_tot[iz,iL,:,iP] ,p_Ew)
        linear_interp.interp_1d_vec(par.grid_A,EVm[iz,iL,iP,:],p_resources-p_C_tot[iz,iL,:,iP] ,p_Em)
        
        p_Vw[iz,iL,:,iP] = usr.util(p_Cw_priv,p_C_pub,*pars2,love,True,1.0-par.grid_wlp[part])+par.β*p_Ew                        
        p_Vm[iz,iL,:,iP] = usr.util(p_Cm_priv,p_C_pub,*pars2,love,True,1.0-par.grid_wlp[part])+par.β*p_Em
    

    if (t>=par.Tr) & (part):p_Vw[iz,iL,:,iP]=p_Vw[iz,iL,:,iP]=-1e10 
    #return p_C_tot[iz,iL,:,iP], p_Vw[iz,iL,:,iP], p_Vm[iz,iL,:,iP]
                    
                 #   if np.minimum(np.diff(n_A_now))<0:# call upper envelop if crossing in optimal assets
  
#Below upper envelop to eliminate points that satisfy FOCs but are not global
#optimum. This follows Iskhakov et al 2017
@njit
def upper_envelop(par,sol,ae,t,iz,iL,iP,EVw,EVm,part,p_C_tot,p_Vw,p_Vm):
    
    izw=iz//par.num_zm;izm=iz%par.num_zw# indexes 
    resources = usr.resources_couple(t,par.Tr,par.grid_A,par.grid_zw[t,izw],par.grid_wlp[part],par.grid_zm[t,izm],par.R)
    pars2=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb) 
    love=par.grid_love[iL]; power = par.grid_power[iP]
    
    p_Ew,p_Em= np.ones((2,par.num_A))
    
    for j in range(par.num_A-1):
        for i in range(par.num_A):
        
            if  (par.grid_A[i]>ae[j]) & (par.grid_A[i]<ae[j+1]):
                print(j,i)
                opt_a_new=np.interp(par.grid_A[i],ae[j:j+2],par.grid_A[j:j+2])
                opt_c_new=resources[i]-opt_a_new
                
                vw_new,vm_new,v_new= \
                    usr.value_of_choice(opt_c_new,par,sol,iP,EVw[iz,iL,iP,:],EVm[iz,iL,iP,:],part,power,love,resources[i]-opt_c_new,pars2)
                
                 
                if (v_new>power*p_Vw[iz,iL,i,iP]+(1.0-power)*p_Vm[iz,iL,i,iP]): 
                    print(j,i,111)
                    p_C_tot[iz,iL,i,iP]=opt_c_new.copy()
                    p_Vw[iz,iL,i,iP] = vw_new.copy()
                    p_Vm[iz,iL,i,iP] = vm_new.copy()
                    
    return p_C_tot[iz,iL,:,iP]




@njit
def intraperiod_allocation_vector(C_tot,num_Ctot,grid_Ctot,pre_Ctot_Cw_priv,pre_Ctot_Cm_priv):

    # interpolate pre-computed solution
    Cw_priv,Cm_priv=np.ones((2,num_Ctot))
    
    linear_interp.interp_1d_vec(grid_Ctot,pre_Ctot_Cw_priv,C_tot,Cw_priv)
    linear_interp.interp_1d_vec(grid_Ctot,pre_Ctot_Cm_priv,C_tot,Cm_priv)

    return Cw_priv, Cm_priv, C_tot - Cw_priv - Cm_priv

                

@njit(parallel=parallel)
def check_participation_constraints(remain_Vw,remain_Vm,p_remain_Vw,p_remain_Vm,n_remain_Vw,n_remain_Vm,p_C_tot,n_C_tot, remain_wlp,remain_Vwd,remain_Vmd,par,sol,t):
 
    power_idx=sol.power_idx
    power=sol.power
            
    for iL in prange(par.num_love):
        for iA in range(par.num_A):
            for iz in range(par.num_z):
               
                # check the participation constraints
                idx_single_w = (t,iz//par.num_zm,iA);idx_single_m = (t,iz%par.num_zw,iA);idd=(iz,iL,iA)
     
                
                list_couple = (sol.Vw_couple, sol.Vm_couple,sol.marg_Vw_couple,sol.marg_Vm_couple)#list_start_as_couple
                list_raw    = (remain_Vw[idd],remain_Vm[idd],remain_Vwd[idd],remain_Vmd[idd])#list_remain_couple
                list_single = (sol.Vw_single,sol.Vm_single,sol.Vw_marg_single,sol.Vm_marg_single) # list_trans_to_single: last input here not important in case of divorce
                list_single_w = (True,False,True,False) # list that say whether list_single[i] is about w (as oppoesd to m)
                
                Sw = remain_Vw[iz,iL,iA,:] - sol.Vw_single[idx_single_w] 
                Sm = remain_Vm[iz,iL,iA,:] - sol.Vm_single[idx_single_m] 
                            
                # check the participation constraints. Array
                min_Sw = np.min(Sw)
                min_Sm = np.min(Sm)
                max_Sw = np.max(Sw)
                max_Sm = np.max(Sm)
            
                if (min_Sw >= 0.0) & (min_Sm >= 0.0): # all values are consistent with marriage
                    for iP in range(par.num_power):
            
                        # overwrite output for couple
                        idx = (t,iz,iP,iL,iA)
                        for i,key in enumerate(list_couple):
                            list_couple[i][idx] = list_raw[i][iP]
            
                        power_idx[idx] = iP
                        power[idx] = par.grid_power[iP]
            
                elif (max_Sw < 0.0) | (max_Sm < 0.0): # no value is consistent with marriage
                    for iP in range(par.num_power):
            
                        # overwrite output for couple
                        idx = (t,iz,iP,iL,iA)
                        for i,key in enumerate(list_couple):
                            list_couple[i][idx] = list_single[i][idx_single_w] if list_single_w[i] else list_single[i][idx_single_m]
            
                        power_idx[idx] = -1
                        power[idx] = -1.0
            
                else: 
                
                    # find lowest (highest) value with positive surplus for women (men)
                    Low_w = 1 #0 # in case there is no crossing, this will be the correct value
                    Low_m = par.num_power-1-1 #par.num_power-1 # in case there is no crossing, this will be the correct value
                    for iP in range(par.num_power-1):
                        if (Sw[iP]<0) & (Sw[iP+1]>=0):
                            Low_w = iP+1
                            
                        if (Sm[iP]>=0) & (Sm[iP+1]<0):
                            Low_m = iP
            
                    # b. interpolate the surplus of each member at indifference points
                    # women indifference
                    id = Low_w-1
                    denom = (par.grid_power[id+1] - par.grid_power[id])
                    ratio_w = (Sw[id+1] - Sw[id])/denom
                    ratio_m = (Sm[id+1] - Sm[id])/denom
                    power_at_zero_w = par.grid_power[id] - Sw[id]/ratio_w
                    Sm_at_zero_w = Sm[id] + ratio_m*( power_at_zero_w - par.grid_power[id] )
            
                    # men indifference
                    id = Low_m
                    denom = (par.grid_power[id+1] - par.grid_power[id])
                    ratio_w = (Sw[id+1] - Sw[id])/denom
                    ratio_m = (Sm[id+1] - Sm[id])/denom
                    power_at_zero_m = par.grid_power[id] - Sm[id]/ratio_m
                    Sw_at_zero_m = Sw[id] + ratio_w*( power_at_zero_m - par.grid_power[id] )
                    
    
    
                    # update the outcomes
                    for iP in range(par.num_power):
            
                        # index to store solution for couple 
                        idx = (t,iz,iP,iL,iA)
                
                        # woman wants to leave
                        if iP<Low_w: 
                            if Sm_at_zero_w > 0: # man happy to shift some bargaining power
            
                                for i,key in enumerate(list_couple):
                                    if iP==0:
                                        list_couple[i][idx] = linear_interp_1d._interp_1d(par.grid_power,list_raw[i],power_at_zero_w,Low_w-1) 
                                    else:
                                        list_couple[i][idx] = list_couple[i][(t,iz,0,iL,iA)]; # re-use that the interpolated values are identical
            
                                power_idx[idx] = Low_w
                                power[idx] = power_at_zero_w
                                
                            else: # divorce
            
                                for i,key in enumerate(list_couple):
                                    list_couple[i][idx] = list_single[i][idx_single_w] if list_single_w[i] else list_single[i][idx_single_m]
            
                                power_idx[idx] = -1
                                power[idx] = -1.0
                            
                        # man wants to leave
                        elif iP>Low_m: 
                            if Sw_at_zero_m > 0: # woman happy to shift some bargaining power
                                
                                for i,key in enumerate(list_couple):
                                    if (iP==(Low_m+1)):
                                        list_couple[i][idx] = linear_interp_1d._interp_1d(par.grid_power,list_raw[i],power_at_zero_m,Low_m) 
                                    else:
                                        list_couple[i][idx] = list_couple[i][(t,iz,Low_m+1,iL,iA)]; # re-use that the interpolated values are identical
            
                                power_idx[idx] = Low_m
                                power[idx] = power_at_zero_m
                                
                            else: # divorce
            
                                for i,key in enumerate(list_couple):
                                    list_couple[i][idx] = list_single[i][idx_single_w] if list_single_w[i] else list_single[i][idx_single_m]
            
                                power_idx[idx] = -1
                                power[idx] = -1.0
            
                        else: # no-one wants to leave
            
                            for i,key in enumerate(list_couple):
                                list_couple[i][idx] = list_raw[i][iP]
            
                            power_idx[idx] = iP
                            power[idx] = par.grid_power[iP]
                        
                        
    # save remain values
    for iL in range(par.num_love):
        for iA in range(par.num_A):
            for iP in range(par.num_power):
                for iz in range(par.num_z):
                    idx = (t,iz,iP,iL,iA);idz = (iz,iL,iA,iP)

                    sol.p_C_tot_remain_couple[idx] = p_C_tot[idz]
                    sol.n_C_tot_remain_couple[idx] = n_C_tot[idz]
                    sol.Vw_remain_couple[idx] = remain_Vw[idz]
                    sol.Vm_remain_couple[idx] = remain_Vm[idz]
                    sol.p_Vw_remain_couple[idx] = p_remain_Vw[idz]
                    sol.p_Vm_remain_couple[idx] = p_remain_Vm[idz]
                    sol.n_Vw_remain_couple[idx] = n_remain_Vw[idz]
                    sol.n_Vm_remain_couple[idx] = n_remain_Vm[idz]
                    sol.remain_WLP[idx] = remain_wlp[idz]
                    sol.marg_Vw_remain_couple[idx] = remain_Vwd[idz]
                    sol.marg_Vm_remain_couple[idx] = remain_Vmd[idz]


##################################
#        SIMULATIONS
#################################

@njit(parallel=parallel)
def simulate_lifecycle(sim,sol,par):     #TODO: updating power should be continuous...
    

    # unpacking some values to help numba optimize
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)#single
    A=sim.A;Aw=sim.Aw; Am=sim.Am;Cw_priv=sim.Cw_priv;Cw_pub=sim.Cw_pub;Cm_priv=sim.Cm_priv;Cm_pub=sim.Cm_pub
    couple=sim.couple;power_idx=sim.power_idx;power=sim.power;love=sim.love;draw_love=sim.draw_love;iz=sim.iz;
    wlp=sim.WLP;incw=sim.incw;incm=sim.incm
    
    
    for i in prange(par.simN):
        for t in range(par.simT):
            
            # initial condition
            if t==0:
                A_lag = sim.init_A[i]
                Aw_lag = sim.init_Aw[i]
                Am_lag = sim.init_Am[i]
                couple_lag = sim.init_couple[i]
                power_idx_lag = sim.init_power_idx[i]
                love[i,t] = sim.init_love[i]
                iz_w = usr.mc_simulate(par.num_zw//2,par.matw0,sim.shock_z_init[0,i])
                iz_m = usr.mc_simulate(par.num_zm//2,par.matm0,sim.shock_z_init[1,i])
                iz[i,t] = iz_w*par.num_zm+iz_m
            else:
                A_lag = A[i,t-1]
                Aw_lag = Aw[i,t-1]
                Am_lag = Am[i,t-1]
                couple_lag = couple[i,t-1]
                power_idx_lag = power_idx[i,t-1]
                
                #shocks below
                love[i,t] = love[i,t-1] + par.sigma_love*draw_love[i,t] #useless if single
                iz[i,t] = usr.mc_simulate(iz[i,t-1],par.Π[t-1],sim.shock_z[i,t])
                iz_w=iz[i,t]//par.num_zm;iz_m=iz[i,t]%par.num_zw

            # resources
            incw[i,t]=par.grid_zw[t,iz_w];incm[i,t]=par.grid_zm[t,iz_m]         
           
            
            # first check if they want to remain together and what the bargaining power will be if they do.
            if couple_lag:                   

                # value of transitioning into singlehood
                Vw_single = linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,iz_w],Aw_lag)
                Vm_single = linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,iz_m],Am_lag)

                idx = (t,iz[i,t],power_idx_lag)
                Vw_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[idx],love[i,t],A_lag)
                Vm_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[idx],love[i,t],A_lag)

                if ((Vw_couple_i>=Vw_single) & (Vm_couple_i>=Vm_single)):
                    power_idx[i,t] = power_idx_lag

                else:
                    # value of partnerhip for all levels of power
                    Vw_couple = np.zeros(par.num_power)
                    Vm_couple = np.zeros(par.num_power)
                    for iP in range(par.num_power):
                        Vw_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[t,iz_w,iP],love[i,t],A_lag)
                        Vm_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[t,iz_m,iP],love[i,t],A_lag)

                    # check participation constraint 
                    Sw = Vw_couple - Vw_single
                    Sm = Vm_couple - Vm_single
                    power_idx[i,t] = update_bargaining_index(Sw,Sm,power_idx_lag, par.num_power)

                # infer partnership status
                if power_idx[i,t] < 0.0: # divorce is coded as -1
                    couple[i,t] = False

                else:
                    couple[i,t] = True

            else: # remain single

                couple[i,t] = False

            # update behavior
            if couple[i,t]:
                
                
                # first decide about labor participation
                power[i,t] = par.grid_power[power_idx[i,t]]
                V_p_grid=power[i,t]*sol.p_Vw_remain_couple[t,iz[i,t],power_idx[i,t]]+(1.0-power[i,t])*sol.p_Vm_remain_couple[t,iz[i,t],power_idx[i,t]]
                V_n_grid=power[i,t]*sol.n_Vw_remain_couple[t,iz[i,t],power_idx[i,t]]+(1.0-power[i,t])*sol.n_Vm_remain_couple[t,iz[i,t],power_idx[i,t]]
                V_p=linear_interp.interp_2d(par.grid_love,par.grid_A,V_p_grid,love[i,t],A_lag)
                V_n=linear_interp.interp_2d(par.grid_love,par.grid_A,V_n_grid,love[i,t],A_lag)
                c=np.maximum(V_p,V_n)/par.σ
                v_couple=par.σ*(c+np.log(np.exp(V_p/par.σ-c)+np.exp(V_n/par.σ-c)))
                part_i=np.exp(V_p/par.σ-v_couple/par.σ)#np.minimum(np.maximum(np.exp(p_v_couple/par.σ-v_couple/par.σ) ,1e-6),1.0-1e-6)# probability of optimal labor supply
                wlp[i,t]=(part_i>sim.shock_taste[i,t])
                
                # pr= sol.remain_WLP[t,iz[i,t],power_idx[i,t]] #probability of participation
                # part_i=linear_interp.interp_2d(par.grid_love,par.grid_A,pr,love[i,t],A_lag)
                # wlp[i,t]=(part_i>sim.shock_taste[i,t])
                
                # optimal consumption allocation if couple (note use of the updated index)
                sol_C_tot = sol.p_C_tot_remain_couple[t,iz[i,t],power_idx[i,t]] if wlp[i,t] else sol.n_C_tot_remain_couple[t,iz[i,t],power_idx[i,t]] 
                C_tot = linear_interp.interp_2d(par.grid_love,par.grid_A,sol_C_tot,love[i,t],A_lag)
                args=(C_tot,par.num_Ctot,par.grid_Ctot,sol.pre_Ctot_Cw_priv[wlp[i,t],power_idx[i,t]],sol.pre_Ctot_Cm_priv[wlp[i,t],power_idx[i,t]])
                Cw_priv[i,t], Cm_priv[i,t], C_pub = usr.intraperiod_allocation(*args)
                Cw_pub[i,t] = Cm_pub[i,t] = C_pub
            

                # update end-of-period states
                pres=(incw[i,t],wlp[i,t],incm[i,t],par.R)
                M_resources = usr.resources_couple(t,par.Tr,A_lag,*pres) 
                A[i,t] = M_resources - Cw_priv[i,t] - Cm_priv[i,t] - Cw_pub[i,t]

                # in case of divorce 
                Aw[i,t] = par.div_A_share * A[i,t]
                Am[i,t] = (1.0-par.div_A_share) * A[i,t]

                

            else: # single

               
                # pick relevant solution for single
                sol_single_w = sol.Cw_tot_single[t,iz_w]
                sol_single_m = sol.Cm_tot_single[t,iz_m]

                # optimal consumption allocations
                Cw_tot = linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw_lag)
                Cm_tot = linear_interp.interp_1d(par.grid_Am,sol_single_m,Am_lag)
                
                Cw_priv[i,t],Cw_pub[i,t] = usr.intraperiod_allocation_single(Cw_tot,*pars)
                Cm_priv[i,t],Cm_pub[i,t] = usr.intraperiod_allocation_single(Cm_tot,*pars)

                # update end-of-period states
                Mw = par.R*Aw_lag + incw[i,t] # total resources woman
                Mm = par.R*Am_lag + incm[i,t] # total resources man
                Aw[i,t] = Mw - Cw_priv[i,t] - Cw_pub[i,t] #assets woman
                Am[i,t] = Mm - Cm_priv[i,t] - Cm_pub[i,t] #assets man

    # total consumption
    sim.Cw_tot[::] = sim.Cw_priv[::] + sim.Cw_pub[::]
    sim.Cm_tot[::] = sim.Cm_priv[::] + sim.Cm_pub[::]
    sim.C_tot[::] = sim.Cw_priv[::] + sim.Cm_priv[::] + sim.Cw_pub[::]
       
@njit
def update_bargaining_index(Sw,Sm,iP, num_power):
    
    # check the participation constraints. Array
    min_Sw = np.min(Sw)
    min_Sm = np.min(Sm)
    max_Sw = np.max(Sw)
    max_Sm = np.max(Sm)

    if (min_Sw >= 0.0) & (min_Sm >= 0.0): # all values are consistent with marriage
        return iP

    elif (max_Sw < 0.0) | (max_Sm < 0.0): # no value is consistent with marriage
        return -1

    else: 
    
        # find lowest (highest) value with positive surplus for women (men)
        Low_w = 0 # in case there is no crossing, this will be the correct value
        Low_m = num_power-1 # in case there is no crossing, this will be the correct value
        for _iP in range(num_power-1):
            if (Sw[_iP]<0) & (Sw[_iP+1]>=0):
                Low_w = _iP+1
                
            if (Sm[_iP]>=0) & (Sm[_iP+1]<0):
                Low_m = iP

        # update the outcomes
        # woman wants to leave
        if iP<Low_w: 
            if Sm[Low_w] > 0: # man happy to shift some bargaining power
                return Low_w
                
            else: # divorce
                return -1
            
        # man wants to leave
        elif iP>Low_m: 
            if Sw[Low_m] > 0: # woman happy to shift some bargaining power
                return Low_m
                
            else: # divorce
                return -1

        else: # no-one wants to leave
            return iP