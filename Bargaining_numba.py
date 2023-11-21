import numpy as np#import autograd.numpy as np#
from EconModel import EconModelClass, jit
from consav.grids import nonlinspace
from consav import linear_interp, linear_interp_1d,quadrature
from numba import njit,prange,config
import UserFunctions_numba as usr
import vfi as vfi
from quantecon.optimize.nelder_mead import nelder_mead
import setup
upper_envelope=usr.create(usr.couple_time_utility)

#general configuratiion and glabal variables (common across files)
config.DISABLE_JIT = setup.nojit;parallel=setup.parallel;cache=setup.cache
woman=setup.woman;man=setup.man

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
        
        
        par.EGM = True# Want to use the EGM method to solve the model?
        #Note: to get EGM vs. vfi equivalence max assets should be VERY large
        
        par.R = 1.00#3
        par.β = 1.00# Discount factor
        
        par.div_A_share = 0.5 # divorce share of wealth to wife

        # Utility: CES aggregator or additively linear
        par.ρ = 1.5#        # CRRA      
        par.α1 = 0.55
        par.α2 = 0.45
        par.ϕ1 = 0.2
        par.ϕ2 = (1.0-par.ρ)/par.ϕ1
        
        # production of home good
        par.θ = 0.21 #weight on money vs. time to produce home good
        par.λ = 0.19 #elasticity betwen money and time in public good
        par.tb = 0.2 #time spend on public goods by singles
        
        #Taste shock
        par.σ = 0.02 #taste shock applied to working/not working
        
        ####################
        # state variables
        #####################
        
        par.T = 10 # terminal age
        par.Tr = 6 # age at retirement
        
        # wealth
        par.num_A = 20;par.max_A = 2.0
        
        # bargaining power
        par.num_power = 15

        # love/match quality
        par.num_love = 41;par.max_love = 3.0
        par.sigma_love = 0.1;par.num_shock_love = 5
        
        # income of men and women: gridpoints
        par.num_zw=3;par.num_zm=3; par.num_z=par.num_zm*par.num_zw
        
        # income of men and women: parameters of the age log-polynomial
        par.t0w=-0.5;par.t1w=0.03;par.t2w=0.0;par.t0m=-0.5;par.t1m=0.03;par.t2m=0.0;
    
        # income of men and women: sd of income shocks in t=0 and after that
        par.σzw=0.0001;par.σ0zw=0.00005;par.σzm=0.0001;par.σ0zm=0.00005
        
        # pre-computation fo consumption
        par.num_Ctot = 500;par.max_Ctot = par.max_A
        
        # simulation
        par.seed = 9210;par.simT = par.T;par.simN = 20_000
        
    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # setup grids
        par.simT = par.T
        self.setup_grids()
        
        # singles: value functions (vf), consumption, marg util
        shape_singlew = (par.T,par.num_zw,par.num_A)
        shape_singlem = (par.T,par.num_zm,par.num_A)
        sol.Vw_single = np.nan + np.ones(shape_singlew) #vf in t
        sol.Vm_single = np.nan + np.ones(shape_singlem) #vf in t
        sol.Cw_tot_single = np.nan + np.ones(shape_singlew) #priv+tot cons
        sol.Cm_tot_single = np.nan + np.ones(shape_singlem) #priv+tot cons


        # couples: value functions (vf), consumption, marg util, bargaining power
        shape_couple = (par.T,par.num_z,par.num_power,par.num_love,par.num_A) # 2 is for men/women
        sol.Vw_couple = np.nan + np.ones(shape_couple) #vf in t
        sol.Vm_couple = np.nan + np.ones(shape_couple) #vf in t
        sol.Vw_remain_couple = np.nan + np.ones(shape_couple) #vf|couple
        sol.Vm_remain_couple = np.nan + np.ones(shape_couple) #vf|couple
        sol.p_Vw_remain_couple = np.nan + np.ones(shape_couple)#vf|(couple+part)
        sol.p_Vm_remain_couple = np.nan + np.ones(shape_couple)#vf|(couple+part)
        sol.n_Vw_remain_couple = np.nan + np.ones(shape_couple)#vf|(couple+no part)
        sol.n_Vm_remain_couple = np.nan + np.ones(shape_couple)#vf|(couple+no part)      
        sol.p_C_tot_remain_couple = np.nan + np.ones(shape_couple)#cons|(couple+part)
        sol.n_C_tot_remain_couple = np.nan + np.ones(shape_couple)#cons|(couple+nopart)
        sol.remain_WLP = np.ones(shape_couple)#pr. of participation|couple   
        sol.power_idx =  np.nan +np.zeros(shape_couple,dtype=np.int_)#barg power index
        sol.power =  np.nan +np.zeros(shape_couple)                  #barg power value

        # pre-compute optimal consumption allocation: private and public
        shape_pre = (par.num_wlp,par.num_power,par.num_Ctot)
        sol.pre_Ctot_Cw_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_Cm_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_C_pub = np.nan + np.ones(shape_pre)
        
        # simulation
        shape_sim = (par.simN,par.simT)
        sim.Cw_priv = np.nan + np.ones(shape_sim) #w' priv consumption
        sim.Cm_priv = np.nan + np.ones(shape_sim) #m' priv consumption
        sim.Cw_pub = np.nan + np.ones(shape_sim)  #w' pub consumption
        sim.Cm_pub = np.nan + np.ones(shape_sim)  #m' pub consumption
        sim.Cw_tot = np.nan + np.ones(shape_sim)  #w' priv+pub consumption
        sim.Cm_tot = np.nan + np.ones(shape_sim)  #m' priv+pub consumption
        sim.C_tot = np.nan + np.ones(shape_sim)   #w+m total consumtion
        sim.iz = np.ones(shape_sim,dtype=np.int_) #index of income shocks
        
        sim.A = np.nan + np.ones(shape_sim)             # total assets (m+w)
        sim.Aw = np.nan + np.ones(shape_sim)            # w's assets
        sim.Am = np.nan + np.ones(shape_sim)            # m's assets
        sim.couple = np.nan + np.ones(shape_sim)        # In a couple? True/False
        sim.power_idx = np.ones(shape_sim,dtype=np.int_)# barg power index
        sim.power = np.nan + np.ones(shape_sim)         # barg power
        sim.love = np.nan + np.ones(shape_sim)          # love
        sim.incw = np.nan + np.ones(shape_sim)          # w's income
        sim.incm = np.nan + np.ones(shape_sim)          # m's income
        sim.WLP = np.ones(shape_sim,dtype=np.int_)      # w's labor participation

        # shocks
        np.random.seed(par.seed)
        sim.draw_love = np.random.normal(size=shape_sim)           #love
        sim.shock_z=np.random.random_sample((par.simN,par.simT))   #income
        sim.shock_z_init=np.random.random_sample((2,par.simN))     #income initial
        sim.shock_taste=np.random.random_sample((par.simN,par.simT))#taste shock

        # initial distribution
        sim.init_A = par.grid_A[0] + np.zeros(par.simN)            #total assetes
        sim.init_Aw = par.div_A_share * sim.init_A                 #w's assets
        sim.init_Am = (1.0 - par.div_A_share) * sim.init_A         #m's assets
        sim.init_couple = np.ones(par.simN,dtype=bool)             #state (couple=1/single=0)
        sim.init_power_idx = 3*np.ones(par.simN,dtype=np.int_)#par.num_power//2*np.ones(par.simN,dtype=np.int_)#barg power index
        sim.init_love = np.zeros(par.simN)                         #initial love
        sim.init_zw = np.ones(par.simN,dtype=np.int_)*par.num_zw//2#w's initial income 
        sim.init_zm = np.ones(par.simN,dtype=np.int_)*par.num_zm//2#m's initial income
        

    def setup_grids(self):
        par = self.par
        
        # wealth. Single grids are such to avoid interpolation
        par.grid_A = par.grid_A = np.linspace(0.0,par.max_A,par.num_A)#nonlinspace(0.0,par.max_A,par.num_A,1.1)
        par.grid_Aw = par.div_A_share * par.grid_A
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # bargaining power. non-linear grid with more mass in both tails.        
        par.grid_power = np.linspace(0.01,0.99,par.num_power)#usr.grid_fat_tails(0.01,0.99,par.num_power)#

        # love grid and shock
        if par.num_love>1:par.grid_love = np.linspace(-par.max_love,par.max_love,par.num_love)           
        else:             par.grid_love = np.array([0.0])
            
        if par.sigma_love<=1.0e-6:par.grid_shock_love,par.grid_weight_love = np.array([0.0]),np.array([1.0]);par.num_shock_love = 1
        else:par.grid_shock_love,par.grid_weight_love = quadrature.normal_gauss_hermite(par.sigma_love,par.num_shock_love)
        
        # grid for women's labor supply
        par.grid_wlp=np.array([0.0,1.0]);par.num_wlp=len(par.grid_wlp)
         
        # pre-computation of total consumption
        par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)

        # marginal utility grids for EGM 
        par.grid_inv_marg_u = np.flip(par.grid_Ctot)
        par.grid_marg_u = np.nan + np.ones((par.num_wlp,par.num_power,par.num_Ctot))# couples
        par.grid_marg_uw = np.nan + np.ones((par.num_wlp,par.num_power,par.num_Ctot))# couples
        par.grid_marg_um = np.nan + np.ones((par.num_wlp,par.num_power,par.num_Ctot))# couples
        par.grid_marg_u_for_inv = np.nan + np.ones(par.grid_marg_u.shape)# couples
        par.grid_cpriv_s =  np.nan + np.ones(par.num_Ctot)# singles
        par.grid_marg_u_s = np.nan + np.ones(par.num_Ctot)# singles
   
        # income shocks grids: singles and couples
        par.grid_zw,par.Π_zw= usr.labor_income(par.t0w,par.t1w,par.t2w,par.T,par.Tr,par.σzw,par.σ0zw,par.num_zw)
        par.grid_zm,par.Π_zm= usr.labor_income(par.t0m,par.t1m,par.t2m,par.T,par.Tr,par.σzm,par.σ0zm,par.num_zm)
        par.Π=[np.kron(par.Π_zw[t],par.Π_zm[t]) for t in range(par.T-1)] # couples trans matrix    
        par.matw0=usr.rouw_nonst(2,par.σ0zw,0.0,par.num_zw)[1][0] # w's initial income
        par.matm0=usr.rouw_nonst(2,par.σ0zw,0.0,par.num_zw)[1][0] # m's initial income
        
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
                
                # solve the couple's problem (EGM vs. vfi done later)
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
    C_pub,  Cw_priv, Cm_priv, grid_marg_u, grid_marg_u_for_inv, grid_marg_u_s, grid_cpriv_s, grid_marg_uw, grid_marg_um =\
        sol.pre_Ctot_C_pub, sol.pre_Ctot_Cw_priv, sol.pre_Ctot_Cm_priv, par.grid_marg_u, par.grid_marg_u_for_inv, par.grid_marg_u_s,\
        par.grid_cpriv_s, par.grid_marg_uw, par.grid_marg_um
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)  
    ϵ = 1e-8# delta increase in xs to compute numerical deratives

    ################ Singles part #####################
    minus_util_s = lambda x,resources,pars:-usr.util(x,resources-x,*pars)
    for i,C_tot in enumerate(par.grid_Ctot):
        # get util from *total consumption=priv+public*  and the numerical derivative
        grid_cpriv_s[i] = usr.optimizer(minus_util_s,1.0e-8, C_tot-1.0e-8,args=(C_tot,pars))[0]
        
        # compute numerical derivative of vf of singles over total consumption
        share_p=grid_cpriv_s[i]/C_tot
        forward  =    usr.util(share_p*(C_tot+ϵ),(1.0-share_p)*(C_tot+ϵ),*pars)#-usr.optimizer(minus_util_s,1.0e-8, C_tot+ϵ-1.0e-8,args=(C_tot+ϵ,pars))[1]
        backward =    usr.util(share_p*(C_tot-ϵ),(1.0-share_p)*(C_tot-ϵ),*pars)#-usr.optimizer(minus_util_s,1.0e-8, C_tot-ϵ-1.0e-8,args=(C_tot-ϵ,pars))[1]
        grid_marg_u_s[i] = (forward - backward)/(2*ϵ)#usr.marg_util(C_tot,*pars)#

        
    ################ Couples part ##########################  
    couples_util=lambda cp, prs: usr.couple_util(cp,*prs)[0]#fun to maximize
    ini_cond=np.array([0.33,0.33])#initial condition, to be overwritten
    for iP in prange(par.num_power):     
        for iwlp,wlp in enumerate(par.grid_wlp):
            for i,C_tot in enumerate(par.grid_Ctot):
                
                # initialize bounds and bargaining power
                bounds=np.array([[0.0,C_tot],[0.0,C_tot]]);power=par.grid_power[iP]
                
                # estimate
                res = nelder_mead(couples_util,ini_cond*C_tot,bounds=bounds,args=((C_tot,power,1.0-wlp,*pars),))

                # unpack
                Cw_priv[iwlp,iP,i]= res.x[0];Cm_priv[iwlp,iP,i]= res.x[1]  
                C_pub[iwlp,iP,i] = C_tot - Cw_priv[iwlp,iP,i] - Cm_priv[iwlp,iP,i]
                
                # update initial conidtion
                ini_cond=res.x/C_tot if i<par.num_Ctot-1 else np.array([0.33,0.33])
      
    for iP in prange(par.num_power):     
        for iwlp,wlp in enumerate(par.grid_wlp):
            for i,C_tot in enumerate(par.grid_Ctot):
                power=par.grid_power[iP]
                res.x[0]=Cw_priv[iwlp,iP,i];res.x[1]=Cm_priv[iwlp,iP,i] 
                # get individual derivative using envelope theorem: share of private/public C doesnt change
                _,forward_w,  forward_m  =usr.couple_util(res.x/(C_tot)*(C_tot+ϵ),C_tot+ϵ,power,1.0-wlp,*pars)# usr.couple_time_utility(C_tot+ϵ,par,sol,iP,iwlp,power,0.0,pars)#
                _,backward_w, backward_m = usr.couple_util(res.x/(C_tot)*(C_tot-ϵ),C_tot-ϵ,power,1.0-wlp,*pars)#usr.couple_time_utility(C_tot-ϵ,par,sol,iP,iwlp,power,0.0,pars)#
                grid_marg_uw[iwlp,iP,i] = (forward_w - backward_w)/(2*ϵ) 
                grid_marg_um[iwlp,iP,i] = (forward_m - backward_m)/(2*ϵ) 
                
            #Create grid of couple's marginal util and inverse marginal utility 
            grid_marg_u[iwlp,iP,:] = power*grid_marg_uw[iwlp,iP,:]+(1.0-power)*grid_marg_um[iwlp,iP,:]
            grid_marg_u_for_inv[iwlp,iP,:]=np.flip(par.grid_marg_u[iwlp,iP,:])    
        

#######################
# SOLUTIONS - SINGLES #
#######################

@njit(parallel=parallel)
def integrate_single(sol,par,t):
    Ew,Ewd=np.ones((2,par.num_zw,par.num_A));Em,Emd=np.ones((2,par.num_zm,par.num_A)) 
    
    for iA in prange(par.num_A):Ew[:,iA]  = sol.Vw_single[t+1,:,iA].flatten() @ par.Π_zw[t] 
    for iA in prange(par.num_A):Em[:,iA]  = sol.Vm_single[t+1,:,iA].flatten() @ par.Π_zm[t]

    return Ew,Em 
    
@njit(parallel=parallel)
def solve_single_egm(sol,par,t):#TODO add upper-envelope if remarriage...

    #Integrate to get continuation value...
    if t<par.T-1:Ew,Em = integrate_single(sol,par,t)
    else:#...unless if you are in the last period
        Ew= np.zeros((par.num_zw,par.num_A));Em= np.zeros((par.num_zm,par.num_A))
             
    #Pre-define outcomes (if update .sol directly, parallelization go crazy)
    vw=sol.Vw_single[t,:,:];vm=sol.Vm_single[t,:,:]
    cw=sol.Cw_tot_single[t,:,:];cm=sol.Cm_tot_single[t,:,:];cwt,Ewt,cwp=np.ones((3,par.num_zw,par.num_A));cmt,Emt,cmp=np.ones((3,par.num_zm,par.num_A));
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)
    
    #def loop_savings_singles(num_zi,grid_zi,grid_Ai,ci,Ei,,cit)
    # Women
    for iz in prange(par.num_zw):

        resw = (par.R*par.grid_Aw + par.grid_zw[t,iz])#*0.0000001
        if t==(par.T-1): cw[iz,:] = resw.copy() #consume all resources
        else: #before T-1 make consumption saving choices
            
            # marginal utility next period
            βEwd=par.β*usr.deriv(par.grid_Aw,Ew[iz,:])
            
            # first get toatl -consumption out of grid using FOCs
            linear_interp.interp_1d_vec(np.flip(par.grid_marg_u_s),par.grid_inv_marg_u,βEwd,cwt[iz,:])
            
            # use budget constraint to get current resources
            Aw_now = (par.grid_Aw.flatten() + cwt[iz,:] - par.grid_zw[t,iz])/par.R
            
            # now interpolate onto common beginning-of-period asset grid to get consumption
            linear_interp.interp_1d_vec(Aw_now,cwt[iz,:],par.grid_Aw,cw[iz,:])
            
            # get consumption (+make sure that the no-borrowing constraint is respected)
            cw[iz,:] = np.minimum(cw[iz,:], resw.copy())       
            
        # get utility: interpolate Exp value (line 1), current util (line 2) and add continuation (line 3)#TODO improve precision
        linear_interp.interp_1d_vec(par.grid_Aw,Ew[iz,:], resw.copy()-cw[iz,:],Ewt[iz,:])
        linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_cpriv_s,cw[iz,:],cwp[iz,:])
        vw[iz,:]=usr.util(cwp[iz,:],cw[iz,:]-cwp[iz,:],*pars)+par.β*Ewt[iz,:]
        
       
        
    # Men
    for iz in prange(par.num_zm):

        resm = (par.R*par.grid_Am + par.grid_zm[t,iz])#*0.0000001
        if t==(par.T-1): cm[iz,:] = resm.copy() #consume all resources
        else: #before T-1 make consumption saving choices
            
            # marginal utility next period
            βEmd=par.β*usr.deriv(par.grid_Am,Em[iz,:])
            
            # first get total consumption out of grid using FOCs
            linear_interp.interp_1d_vec(np.flip(par.grid_marg_u_s),par.grid_inv_marg_u,βEmd,cmt[iz,:])
            
            # use budget constraint to get current resources
            Am_now = (par.grid_Am.flatten() + cmt[iz,:] - par.grid_zm[t,iz])/par.R
            
            # now interpolate onto common beginning-of-period asset grid to get consumption
            linear_interp.interp_1d_vec(Am_now,cwt[iz,:],par.grid_Am,cm[iz,:])
            
            # get consumption (+make sure that the no-borrowing constraint is respected)
            cm[iz,:] = np.minimum(cm[iz,:], resm.copy())
            
        # get utility: interpolate Exp value (line 1), current util (line 2) and add continuation (line 3)#TODO improve precision
        linear_interp.interp_1d_vec(par.grid_Am,Em[iz,:],resm.copy() - cm[iz,:],Emt[iz,:])
        linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_cpriv_s,cm[iz,:],cmp[iz,:])
        vm[iz,:]=usr.util(cmp[iz,:],cm[iz,:]-cmp[iz,:],*pars)+par.β*Emt[iz,:]
            
        
#################################################
# SOLUTION - COUPLES
################################################

def solve_couple(sol,par,t):#Solve the couples's problem, choose EGM of VFI techniques
 
    # solve the couple's problem: choose your fighter
    if par.EGM: tuple_with_outcomes =     solve_remain_couple_egm(par,sol,t)# EGM solution        
    else:       tuple_with_outcomes = vfi.solve_remain_couple(par,sol,t)    # vfi solution
              
    #Check participation constrints and eventually update policy function and pareto weights
    check_participation_constraints(*tuple_with_outcomes,par,sol,t)
    
    
@njit(parallel=parallel)
def integrate_couple(par,sol,t): 
     
    EVw,EVm,pEVw,pEVm,pEVdw,pEVdm,EVdw,EVdm=np.ones((8,par.num_z,par.num_love,par.num_power,par.num_A)) 
    aw,am,adw,adm = np.ones((4,par.num_z,par.num_love,par.num_power,par.num_A,par.num_shock_love)) 
    love_next_vec=np.ones((par.num_love,par.num_shock_love)) 
     
    for iL in range(par.num_love):love_next_vec[iL,:] = par.grid_love[iL] + par.grid_shock_love 
     
    #Integrate income shocks
    for iL in prange(par.num_love): 
        for iP in range(par.num_power):     
            for iA in range(par.num_A): 
                id_i=(slice(None),iL,iP,iA);id_ii=(t+1,slice(None),iP,iL,iA)
                pEVw[id_i] =  sol.Vw_couple[id_ii].flatten() @ par.Π[t] 
                pEVm[id_i] =  sol.Vm_couple[id_ii].flatten() @ par.Π[t] 

    #Integrate love shocks
    for iL in prange(par.num_love): 
        for iP in range(par.num_power):     
            for iA in range(par.num_A): 
                for iz in range(par.num_z): 
                    id_i=(iz,slice(None),iP,iA);id_ii=(iz,iL,iP,iA)
                    linear_interp.interp_1d_vec(par.grid_love,pEVw[id_i],love_next_vec[iL,:],aw[id_ii]) 
                    linear_interp.interp_1d_vec(par.grid_love,pEVm[id_i],love_next_vec[iL,:],am[id_ii])

                    EVw[id_ii] =  aw[id_ii].flatten() @ par.grid_weight_love 
                    EVm[id_ii] =  am[id_ii].flatten() @ par.grid_weight_love 
           
    return EVw,EVm


@njit(parallel=parallel) 
def solve_remain_couple_egm(par,sol,t): 
 
    #Integration if not last period
    if t<(par.T-1): EVw, EVm = integrate_couple(par,sol,t)
 
    # initialize 
    Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_C_tot,n_C_tot,wlp,Vwd,Vmd,wgt_w,wgt_m,wgt_w_p,wgt_m_p,wgt_w_n,wgt_m_n\
        =np.zeros((17,par.num_z,par.num_love,par.num_A,par.num_power)) 
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)    
    cw,cm = sol.Cw_tot_single[t],sol.Cm_tot_single[t] 
    for iL in prange(par.num_love): 
        for iP in range(par.num_power):            
            for iz in range(par.num_z):  
                
                # indexes
                izw=iz//par.num_zm;izm=iz%par.num_zw;idz=(iz,iL,iP)
                love=par.grid_love[iL];power = par.grid_power[iP]
                idx=(iz,iL,slice(None),iP)#; I=lambda a: (iz,iL,a,iP)
                
                # resources if participation (p_) or no participation (_n)
                p_res=usr.resources_couple(t,par.grid_A,izw,izm,par,wlp=1) 
                n_res=usr.resources_couple(t,par.grid_A,izw,izm,par,wlp=0) 
                
                # continuation values 
                if t==(par.T-1):#last period 
                    
                    #Get consumption then utilities (assume no labor participation). Note: no savings!
                    Vw[idx],Vm[idx]=usr.couple_time_utility(n_res,par,sol,iP,0,love,pars)            
                    wlp[idx]=0.0;p_Vm[idx]=p_Vw[idx]=-1e10;n_Vw[idx],n_Vm[idx]=Vw[idx],Vm[idx];n_C_tot[idx] = n_res.copy() 
                    
                    
                else:#periods before the last 
                             
                    # compute optimal consumption and utility given partecipation (0/1).
                    # the last 3 arguments of fun below are the output at iz,iL,iP
                    compute_couple(par,sol,iP,idx,idz,love,power,pars,EVw,EVm,1,p_res,p_C_tot,p_Vw,p_Vm) # participation 
                    compute_couple(par,sol,iP,idx,idz,love,power,pars,EVw,EVm,0,n_res,n_C_tot,n_Vw,n_Vm) # no participation 
                    if (t>=par.Tr):p_Vw[idx]=p_Vm[idx]=-1e10 # before retirement no labor participation 
                    
                    # compute participation choices below
                    p_Vwd, n_Vwd,p_Vmd, n_Vmd = np.ones((4,par.num_A))
                    p_v_couple = power*p_Vw[idx]+(1.0-power)*p_Vm[idx]
                    n_v_couple = power*n_Vw[idx]+(1.0-power)*n_Vm[idx]
                                                   
                    # get the probabilit of workin wlp, based on couple utility choices
                    c=np.maximum(p_v_couple,n_v_couple)/par.σ # constant to avoid overflow
                    v_couple=par.σ*(c+np.log(np.exp(p_v_couple/par.σ-c)+np.exp(n_v_couple/par.σ-c)))
                    wlp[iz,iL,:,iP]=np.exp(p_v_couple/par.σ-v_couple/par.σ)
                    
                    # now the value of making the choice: see Shepard (2019), page 11
                    Δp=p_Vw[idx]-p_Vm[idx];Δn=n_Vw[idx]-n_Vm[idx]
                    Vw[idx]=v_couple+(1.0-par.grid_power[iP])*(wlp[idx]*Δp+(1.0-wlp[idx])*Δn)
                    Vm[idx]=v_couple+(-par.grid_power[iP])   *(wlp[idx]*Δp+(1.0-wlp[idx])*Δn)
 
                    
    return (Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_C_tot,n_C_tot,wlp,Vwd,Vmd,wgt_w,wgt_m) # return a tuple
 
@njit    
def compute_couple(par,sol,iP,idx,idz,love,power,pars2,EVw,EVm,part,res,C_tot,Vw,Vm): 
 
    # initialization 
    C_pd,βEw,βEm,Vwd,Vmd,_= np.ones((6,par.num_A));pars=(par,sol,iP,part,love,pars2)        
         
    
    βEVd=par.β*usr.deriv(par.grid_A,power*EVw[idz]+(1.0-power)*EVm[idz])

    # get consumption out of grid using FOCs (i) + use budget constraint to get current resources (ii)  
    linear_interp.interp_1d_vec(par.grid_marg_u_for_inv[part,iP,:],par.grid_inv_marg_u,βEVd,C_pd) #(i) 
    A_now =  par.grid_A.flatten() + C_pd                                                         #(ii) 
 

    if 1>0:#np.any(np.diff(A_now)<0):#apply upperenvelope + enforce no borrowing constraint 
 
        upper_envelope(par.grid_A,A_now,C_pd,par.β*EVw[idz],par.β*EVm[idz],power,res, 
                       C_tot[idx],Vw[idx],Vm[idx],*pars) 
        
    else:#upperenvelope not necessary: enforce no borrowing constraint 
     
        # interpolate onto common beginning-of-period asset grid to get consumption 
        linear_interp.interp_1d_vec(A_now,C_pd,res,C_tot[idx]) 
        C_tot[idx] = np.minimum(C_tot[idx] , res) #...+ apply borrowing constraint 
     
        # compute the value function 
        Cw_priv, Cm_priv, C_pub =\
            usr.intraperiod_allocation(C_tot[idx],par.grid_Ctot,sol.pre_Ctot_Cw_priv[part,iP],sol.pre_Ctot_Cm_priv[part,iP])  
             
        linear_interp.interp_1d_vec(par.grid_A,par.β*EVw[idz],res-C_tot[idx],βEw) 
        linear_interp.interp_1d_vec(par.grid_A,par.β*EVm[idz],res-C_tot[idx],βEm)     
        Vw[idx] = usr.util(Cw_priv,C_pub,*pars2,love,True,1.0-par.grid_wlp[part])+βEw                         
        Vm[idx] = usr.util(Cm_priv,C_pub,*pars2,love,True,1.0-par.grid_wlp[part])+βEm 
        
       
@njit(parallel=parallel)
def check_participation_constraints(remain_Vw,remain_Vm,p_remain_Vw,p_remain_Vm,n_remain_Vw,n_remain_Vm,p_C_tot,n_C_tot, remain_wlp,remain_Vwd,remain_Vmd,wgt_w,wgt_m,par,sol,t):
 
    power_idx=sol.power_idx
    power=sol.power
            
    for iL in prange(par.num_love):
        for iA in range(par.num_A):
            for iz in range(par.num_z):
               
                # check the participation constraints
                idx_single_w = (t,iz//par.num_zm,iA);idx_single_m = (t,iz%par.num_zw,iA);idd=(iz,iL,iA)
     
                
                list_couple = (sol.Vw_couple, sol.Vm_couple)#list_start_as_couple
                list_raw    = (remain_Vw[idd],remain_Vm[idd])#list_remain_couple
                list_single = (sol.Vw_single,sol.Vm_single) # list_trans_to_single: last input here not important in case of divorce
                list_single_w = (True,False) # list that say whether list_single[i] is about w (as oppoesd to m)
                
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
                            if list_single_w[i]: list_couple[i][idx] = list_single[i][idx_single_w]
                            else:                list_couple[i][idx] = list_single[i][idx_single_m]
            
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
                                    if list_single_w[i]: list_couple[i][idx] = list_single[i][idx_single_w]
                                    else:                list_couple[i][idx] = list_single[i][idx_single_m]
            
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
                                    if list_single_w[i]: list_couple[i][idx] = list_single[i][idx_single_w]  
                                    else:                list_couple[i][idx] = list_single[i][idx_single_m]
            
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
                part_i=np.exp(V_p/par.σ-v_couple/par.σ)
                wlp[i,t]=(part_i>sim.shock_taste[i,t])
                               
                # optimal consumption allocation if couple (note use of the updated index)
                sol_C_tot = sol.p_C_tot_remain_couple[t,iz[i,t],power_idx[i,t]] if wlp[i,t] else sol.n_C_tot_remain_couple[t,iz[i,t],power_idx[i,t]] 
                C_tot = linear_interp.interp_2d(par.grid_love,par.grid_A,sol_C_tot,love[i,t],A_lag)
                args=(np.array([C_tot]),par.grid_Ctot,sol.pre_Ctot_Cw_priv[wlp[i,t],power_idx[i,t]],sol.pre_Ctot_Cm_priv[wlp[i,t],power_idx[i,t]])
                Cw_priv[i:i+1,t], Cm_priv[i:i+1,t], C_pub = usr.intraperiod_allocation(*args)
                Cw_pub[i,t] = Cm_pub[i,t] = C_pub[0]
            
                # update end-of-period states
                M_resources = usr.resources_couple(t,A_lag,iz_w,iz_m,par,wlp=wlp[i,t]) 
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