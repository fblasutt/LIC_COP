import numpy as np#import autograd.numpy as np#
from EconModel import EconModelClass, jit
from consav.grids import nonlinspace
from consav import linear_interp, linear_interp_1d
from numba import njit,prange,config
import UserFunctions_numba as usr
from quantecon.optimize.nelder_mead import nelder_mead
import setup
upper_envelope=usr.create(usr.couple_time_utility)

#general configuratiion and glabal variables (common across files)
config.DISABLE_JIT = setup.nojit;parallel=setup.parallel;cache=setup.cache
woman=setup.woman;man=setup.man;mar=setup.mar;coh=setup.coh

class HouseholdModelClass(EconModelClass):
    
    def settings(self):self.namespaces = []#needed to make class work, otherwise useless
                   
    def setup(self):
        par = self.par
        
        par.R = 1.03
        par.β = 1/1.03# Discount factor
        par.div_A_share = 0.5 # divorce share of wealth to wife
        par.sep_cost = [0.1, 0.0] # share of wealth lost upon divorce (0), breakup (1) 
        par.γ        = [0.0, 0.0] # utility stigma towards marriage (0) and cohabitation (1)

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
        par.num_A = 20;par.max_A = 30.0
        
        # bargaining power
        par.num_power = 15
        
        #women's human capital states
        par.num_h = 2
        par.drift = 0.4 #human capital depreciation drift
        par.pr_h_change = 0.1 # probability that  human capital depreciates

        # love/match quality
        par.num_lovew = 3;par.num_lovem = 3;par.num_love=par.num_lovem*par.num_lovew
        par.sigma_love = 0.1
        
        # income of men and women: gridpoints
        par.num_zw=3;par.num_zm=3; par.num_z=par.num_zm*par.num_zw
        
        # income of men and women: parameters of the age log-polynomial
        par.t0w=-0.5;par.t1w=0.03;par.t2w=0.0;par.t0m=-0.5;par.t1m=0.03;par.t2m=0.0;
    
        # income of men and women: sd of income shocks in t=0 and after that
        par.σzw=0.4;par.σ0zw=0.3;par.σzm=0.4;par.σ0zm=0.3
        
        # money you get at retirement
        par.pension = 0.2
        
        # pre-computation fo consumption
        par.num_Ctot = 150;par.max_Ctot = par.max_A
        
        # simulation
        par.seed = 9210;par.simT = par.T;par.simN = 20_000
        
    def allocate(self):
        par = self.par;sol = self.sol;sim = self.sim;self.setup_grids()

        # setup grids
        par.simT = par.T
         
        # singles: value functions (vf), consumption, marg util
        shape_singlew = (par.T,par.num_h,par.num_zw,par.num_A)
        shape_singlem = (par.T,1        ,par.num_zm,par.num_A)
        sol.Vw_single = np.nan + np.ones(shape_singlew) #vf in t
        sol.Vm_single = np.nan + np.ones(shape_singlem) #vf in t
        sol.Cw_tot_single = np.nan + np.ones(shape_singlew) #priv+tot cons
        sol.Cm_tot_single = np.nan + np.ones(shape_singlem) #priv+tot cons

        # couples: value functions (vf), consumption, marg util, bargaining power
        shape_couple = (par.T,2,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A) # 2 is for mar/coh
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
        sol.power =  np.nan +np.zeros(shape_couple)                  #barg power value

        # pre-compute optimal consumption allocation: private and public
        shape_pre = (par.num_wlp,par.num_power,par.num_Ctot)
        sol.pre_Ctot_Cw_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_Cm_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_C_pub = np.nan + np.ones(shape_pre)
        
        # simulation
        shape_sim = (par.simN,par.simT)
        sim.C_tot = np.nan + np.ones(shape_sim)                # total consumption
        sim.iz = np.ones(shape_sim,dtype=np.int_)              # index of income shocks 
        sim.A = np.nan + np.ones(shape_sim)                    # total assets (m+w)
        sim.Aw = np.nan + np.ones(shape_sim)                   # w's assets
        sim.Am = np.nan + np.ones(shape_sim)                   # m's assets
        sim.rel = np.ones(shape_sim,dtype=np.int_)    # mar (0), coh (1), single (2)
        sim.rel_lag = np.ones(shape_sim,dtype=np.int_)# mar (0), coh (1), single (2)
        sim.power = np.nan + np.ones(shape_sim)                # barg power
        sim.power_lag = np.ones(shape_sim)                     # barg power index
        sim.love = np.ones(shape_sim,dtype=np.int_)            # love
        sim.incw = np.nan + np.ones(shape_sim)                 # w's income
        sim.incm = np.nan + np.ones(shape_sim)                 # m's income
        sim.WLP = np.ones(shape_sim,dtype=np.int_)             # w's labor participation
        sim.ih = np.ones(shape_sim,dtype=np.int_)              # w's human capital

        # shocks
        np.random.seed(par.seed)
        sim.shock_love = np.random.random_sample((par.simN,par.simT))#love
        sim.shock_z=np.random.random_sample((par.simN,par.simT))     #income
        sim.shock_taste=np.random.random_sample((par.simN,par.simT)) #taste shock
        sim.shock_h=np.random.random_sample((par.simN,par.simT))     #human capital

        # initial distribution
        sim.init_ih = np.zeros(par.simN,dtype=np.int_)                    #initial w's human capital
        sim.A[:,0] = par.grid_A[0] + np.zeros(par.simN)                   #total assetes
        sim.Aw[:,0] = par.div_A_share * sim.A[:,0]                        #w's assets
        sim.Am[:,0] = (1.0 - par.div_A_share) * sim.A[:,0]                #m's assets
        sim.init_rel =    np.ones(par.simN,dtype=np.int_)                  #mar (0), coh (1), single (2)
        sim.init_power =  0.11373989**np.ones(par.simN)                   #barg power 
        sim.init_lovew = np.ones(par.simN,dtype=np.int_)**par.num_lovew//2#w's initial love
        sim.init_lovem = np.ones(par.simN,dtype=np.int_)**par.num_lovem//2#m's initial love
        sim.init_love = sim.init_lovew*par.num_zm+sim.init_lovem          #initial love
        sim.init_zw = np.ones(par.simN,dtype=np.int_)*par.num_zw//2       #w's initial income 
        sim.init_zm = np.ones(par.simN,dtype=np.int_)*par.num_zm//2       #m's initial income
        sim.init_z  = sim.init_zw*par.num_zm+sim.init_zm                  #    initial income
        
    def setup_grids(self):
        par = self.par
        
        # wealth. Single grids are such to avoid interpolation
        par.grid_A  = nonlinspace(0.0,par.max_A,par.num_A,1.1)#np.linspace(0.0,par.max_A,par.num_A)
        par.grid_Aw =  par.grid_A; par.grid_Am =  par.grid_A

        # women's human capital grid plus transition if w working (p) or not (n)
        par.grid_h = np.flip(np.linspace(-par.num_h*par.drift,0.0,par.num_h))
        par.Πh_p  = np.array([[1.0,0.0],[1.0-par.pr_h_change,par.pr_h_change]]).T #assumes 2 states for h
        par.Πh_n  = np.array([[1.0-par.pr_h_change,par.pr_h_change],[0.0,1.0]]).T #assumes 2 states for h
        
        # bargaining power. non-linear grid with more mass in both tails.        
        par.grid_power = usr.grid_fat_tails(0.01,0.99,par.num_power)

        # love grid and shock    
        par.grid_lovew,par.Πlw= usr.rouw_nonst(par.T,par.sigma_love,par.sigma_love,par.num_lovew)
        par.grid_lovem,par.Πlm= usr.rouw_nonst(par.T,par.sigma_love,par.sigma_love,par.num_lovem)
        par.Πl=[np.kron(par.Πlw[t],par.Πlm[t]) for t in range(par.T-1)] # couples trans matrix
        
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
        par.grid_zw,par.Π_zw= usr.labor_income(par.t0w,par.t1w,par.t2w,par.T,par.Tr,par.σzw,par.σ0zw,par.num_zw,par.pension)
        par.grid_zm,par.Π_zm= usr.labor_income(par.t0m,par.t1m,par.t2m,par.T,par.Tr,par.σzm,par.σ0zm,par.num_zm,par.pension)
        par.Π=[np.kron(par.Π_zw[t],par.Π_zm[t]) for t in range(par.T-1)] # couples trans matrix    
                        
    def solve(self):

        with jit(self) as model:#This allows passing sol and par to jiit functions  
            
            #Import parameters and arrays for solution
            par = model.par; sol = model.sol
            
            # precompute the optimal intra-temporal consumption allocation given total consumpotion
            solve_intraperiod(sol,par)
            
            # loop backwards and obtain policy functions
            for t in reversed(range(par.T)):
                
                # solve the single's problem
                solve_single_egm(sol,par,t)
                
                # solve the couple's problem 
                solve_couple(sol,par,t)
                         
    def simulate(self):
        
        with jit(self) as model:    
            
            #Import parameter, policy functions and simulations arrays
            par = model.par; sol = model.sol; sim = model.sim
            
            #Call routing performing the simulation
            simulate_lifecycle(sim,sol,par)
             
####################################################
# INTRAPERIOD OPTIMIZATION FOR SINGLES AND COUPLES #
####################################################
@njit(parallel=parallel)
def solve_intraperiod(sol,par):
        
    # unpack to help numba (horrible)
    C_pub,  Cw_priv, Cm_priv, grid_marg_u, grid_marg_u_for_inv, grid_marg_u_s, grid_cpriv_s, grid_marg_uw, grid_marg_um =\
        sol.pre_Ctot_C_pub, sol.pre_Ctot_Cw_priv, sol.pre_Ctot_Cm_priv, par.grid_marg_u, par.grid_marg_u_for_inv, par.grid_marg_u_s,\
        par.grid_cpriv_s, par.grid_marg_uw, par.grid_marg_um
        
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)  
    ϵ = 1e-8# delta increase in xs to compute numerical deratives

    ################ Singles part #####################
    for i,C_tot in enumerate(par.grid_Ctot):
        
        # optimize to get util from total consumption(m<->C_tot)=private cons(c)+public cons(m-c)
        grid_cpriv_s[i] = usr.optimizer(lambda c,m,p:-usr.util(c,m-c,*p),ϵ,C_tot-ϵ,args=(C_tot,pars))[0]
        
        # numerical derivative of util wrt total consumption C_tot, using envelope thm
        share_priv=grid_cpriv_s[i]/C_tot
        forward  = usr.util(share_priv*(C_tot+ϵ),(1.0-share_priv)*(C_tot+ϵ),*pars)
        backward = usr.util(share_priv*(C_tot-ϵ),(1.0-share_priv)*(C_tot-ϵ),*pars)
        grid_marg_u_s[i] = (forward - backward)/(2*ϵ)
  
    ################ Couples part ##########################   
    icon=np.array([0.33,0.33])#initial condition, to be overwritten 
    for iP in prange(par.num_power):      
        for iwlp,wlp in enumerate(par.grid_wlp): 
            for i,C_tot in enumerate(par.grid_Ctot): 
                 
                # initialize bounds and bargaining power 
                bounds=np.array([[0.0,C_tot],[0.0,C_tot]]);power=par.grid_power[iP] 
                 
                # estimate optima private and public cons, unpack, update initial condition
                res = nelder_mead(lambda c,p:usr.couple_util(c,*p)[0],icon*C_tot,bounds=bounds,args=((C_tot,power,1.0-wlp,*pars),)) 
                Cw_priv[iwlp,iP,i]= res.x[0];Cm_priv[iwlp,iP,i]= res.x[1];C_pub[iwlp,iP,i] = C_tot - res.x.sum()             
                icon=res.x/C_tot if i<par.num_Ctot-1 else np.array([0.33,0.33]) 
       
                # numerical derivative of util wrt total consumption C_tot, using envelope thm 
                _,forw_w,forw_m = usr.couple_util(res.x/(C_tot)*(C_tot+ϵ),C_tot+ϵ,power,1.0-wlp,*pars) 
                _,bakw_w,bakw_m = usr.couple_util(res.x/(C_tot)*(C_tot-ϵ),C_tot-ϵ,power,1.0-wlp,*pars) 
                grid_marg_uw[iwlp,iP,i] = (forw_w - bakw_w)/(2*ϵ);grid_marg_um[iwlp,iP,i] = (forw_m - bakw_m)/(2*ϵ)
                                  
            #Create grid of couple's marginal util and inverse marginal utility  
            grid_marg_u[iwlp,iP,:] = power*grid_marg_uw[iwlp,iP,:]+(1.0-power)*grid_marg_um[iwlp,iP,:] 
            grid_marg_u_for_inv[iwlp,iP,:]=np.flip(par.grid_marg_u[iwlp,iP,:])   
        
#######################
# SOLUTIONS - SINGLES #
#######################

@njit(parallel=parallel)
def integrate_single(sol,par,t):
    Ew=np.ones((par.num_h,par.num_zw,par.num_A));Em=np.ones((1,par.num_zm,par.num_A)) 
     
    # men
    for iA in prange(par.num_A):Em[:,:,iA]  = sol.Vm_single[t+1,:,:,iA].flatten() @ par.Π_zm[t]
    
    # women: always work, is low human (1) capital have a change to improve their position
    for iA in prange(par.num_A):
        for iz in range(par.num_zw):
            for ih in range(par.num_h):
                Ew[ih,iz,iA]  = sol.Vw_single[t+1,:,iz,iA] @ par.Πh_p[:,ih]

    return Ew,Em 
    
@njit(parallel=parallel)
def solve_single_egm(sol,par,t):#TODO add upper-envelope if remarriage...

    #Integrate to get continuation value unless if you are in the last period
    if t<par.T-1:Ew,Em = integrate_single(sol,par,t)
    else:Ew= np.zeros((par.num_h,par.num_zw,par.num_A));Em= np.zeros((1,par.num_zm,par.num_A))
             
    #Pre-define outcomes (if update .sol directly, parallelization go crazy)
    vw=sol.Vw_single[t,:,:,:];vm=sol.Vm_single[t,:,:,:];cw=sol.Cw_tot_single[t,:,:,:];cm=sol.Cm_tot_single[t,:,:,:]
    cwt,Ewt,cwp=np.ones((3,par.num_h,par.num_zw,par.num_A));cmt,Emt,cmp=np.ones((3,1,par.num_zm,par.num_A));
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)
    
    #function to find optimal savings, called for both men and women below
    def loop_savings_singles(par,num_hi,num_zi,grid_zi,grid_Ai,ci,Ei,cit,Eit,cip,vi):
        
        for iz in prange(num_zi):
            for ih in range(num_hi):

                resi = (par.R*grid_Ai + np.exp(np.log(grid_zi[t,iz]) + par.grid_h[ih]))
                if t==(par.T-1): ci[ih,iz,:] = resi.copy() #consume all resources
                else: #before T-1 make consumption saving choices
                    
                    # marginal utility of assets next period
                    βEid=par.β*usr.deriv(grid_Ai,Ei[ih,iz,:])
                    
                    # first get toatl -consumption out of grid using FOCs
                    linear_interp.interp_1d_vec(np.flip(par.grid_marg_u_s),par.grid_inv_marg_u,βEid,cit[ih,iz,:])
                    
                    # use budget constraint to get current resources
                    Ai_now = (grid_Ai.flatten() + cit[ih,iz,:] - grid_zi[t,iz])/par.R
                    
                    # now interpolate onto common beginning-of-period asset grid to get consumption
                    linear_interp.interp_1d_vec(Ai_now,cit[ih,iz,:],grid_Ai,ci[ih,iz,:])
                    
                    # get consumption (+make sure that the no-borrowing constraint is respected)
                    ci[ih,iz,:] = np.minimum(ci[ih,iz,:], resi.copy())       
                    
                # get utility: interpolate Exp value (line 1), current util (line 2) and add continuation (line 3)#TODO improve precision
                linear_interp.interp_1d_vec(grid_Ai,Ei[ih,iz,:], resi.copy()-ci[ih,iz,:],Eit[ih,iz,:])
                linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_cpriv_s,ci[ih,iz,:],cip[ih,iz,:])
                vi[ih,iz,:]=usr.util(cip[ih,iz,:],ci[ih,iz,:]-cip[ih,iz,:],*pars)+par.β*Eit[ih,iz,:]
            
    loop_savings_singles(par,par.num_h,par.num_zw,par.grid_zw,par.grid_Aw,cw,Ew,cwt,Ewt,cwp,vw)#women savings
    loop_savings_singles(par,1        ,par.num_zm,par.grid_zm,par.grid_Am,cm,Em,cmt,Emt,cmp,vm)#men   savings
                       
#################################################
# SOLUTION - COUPLES
################################################

def solve_couple(sol,par,t):#Solve the couples's problem
 
    # solve the couple's problem: choose your fighter
    tuple_with_outcomes =     solve_remain_couple_egm(par,sol,t)
              
    #Store above outcomes into solution
    store(*tuple_with_outcomes,par,sol,t)
        
@njit(parallel=parallel)
def integrate_couple(par,sol,t): 
     
    pEVw,pEVm,nEVw,nEVm=np.zeros((4,2,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)) 

    #Π=kroneker product of uncertainty in love, income, human c. + drop close to zero entries in Π
    Πp=usr.rescale_matrix(np.kron(np.kron(par.Πh_p,par.Π[t]),par.Πl[t]))
    Πn=usr.rescale_matrix(np.kron(np.kron(par.Πh_n,par.Π[t]),par.Πl[t]))
    to_pass=(Πp==0.0) & (Πn==0.0)#whether kroneker product is 0 and does not contribute to EV
    
    for iL in prange(par.num_love): 
        for rel in (mar,coh):
            for ih in range(par.num_h):  
                for iP in range(par.num_power):     
                    for iA in range(par.num_A): 
                        for iz in range(par.num_z): 
                            for jL in range(par.num_love): 
                                for jh in range(par.num_h): 
                                    for jz in range(par.num_z):
                                        
                                        zdx = ih*par.num_love*par.num_z + iz*par.num_love+iL
                                        zjdx =jh*par.num_love*par.num_z + jz*par.num_love+jL
                                        
                                        if to_pass[zjdx,zdx]:continue 
                                        
                                        idx=(rel,ih,iz,iP,iL,iA);jdx=(t+1,rel,jh,jz,iP,jL,iA)
                                            
                                        pEVw[idx]+= sol.Vw_couple[jdx]*Πp[zjdx,zdx]
                                        nEVw[idx]+= sol.Vw_couple[jdx]*Πn[zjdx,zdx]
                                        pEVm[idx]+= sol.Vm_couple[jdx]*Πp[zjdx,zdx]
                                        nEVm[idx]+= sol.Vm_couple[jdx]*Πn[zjdx,zdx]                     
    return pEVw,pEVm,nEVw,nEVm

@njit(parallel=parallel) 
def solve_remain_couple_egm(par,sol,t): 
               
    #Integration if not last period
    if t<(par.T-1): pEVw,pEVm,nEVw,nEVm = integrate_couple(par,sol,t)

    # initialize 
    Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_Vc,n_Vc,p_C_tot,n_C_tot,wlp\
        =np.zeros((11,2,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)) 
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)     
    for iL in prange(par.num_love): 
        for rel in (mar,coh):
            for ih in range(par.num_h):
                for iz in range(par.num_z):
                    for iP in range(par.num_power):
                    
                        # indexes
                        idx=(rel,ih,iz,iP,iL,slice(None))
                                          
                        # resources if no participation (n_) or participation (_p)
                        n_res, p_res=usr.resources_couple(par,t,ih,iz,par.grid_A) 
                        
                        #stigma + love shock
                        γ=par.γ[rel];love = (par.grid_lovew[t][iL//par.num_lovem], par.grid_lovem[t][iL%par.num_lovew])
                        
                        # continuation values 
                        if t==(par.T-1):#last period 
                            
                            #Get consumption then utilities (assume no labor participation). Note: no savings!
                            Vw[idx],Vm[idx]=usr.couple_time_utility(n_res,par,sol,iP,0,love,pars,γ)            
                            wlp[idx]=0.0;p_Vm[idx]=p_Vw[idx]=-1e10;n_Vw[idx],n_Vm[idx]=Vw[idx],Vm[idx];n_C_tot[idx] = n_res.copy() 
                                                
                        else:#periods before the last 
                                     
                            # compute consumption* and util given partecipation (0/1). last 4 arguments below are output at iz,iL,iP
                            compute_couple(par,sol,t,idx,pars,pEVw,pEVm,1,p_res,p_C_tot,p_Vw,p_Vm,p_Vc,love) # participation 
                            compute_couple(par,sol,t,idx,pars,nEVw,nEVm,0,n_res,n_C_tot,n_Vw,n_Vm,n_Vc,love) # no participation 
                            if (t>=par.Tr):p_Vw[idx]=p_Vm[idx]=p_Vc[idx]=-1e10 # before retirement no labor participation 
                                                       
                            # compute the Pr. of of labor part. (wlp) + before-taste-shock util Vw and Vm
                            before_taste_shock(par,p_Vc,n_Vc,p_Vw,n_Vw,p_Vm,n_Vm,idx,wlp,Vw,Vm)
           
                    #Eventual rebargaining happens below
                    for iA in range(par.num_A):        
                                          
                        idxx = [(t,rel,ih,iz,i,iL,iA) for i in range(par.num_power)]               
                        list_couple = (sol.Vw_couple, sol.Vm_couple)                 #couple        list
                        list_raw    = (Vw[rel,ih,iz,:,iL,iA],Vm[rel,ih,iz,:,iL,iA])  #remain-couple list
                        iswomen     = (True,False)                                   #iswomen? in   list
                        
                        Aw=par.div_A_share      *par.grid_A[iA]*(1.0-par.sep_cost[rel])   #wealth if separation, w
                        Am=(1.0-par.div_A_share)*par.grid_A[iA]*(1.0-par.sep_cost[rel])   #wealth if separation, m
                        list_single = (linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,ih,iz//par.num_zm],Aw),#single
                                       linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,0,iz%par.num_zw]  ,Am))#list
                                           
                        check_participation_constraints(par,sol.power,par.grid_power,list_raw,list_single,idxx,list_couple,iswomen)   
                         
    return (Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_C_tot,n_C_tot,wlp) # return a tuple
       
@njit    
def compute_couple(par,sol,t,idx,pars2,EVw,EVm,part,res,C_tot,Vw,Vm,Vc,love): 
 
    # indexes & initialization 
    idz=idx[:-1];iP=idx[3];power = par.grid_power[iP];γ=par.γ[idx[0]]
    C_pd,βEw,βEm,Vwd,Vmd,_= np.ones((6,par.num_A));pars=(par,sol,iP,part,love,pars2,γ)  
                  
    # discounted expected marginal utility from t+1, wrt assets
    βEVd=par.β*usr.deriv(par.grid_A,power*EVw[idz]+(1.0-power)*EVm[idz])

    # get consumption out of grid using FOCs (i) + use budget constraint to get current resources (ii)  
    linear_interp.interp_1d_vec(par.grid_marg_u_for_inv[part,iP,:],par.grid_inv_marg_u,βEVd,C_pd) #(i) 
    A_now =  par.grid_A.flatten() + C_pd                                                         #(ii) 
 
    if np.any(np.diff(A_now)<0):#apply upperenvelope + enforce no borrowing constraint 
 
        upper_envelope(par.grid_A,A_now,C_pd,par.β*EVw[idz],par.β*EVm[idz],power,res,C_tot[idx],Vw[idx],Vm[idx],Vc[idx],*pars) 
        
    else:#upperenvelope not necessary: enforce no borrowing constraint 
     
        # interpolate onto common beginning-of-period asset grid to get consumption 
        linear_interp.interp_1d_vec(A_now,C_pd,res,C_tot[idx]) 
        C_tot[idx] = np.minimum(C_tot[idx] , res) #...+ apply borrowing constraint 
     
        # compute the value function 
        Cw_priv, Cm_priv, C_pub =\
            usr.intraperiod_allocation(C_tot[idx],par.grid_Ctot,sol.pre_Ctot_Cw_priv[part,iP],sol.pre_Ctot_Cm_priv[part,iP])  
             
        linear_interp.interp_1d_vec(par.grid_A,par.β*EVw[idz],res-C_tot[idx],βEw) 
        linear_interp.interp_1d_vec(par.grid_A,par.β*EVm[idz],res-C_tot[idx],βEm)     
        Vw[idx] = usr.util(Cw_priv,C_pub,*pars2,love[0],True,1.0-par.grid_wlp[part],γ)+βEw                         
        Vm[idx] = usr.util(Cm_priv,C_pub,*pars2,love[1],True,1.0-par.grid_wlp[part],γ)+βEm 
        Vc[idx] = power*Vw[idx]+(1.0-power)*Vm[idx]
       
@njit
def before_taste_shock(par,p_Vc,n_Vc,p_Vw,n_Vw,p_Vm,n_Vm,idx,wlp,Vw,Vm):
 
    # get the probabilit of workin wlp, based on couple utility choices
    c=np.maximum(p_Vc[idx],n_Vc[idx])/par.σ # constant to avoid overflow
    v_couple=par.σ*(c+np.log(np.exp(p_Vc[idx]/par.σ-c)+np.exp(n_Vc[idx]/par.σ-c)))
    wlp[idx]=np.exp(p_Vc[idx]/par.σ-v_couple/par.σ)
    
    # now the value of making the choice: see Shepard (2019), page 11
    Δp=p_Vw[idx]-p_Vm[idx];Δn=n_Vw[idx]-n_Vm[idx]
    Vw[idx]=v_couple+(1.0-par.grid_power[idx[3]])*(wlp[idx]*Δp+(1.0-wlp[idx])*Δn)
    Vm[idx]=v_couple+(-par.grid_power[idx[3]])   *(wlp[idx]*Δp+(1.0-wlp[idx])*Δn) 
    
@njit
def check_participation_constraints(par,solpower,gridpower,list_raw,list_single,idx,
                                    list_couple=(np.zeros((1,1)),),iswomen=(True,),nosim=True):
                 
    # surplus of marriage, then its min and max given states
    Sw = list_raw[0] - list_single[0] 
    Sm = list_raw[1] - list_single[1] 
    min_Sw = np.min(Sw);min_Sm = np.min(Sm)
    max_Sw = np.max(Sw);max_Sm = np.max(Sm) 

    # if expect rebargaining, interpolate the surplus of each member at indifference points
    if ~((min_Sw >= 0.0) & (min_Sm >= 0.0)) & ~((max_Sw < 0.0) | (max_Sm < 0.0)):             
        power_at_0_w, Sm_at_0_w, Low_w, i_Low_w = interp_barg(par,Sw,Sm,iswomen=True)
        power_at_0_m, Sw_at_0_m, Low_m, i_Low_m = interp_barg(par,Sm,Sw,iswomen=False)
        
    ##################################################################
    # For a given power, find out if marriage, divorce or rebargaining
    # Then, update power and (if no simulation) update value functions
    #################################################################
    for iP,power in enumerate(gridpower):
        
        #1) all iP values are consistent with marriage
        if (min_Sw >= 0.0) & (min_Sm >= 0.0): 
            solpower[idx[iP]] = power #update power, below update value function
            if nosim:no_power_change(list_couple,list_raw,idx,iP,power)
                     
        #2) no iP values consistent with marriage
        elif (max_Sw < 0.0) | (max_Sm < 0.0): 
            solpower[idx[iP]] = -1.0 #update power, below update value function
            if nosim:divorce(list_couple,list_single,iswomen,idx[iP])
                
        #3) some iP are (invidivually) consistent with marriage: try rebargaining
        else:             
            # 3.1) woman wants to leave &  man happy to shift some bargaining power
            if (power<Low_w) & (Sm_at_0_w > 0): 
                solpower[idx[iP]] = power_at_0_w #update power, below update value function
                if nosim:do_power_change(par,list_couple,list_raw,idx,iP,power_at_0_w,i_Low_w-1,0)
                                                                          
            # 3.2) man wants to leave & woman happy to shift some bargaining power
            elif (power>Low_m) & (Sw_at_0_m > 0): 
                solpower[idx[iP]] = power_at_0_m #update power, below update value function
                if nosim:do_power_change(par,list_couple,list_raw,idx,iP,power_at_0_m,i_Low_m,i_Low_m+1)
                                        
            # 3.3) divorce: men (women) wants to leave & woman (men) not happy to shift some bargaining power
            elif ((power<Low_w) & (Sm_at_0_w <=0)) | ((power>Low_m) & (Sw_at_0_m <=0)):
                solpower[idx[iP]] = -1.0  #update power, below update value function
                if nosim:divorce(list_couple,list_single,iswomen,idx[iP])
                
            # 3.4) no-one wants to leave
            else: 
                solpower[idx[iP]] = power #update power, belowe update value function
                if nosim:no_power_change(list_couple,list_raw,idx,iP,power)
@njit
def no_power_change(list_couple,list_raw,idx,iP,power):
    for i,key in enumerate(list_couple): key[idx[iP]] = list_raw[i][iP]        
@njit
def divorce(list_couple,list_single,iswomen,idx):    
    for i,key in enumerate(list_couple):
        key[idx]=list_single[i] if iswomen[i] else list_single[i]                
@njit
def do_power_change(par,list_couple,list_raw,idx,iP,power_at_0_i,low_i,low_0):    
    for i,key in enumerate(list_couple):
        if iP==low_0: key[idx[iP]] = linear_interp_1d._interp_1d(par.grid_power,list_raw[i],power_at_0_i,low_i) 
        else:         key[idx[iP]] = list_couple[i][idx[low_0]]; # re-use that the interpolated values are identical                               
@njit
def interp_barg(par,Si,So,iswomen=True):#i: individual, o: spouse
    
    # find lowest (highest) value with positive surplus for women (men)
    i_Low_i = 1 if iswomen else  par.num_power-1-1
                     
    for iP in range(par.num_power-1):
        if iswomen:
            if (Si[iP]<0) & (Si[iP+1]>=0): i_Low_i = iP+1 
        else:                           
            if (Si[iP]>=0) & (Si[iP+1]<0): i_Low_i = iP
            
    id = i_Low_i-1 if iswomen else i_Low_i
    denom = (par.grid_power[id+1] - par.grid_power[id])
    ratio_i = (Si[id+1] - Si[id])/denom
    ratio_o = (So[id+1] - So[id])/denom
    
    power_at_zero = par.grid_power[id] - Si[id]/ratio_i                     
    So_at_zero=So[id]+ratio_o*(power_at_zero-par.grid_power[id])
        
    return power_at_zero, So_at_zero, par.grid_power[i_Low_i], i_Low_i
                
@njit
def store(Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_C_tot,n_C_tot, wlp,par,sol,t):
                
    sol.p_C_tot_remain_couple[t] = p_C_tot
    sol.n_C_tot_remain_couple[t] = n_C_tot
    sol.Vw_remain_couple[t] = Vw
    sol.Vm_remain_couple[t] = Vm
    sol.p_Vw_remain_couple[t] = p_Vw
    sol.p_Vm_remain_couple[t] = p_Vm
    sol.n_Vw_remain_couple[t] = n_Vw
    sol.n_Vm_remain_couple[t] = n_Vm
    sol.remain_WLP[t] = wlp
                 
##################################
#        SIMULATIONS
#################################

@njit(parallel=parallel)
def simulate_lifecycle(sim,sol,par):
    
    # unpacking some values to help numba optimize
    A=sim.A;Aw=sim.Aw;Am=sim.Am;rel=sim.rel;power=sim.power;C_tot=sim.C_tot;rel_lag=sim.rel_lag;power_lag=sim.power_lag
    love=sim.love;shock_love=sim.shock_love;iz=sim.iz;wlp=sim.WLP;incw=sim.incw;incm=sim.incm;ih=sim.ih;
    
    interp2d = lambda a,b,c : linear_interp.interp_2d(par.grid_power,par.grid_A,a,b,c)
    interp1d = lambda a,b   : linear_interp.interp_1d(par.grid_A,               a,b) 
    
    for i in prange(par.simN):
        for t in range(par.simT):
              
            # Copy variables from t-1 or initial condition. Initial (t>0) assets: preamble (later in the simulation)   
            rel_lag[i,t] = rel[i,t-1]                                                if t>0 else sim.init_rel[i]
            power_lag[i,t] = power[i,t-1]                                            if t>0 else sim.init_power[i]
            love[i,t] = usr.mc_simulate(love[i,t-1],par.Πl[t-1],shock_love[i,t])     if t>0 else sim.init_love[i]
            iz[i,t] = usr.mc_simulate(iz[i,t-1],par.Π[t-1],sim.shock_z[i,t])         if t>0 else sim.init_z[i]
            Π = par.Πh_p if wlp[i,t-1] else par.Πh_n 
            ih[i,t] = usr.mc_simulate(ih[i,t-1],Π,sim.shock_h[i,t])                  if t>0 else sim.init_ih[i]

            # indices and resources
            idx = (t,rel_lag[i,t],ih[i,t],iz[i,t],slice(None),love[i,t]);iz_w=iz[i,t]//par.num_zm;iz_m=iz[i,t]%par.num_zw
            incw[i,t]=np.exp(np.log(par.grid_zw[t,iz_w])+par.grid_h[ih[i,t]]);incm[i,t]=par.grid_zm[t,iz_m]         
                    
            # first check if they want to remain together and what the bargaining power will be if they do.
            if rel_lag[i,t]<2:                   

                # value of transitioning into singlehood
                list_single = (linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,ih[i,t],iz_w],Aw[i,t]),
                               linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,      0,iz_m],Am[i,t]))

                list_raw    = (np.array([interp1d(sol.Vw_remain_couple[idx][iP],A[i,t]) for iP in range(par.num_power)]),
                               np.array([interp1d(sol.Vm_remain_couple[idx][iP],A[i,t]) for iP in range(par.num_power)]))

                check_participation_constraints(par,power,np.array([power_lag[i,t]]),list_raw,list_single,[(i,t)],nosim=False)
                             
                rel[i,t] = 2 if power[i,t] < 0.0 else rel_lag[i,t] # partnership status: divorce is coded as -1
                    
            else: rel[i,t] = 2 # remain single

            # update behavior
            if rel[i,t]<2:
                              
                # first decide about labor participation (if not taste shocks this should be different) 
                part_i=interp2d(sol.remain_WLP[idx],power[i,t],A[i,t])#
                wlp[i,t]=(part_i>sim.shock_taste[i,t])
                               
                # optimal consumption allocation if couple (note use of the updated index)
                sol_C_tot = sol.p_C_tot_remain_couple[idx] if wlp[i,t] else sol.n_C_tot_remain_couple[idx] 
                C_tot[i,t] = interp2d(sol_C_tot,power[i,t],A[i,t])

                # update end-of-period states
                M_resources = usr.resources_couple(par,t,ih[i,t],iz[i,t],A[i,t])[wlp[i,t]]
                if t< par.simT-1:A[i,t+1] = M_resources - C_tot[i,t]#
                if t< par.simT-1:Aw[i,t+1] =       par.div_A_share*A[i,t]*(1.0-par.sep_cost[rel[i,t]])# in case of divorce 
                if t< par.simT-1:Am[i,t+1] = (1.0-par.div_A_share)*A[i,t]*(1.0-par.sep_cost[rel[i,t]])# in case of divorce 
               
            else: # single
               
                # pick relevant solution for single
                sol_single_w = sol.Cw_tot_single[t,ih[i,t],iz_w]
                sol_single_m = sol.Cm_tot_single[t,      0,iz_m]

                # optimal consumption allocations
                Cw_tot = linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw[i,t])
                Cm_tot = linear_interp.interp_1d(par.grid_Am,sol_single_m,Am[i,t])   
                C_tot[i,t] = Cw_tot + Cm_tot

                # update end-of-period states
                Mw = par.R*Aw[i,t] + incw[i,t] # total resources woman
                Mm = par.R*Am[i,t] + incm[i,t] # total resources man
                if t< par.simT-1: Aw[i,t+1] = Mw - Cw_tot
                if t< par.simT-1: Am[i,t+1] = Mm - Cm_tot