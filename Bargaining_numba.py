import numpy as np
import scipy.optimize as optimize

from EconModel import EconModelClass
from consav.grids import nonlinspace
from consav import linear_interp, linear_interp_1d
from consav import quadrature
from numba import njit,prange
from EconModel import EconModelClass, jit
from quantecon.optimize.nelder_mead import nelder_mead
# user-specified functions
import UserFunctions_numba as usr

# set gender indication as globals
woman = 1
man = 2
from numba import config 
config.DISABLE_JIT = False
config.NUMBA_DEBUGINFO = True
config.NUMBA_DEBUG = True
config.NUMBA_BOUNDSCHECK = True
config.NUMBA_FULL_TRACEBACKS = True

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
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
        
        par.R = 1.03
        par.β = 1.0/par.R # Discount factor
        
        par.div_A_share = 0.5 # divorce share of wealth to wife

        # Utility: CES aggregator or additively linear
        par.ρ = 2.0        # CRRA      
        par.α1 = 1.0
        par.α2 = 1.0
        par.ϕ1 = 0.2
        par.ϕ2 = 0.2
        
        # production of home good
        par.θ = 1.0#0.21 #weight on money vs. time to produce home good
        par.λ = 0.19 #elasticity betwen money and time in public good
        par.tb = 0.2 #time spend on public goods by singles
        
        # state variables
        par.T = 10
        
        # wealth
        par.num_A = 50
        par.max_A = 5.0
        
        # bargaining power
        par.num_power = 21

        # love/match quality
        par.num_love = 41
        par.max_love = 1.0

        par.sigma_love = 0.1
        par.num_shock_love = 5
        
        # income of men and women
        par.num_zw=3
        par.num_zm=3
        par.num_z=par.num_zm*par.num_zw
        
        par.t0w=0.0;par.t1w=0.0;par.t2w=0.0;par.σzw=0.00001;par.σ0zw=0.00004
        par.t0m=0.0;par.t1m=0.0;par.t2m=0.0;par.σzm=0.00001;par.σ0zm=0.00004

        # pre-computation
        par.num_Ctot = 100
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
        sol.Cw_priv_single = np.nan + np.ones(shape_singlew)
        sol.Cm_priv_single = np.nan + np.ones(shape_singlem)
        sol.Cw_pub_single = np.nan + np.ones(shape_singlew)
        sol.Cm_pub_single = np.nan + np.ones(shape_singlem)
        sol.Cw_tot_single = np.nan + np.ones(shape_singlew)
        sol.Cm_tot_single = np.nan + np.ones(shape_singlem)

        sol.Vw_trans_single = np.nan + np.ones(shape_singlew)
        sol.Vm_trans_single = np.nan + np.ones(shape_singlem)
        sol.Cw_priv_trans_single = np.nan + np.ones(shape_singlew)
        sol.Cm_priv_trans_single = np.nan + np.ones(shape_singlem)
        sol.Cw_pub_trans_single = np.nan + np.ones(shape_singlew)
        sol.Cm_pub_trans_single = np.nan + np.ones(shape_singlem)
        sol.Cw_tot_trans_single = np.nan + np.ones(shape_singlew)
        sol.Cm_tot_trans_single = np.nan + np.ones(shape_singlem)

        # couples
        shape_couple = (par.T,par.num_z,par.num_power,par.num_love,par.num_A)
        sol.Vw_couple = np.nan + np.ones(shape_couple)
        sol.Vm_couple = np.nan + np.ones(shape_couple)
        
        sol.Cw_priv_couple = np.nan + np.ones(shape_couple)
        sol.Cm_priv_couple = np.nan + np.ones(shape_couple)
        sol.C_pub_couple = np.nan + np.ones(shape_couple)
        sol.C_tot_couple = np.nan + np.ones(shape_couple)

        sol.Vw_remain_couple = np.nan + np.ones(shape_couple)
        sol.Vm_remain_couple = np.nan + np.ones(shape_couple)
        
        sol.Cw_priv_remain_couple = np.nan + np.ones(shape_couple)
        sol.Cm_priv_remain_couple = np.nan + np.ones(shape_couple)
        sol.C_pub_remain_couple = np.nan + np.ones(shape_couple)
        sol.C_tot_remain_couple = np.nan + np.ones(shape_couple)

        sol.power_idx = np.zeros(shape_couple,dtype=np.int_)
        sol.power = np.zeros(shape_couple)

        # temporary containers
        sol.savings_vec = np.zeros(par.num_shock_love)
        sol.Vw_plus_vec = np.zeros(par.num_shock_love) 
        sol.Vm_plus_vec = np.zeros(par.num_shock_love) 

        # # EGM
        # sol.marg_V_couple = np.zeros(shape_couple)
        # sol.marg_V_remain_couple = np.zeros(shape_couple)

        # shape_egm = (par.num_power,par.num_love,par.num_A_pd)
        # sol.EmargU_pd = np.zeros(shape_egm)
        # sol.C_tot_pd = np.zeros(shape_egm)
        # sol.M_pd = np.zeros(shape_egm)

        # pre-compute optimal consumption allocation
        shape_pre = (par.num_power,par.num_Ctot)
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
        
        sim.A = np.nan + np.ones(shape_sim)
        sim.Aw = np.nan + np.ones(shape_sim)
        sim.Am = np.nan + np.ones(shape_sim)
        sim.couple = np.nan + np.ones(shape_sim)
        sim.power_idx = np.ones(shape_sim,dtype=np.int_)
        sim.power = np.nan + np.ones(shape_sim)
        sim.love = np.nan + np.ones(shape_sim)

        # shocks
        np.random.seed(par.seed)
        sim.draw_love = np.random.normal(size=shape_sim)
        #shock_zw=np.random.random_sample((par.simN,par.simT))
        #shock_zm=np.random.random_sample((par.simN,par.simT))

        # initial distribution
        sim.init_A = par.grid_A[0] + np.zeros(par.simN)
        sim.init_Aw = par.div_A_share * sim.init_A #np.zeros(par.simN)
        sim.init_Am = (1.0 - par.div_A_share) * sim.init_A #np.zeros(par.simN)
        sim.init_couple = np.ones(par.simN,dtype=np.bool)
        sim.init_power_idx = par.num_power//2 * np.ones(par.simN,dtype=np.int_)
        sim.init_love = np.zeros(par.simN)
        sim.init_zw = np.ones(par.simN)*par.num_zw//2
        sim.init_zm = np.ones(par.simN)*par.num_zm//2
        

        # matw=usr.addaco_nonst(2,par.σ0zw,0.0,par.num_zw)[1][0][0,:]
        # matm=usr.addaco_nonst(2,par.σ0zw,0.0,par.num_zw)[1][0][0,:]
        # sim.init_zw=np.sum((shock_zw[:,0,None]> np.cumsum(matw)[None,:]), axis=1)
        # sim.init_zm=np.sum((shock_zw[:,0,None]> np.cumsum(matm)[None,:]), axis=1)
    def setup_grids(self):
        par = self.par
        
        # wealth. Single grids are such to avoid interpolation
        par.grid_A = nonlinspace(0.0,par.max_A,par.num_A,1.1)

        par.grid_Aw = par.div_A_share * par.grid_A
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # power. non-linear grid with more mass in both tails.
        odd_num = np.mod(par.num_power,2)
        first_part = nonlinspace(0.0,0.5,(par.num_power+odd_num)//2,1.3)
        last_part = np.flip(1.0 - nonlinspace(0.0,0.5,(par.num_power-odd_num)//2 + 1,1.3))[1:]
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

        # pre-computation
        par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)

        # EGM
        par.grid_util = np.nan + np.ones((par.num_power,par.num_Ctot))
        par.grid_marg_u = np.nan + np.ones(par.grid_util.shape)
        par.grid_inv_marg_u = np.flip(par.grid_Ctot)
        par.grid_marg_u_for_inv = np.nan + np.ones(par.grid_util.shape)

        par.grid_A_pd = nonlinspace(0.0,par.max_A,par.num_A_pd,1.1)
        
        # income shocks grids: singles and couples
        par.grid_zw,par.Π_zw= usr.labor_income(par.t0w,par.t1w,par.t2w,par.T,par.σzw,par.σ0zw,par.num_zw)
        par.grid_zm,par.Π_zm= usr.labor_income(par.t0m,par.t1m,par.t2m,par.T,par.σzm,par.σ0zm,par.num_zm)
        par.Π=[np.kron(par.Π_zw[t],par.Π_zm[t]) for t in range(par.T-1)]
        
    def solve(self):

        with jit(self) as model:#This allows passing sol and par to jiit functions  
            
            #Import parameters and arrays for solution
            par = model.par
            sol = model.sol
            
            # precompute the optimal intra-temporal consumption allocation for couples given total consumpotion
            # this needs to be jitted at some point
            solve_intraperiod_couple(sol,par)
            
            # loop backwards and obtain policy functions
            for t in reversed(range(par.T)):
                solve_single(sol,par,t)
                solve_couple(sol,par,t)
    
            # store results
            store(sol)
            
                     
    def simulate(self):
        
        with jit(self) as model:    
            
            #Import parameter, policy functions and simulations arrays
            par = model.par
            sol = model.sol
            sim = model.sim
            
            #Call routing performing the simulation
            simulate_lifecycle(sim,sol,par)
            
    

############
# routines #
############

@njit
def integrate_single(sol,par,t):
    Ew,Em=np.ones((par.num_zw,par.num_A)),np.ones((par.num_zm,par.num_A)) 
    for iA in range(par.num_A):Ew[:,iA]=sol.Vw_single[t+1,:,iA].flatten() @ par.Π_zw[t] 
    for iA in range(par.num_A):Em[:,iA]=sol.Vm_single[t+1,:,iA].flatten() @ par.Π_zm[t]
    
    return Ew,Em
    
@njit(parallel=True)
def solve_single(sol,par,t):

    #Integrate first (this can be improved)
    if t<par.T-1: Ew, Em = integrate_single(sol,par,t)
        
    # parameters used for optimization: partial unpacking improves speed
    parsw=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)
    parsm=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)
            
    #Pre-define outcomes (if update .sol directly, parallelization go crazy)
    vw,cwpr,cwpu=np.ones((3,par.num_zw,par.num_A))
    vm,cmpr,cmpu=np.ones((3,par.num_zm,par.num_A))
    
    #Function to be minimized later to maximize utility
    def obj(C_tot,M,gender,V_next,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,β,grid_A):
        return -value_of_choice_single(C_tot,M,gender,V_next,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,β,grid_A)
    
    # Women
    for iA in prange(par.num_A):
        for iz in range(par.num_zw):

            # resources
            Aw = par.grid_Aw[iA]
    
            Mw = usr.resources_single(Aw,woman,par.grid_zw[t,iz],par.grid_zm[t,iz],par.R) 
    
            if t == (par.T-1): # terminal period
                
                # intra-period allocation: consume all resources
                cwpr[iz,iA],cwpu[iz,iA] = intraperiod_allocation_single(Mw,woman,*parsw)
                vw[iz,iA] = usr.util(cwpr[iz,iA],cwpu[iz,iA],woman,*parsm)
            
            else: # earlier periods
                
                # search over optimal total consumption, C. obj_s=-value of choice of singles
                argsw=(Mw,woman,Ew[iz,:],*parsw,par.β,par.grid_Aw)
                
                Cw = usr.optimizer(obj,1.0e-8, Mw-1.0e-8,args=argsw)

                # store results
                cwpr[iz,iA],cwpu[iz,iA] = intraperiod_allocation_single(Cw,woman,*parsw)
                vw[iz,iA] = value_of_choice_single(Cw,*argsw)
                
    # Men
    for iA in prange(par.num_A):
        for iz in range(par.num_zm):
            
            # resources
            Am = par.grid_Am[iA]
    
            Mm = usr.resources_single(Am,man,par.grid_zw[t,iz],par.grid_zm[t,iz],par.R) 
    
            if t == (par.T-1): # terminal period
                
                # intra-period allocation: consume all resources
                cmpr[iz,iA],cmpu[iz,iA] = intraperiod_allocation_single(Mm,man,*parsm)
                vm[iz,iA] = usr.util(cmpr[iz,iA],cmpu[iz,iA],man,*parsm)
            
            else: # earlier periods
                
                # search over optimal total consumption, C. obj_s=-value of choice of singles
                argsm=(Mm,man  ,Em[iz,:],*parsm,par.β,par.grid_Am)
                
                Cm = usr.optimizer(obj,1.0e-8, Mm-1.0e-8,args=argsm)
                
                # store results
                cmpr[iz,iA],cmpu[iz,iA] = intraperiod_allocation_single(Cm,man,*parsm)
                vm[iz,iA] = value_of_choice_single(Cm,*argsm)  
                
    sol.Vw_single[t,:,:]=vw.copy();sol.Vm_single[t,:,:]=vm.copy()
    sol.Cw_priv_single[t,:,:]=cwpr.copy();sol.Cm_priv_single[t,:,:]=cmpr.copy()
    sol.Cw_pub_single[t,:,:]=cwpu.copy();sol.Cm_pub_single[t,:,:]=cmpu.copy()

@njit
def solve_couple(sol,par,t):
 
    #Solve the couples's problem for current pareto weight
    remain_Cw_priv,remain_Cm_priv,remain_C_pub,remain_Vw,remain_Vm=solve_remain_couple(par,sol,t)
    
    #Check participation constrints and eventually update policy function and pareto weights
    check_participation_constraints(remain_Cw_priv,remain_Cm_priv,remain_C_pub,remain_Vw,remain_Vm,par,sol,t)
        

@njit
def intraperiod_allocation_single(C_tot,gender,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb):
    
    args=(C_tot,gender,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb)
    def obj_s(x,Ctot,pgender,pρ,pϕ1,pϕ2,pα1,pα2,pθ,pλ,ptb):#=-utility of consuming x
        return -usr.util(x,Ctot-x,pgender,pρ,pϕ1,pϕ2,pα1,pα2,pθ,pλ,ptb)
    
    #find private and public expenditure to max util
    C_priv = usr.optimizer(obj_s,1.0e-6, C_tot - 1.0e-6,args=args) 
    
    C_pub = C_tot - C_priv
    return C_priv,C_pub


@njit
def intraperiod_allocation(C_tot,num_Ctot,grid_Ctot,pre_Ctot_Cw_priv,pre_Ctot_Cm_priv):

    # interpolate pre-computed solution
    j1 = linear_interp.binary_search(0,num_Ctot,grid_Ctot,C_tot)
    Cw_priv = linear_interp_1d._interp_1d(grid_Ctot,pre_Ctot_Cw_priv,C_tot,j1)
    Cm_priv = linear_interp_1d._interp_1d(grid_Ctot,pre_Ctot_Cm_priv,C_tot,j1)
    C_pub = C_tot - Cw_priv - Cm_priv 

    return Cw_priv, Cm_priv, C_pub



    
#@njit at some point this needs to be jiited...
def solve_intraperiod_couple(sol,par):
        
    C_pub,  Cw_priv, Cm_priv = np.ones((3,par.num_power,par.num_Ctot))
    
    bounds = optimize.Bounds(1e-8, 1.0, keep_feasible=True)#bounds for minimization
    
    x0 = np.array([0.33,0.33])#initial condition (to be updated)
    
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)
    
    @njit
    def uobj(x,ct,pw):#function to minimize
        pubc=ct*(1.0-np.sum(x))
        return - (pw*usr.util(x[0]*ct,pubc,woman,*pars,True,False) + (1.0-pw)*usr.util(x[1]*ct,pubc,man,*pars,True,False))
    
   
    for iP,power in enumerate(par.grid_power):
        for i,C_tot in enumerate(par.grid_Ctot):
            
            # estimate
            res = optimize.minimize(uobj,x0,bounds=bounds,args=(C_tot,power))
        
            # unpack
            sol.pre_Ctot_Cw_priv[iP,i]= res.x[0]*C_tot
            sol.pre_Ctot_Cm_priv[iP,i]= res.x[1]*C_tot
            sol.pre_Ctot_C_pub[iP,i] = C_tot - Cw_priv[iP,i] - Cm_priv[iP,i]
            
            # update initial conidtion
            x0=res.x if i<par.num_Ctot-1 else np.array([0.33,0.33])




@njit(parallel=True)
def check_participation_constraints(remain_Cw_priv, remain_Cm_priv, remain_C_pub, remain_Vw, remain_Vm,par,sol,t):
 
    power_idx=sol.power_idx
    power=sol.power
            
    for iL in prange(par.num_love):
        for iA in range(par.num_A):
            for iz in range(par.num_z):
               
                # check the participation constraints
                idx_single_w = (t,iz//par.num_zm,iA);idx_single_m = (t,iz%par.num_zw,iA)
     
                
                list_couple = (sol.Vw_couple,sol.Vm_couple,sol.Cw_priv_couple,sol.Cm_priv_couple,sol.C_pub_couple)#list_start_as_couple
                list_raw = (remain_Vw[iz,iL,iA,:],remain_Vm[iz,iL,iA,:],remain_Cw_priv[iz,iL,iA,:],remain_Cm_priv[iz,iL,iA,:],remain_C_pub[iz,iL,iA,:])#list_remain_couple
                list_single = (sol.Vw_single,sol.Vm_single,sol.Cw_priv_single,sol.Cm_priv_single,sol.Cw_pub_single) # list_trans_to_single: last input here not important in case of divorce
                list_single_w = (True,False,True,False,True) # list that say whether list_single[i] is about w (as oppoesd to m)
                
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
                    sol.Cw_priv_remain_couple[idx] = remain_Cw_priv[idz] 
                    sol.Cm_priv_remain_couple[idx] = remain_Cm_priv[idz]
                    sol.C_pub_remain_couple[idx] = remain_C_pub[idz]
                    sol.Vw_remain_couple[idx] = remain_Vw[idz]
                    sol.Vm_remain_couple[idx] = remain_Vm[idz]

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

@njit
def value_of_choice_single(C_tot,M,gender,EV_next,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,β,grid_A):
   
    # flow-utility
    C_priv, C_pub =  intraperiod_allocation_single(C_tot,gender,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb)
    Util = usr.util(C_priv,C_pub,gender,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb)
    
    # continuation value
    A = M - C_tot
    
    EVnext = linear_interp.interp_1d(grid_A,EV_next,A)
    
    # return discounted sum
    return Util + β*EVnext


@njit 
def value_of_choice_couple(Ctot,tt,M_resources,iL,power,Eaw,Eam,pre_Ctot_Cw_priviP, 
        pre_Ctot_Cm_priviP,Vw_plus_vec,Vm_plus_vec, grid_love,num_Ctot,grid_Ctot, 
        ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,couple,ishom, 
        T,grid_A,grid_weight_love,β,num_shock_love,love_next_vec,savings_vec): 
 
    pars= (ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,grid_love[iL],couple,ishom) 
     
    # current utility from consumption allocation    
    Cw_priv, Cm_priv, C_pub = intraperiod_allocation(Ctot,num_Ctot,grid_Ctot,pre_Ctot_Cw_priviP,pre_Ctot_Cm_priviP) 
    Vw = usr.util(Cw_priv,C_pub,woman,*pars) 
    Vm = usr.util(Cm_priv,C_pub,man  ,*pars) 
 
    EVw_plus=linear_interp.interp_1d(grid_A, Eaw, M_resources - Ctot) 
    EVm_plus=linear_interp.interp_1d(grid_A, Eam, M_resources - Ctot) 
     
    Vw += β*EVw_plus 
    Vm += β*EVm_plus 
      
    return power*Vw + (1.0-power)*Vm, Cw_priv, Cm_priv, C_pub, Vw,Vm 

@njit#this should be parallelized, but it's tricky... 
def integrate(par,sol,t): 
     
    EVw,EVm,pEVw,pEVm= np.ones((4,par.num_z,par.num_love,par.num_power,par.num_A)) 
    aw=np.ones(par.num_shock_love) 
    am=np.ones(par.num_shock_love) 
    love_next_vec=np.ones((par.num_love,par.num_shock_love)) 
     
    for iL in range(par.num_love):love_next_vec[iL,:] = par.grid_love[iL] + par.grid_shock_love 
     
    for iL in range(par.num_love): 
        for iP in range(par.num_power):     
            for iA in range(par.num_A): 
                #for iz in range(par.num_z): 
                 
                #Integrate income shocks
                pEVw[:,iL,iP,iA] = sol.Vw_couple[t+1,:,iP,iL,iA].flatten() @ par.Π[t] 
                pEVm[:,iL,iP,iA] = sol.Vm_couple[t+1,:,iP,iL,iA].flatten() @ par.Π[t] 
   
    for iL in range(par.num_love): 
        for iP in range(par.num_power):     
            for iA in range(par.num_A): 
                for iz in range(par.num_z): 
                    #Integrate love shocks
                    linear_interp.interp_1d_vec_mon_noprep(par.grid_love,pEVw[iz,:,iP,iA],love_next_vec[iL,:],aw) 
                    linear_interp.interp_1d_vec_mon_noprep(par.grid_love,pEVm[iz,:,iP,iA],love_next_vec[iL,:],am) 
                     
                    EVw[iz,iL,iP,iA]=aw @ par.grid_weight_love 
                    EVm[iz,iL,iP,iA]=am @ par.grid_weight_love                 
           
    return EVw,EVm 

@njit(parallel=True)
def solve_remain_couple(par,sol,t):

    #Integration   
    if t<(par.T-1): EVw, EVm = integrate(par,sol,t) 

    # initialize
    remain_Vw,remain_Vm,remain_Cw_priv,remain_Cm_priv,remain_C_pub = np.ones((5,par.num_z,par.num_love,par.num_A,par.num_power))
    
    #parameters: useful to unpack this to improve speed
    couple=1.0;ishome=0.0
    pars1=(par.grid_love,par.num_Ctot,par.grid_Ctot, par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb,couple,ishome,
           par.T,par.grid_A,par.grid_weight_love,par.β,par.num_shock_love)
    
    pars2=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)
    
    for iL in prange(par.num_love):
        
        #Variables defined in advance to improve speed
        Vw_plus_vec,Vm_plus_vec=np.ones(par.num_shock_love)+np.nan,np.ones(par.num_shock_love)+np.nan
        love_next_vec = par.grid_love[iL] + par.grid_shock_love
        savings_vec = np.ones(par.num_shock_love)
        
        for iA in range(par.num_A):
            for iP in range(par.num_power):           
                for iz in range(par.num_z):
                       
                    idx=(iz,iL,iA,iP);izw=iz//par.num_zm;izm=iz%par.num_zw# indexes
                             
                    M_resources = usr.resources_couple(par.grid_A[iA],par.grid_zw[t,izw],par.grid_zw[t,izm],par.R)
                    
                    # continuation values
                    if t==(par.T-1):#last period
                 
                        C_tot = M_resources#No savings, it's the last period
                        
                        # current utility from consumption allocation
                        remain_Cw_priv[idx], remain_Cm_priv[idx], remain_C_pub[idx] =\
                            intraperiod_allocation(C_tot,par.num_Ctot,par.grid_Ctot,sol.pre_Ctot_Cw_priv[iP],sol.pre_Ctot_Cm_priv[iP])
                        remain_Vw[idx] = usr.util(remain_Cw_priv[idx],remain_C_pub[idx],woman,*pars2,par.grid_love[iL],couple,ishome)
                        remain_Vm[idx] = usr.util(remain_Cm_priv[idx],remain_C_pub[idx],man  ,*pars2,par.grid_love[iL],couple,ishome)
                        
                    else:#periods before the last
                                
                        #first find optimal total consumption
                        args=(t,M_resources,iL,par.grid_power[iP],EVw[iz,iL,iP,:],EVm[iz,iL,iP,:],
                              sol.pre_Ctot_Cw_priv[iP],sol.pre_Ctot_Cm_priv[iP],Vw_plus_vec,Vm_plus_vec,*pars1,love_next_vec,savings_vec)             
                       
                        def obj(x,*args):#function to minimize (= maximize value of choice)
                            return - value_of_choice_couple(x,*args)[0] 
                        C_tot=usr.optimizer(obj,1.0e-6, M_resources - 1.0e-6,args=args)
    
                        # current utility from consumption allocation
                        _, remain_Cw_priv[idx], remain_Cm_priv[idx], remain_C_pub[idx], remain_Vw[idx],remain_Vm[idx] =\
                            value_of_choice_couple(C_tot,*args)

    # return objects
    return remain_Cw_priv, remain_Cm_priv, remain_C_pub, remain_Vw, remain_Vm

#@njit
def store(sol):
    
    # total consumption
    sol.C_tot_couple[::] = sol.Cw_priv_couple[::] + sol.Cm_priv_couple[::] + sol.C_pub_couple[::]
    sol.C_tot_remain_couple[::] = sol.Cw_priv_remain_couple[::] + sol.Cm_priv_remain_couple[::] + sol.C_pub_remain_couple[::]
    sol.Cw_tot_single[::] = sol.Cw_priv_single[::] + sol.Cw_pub_single[::]
    sol.Cm_tot_single[::] = sol.Cm_priv_single[::] + sol.Cm_pub_single[::]

    # value of transitioning to singlehood. Done here because absorbing . it is the same as entering period as single.
    sol.Vw_trans_single[::] = sol.Vw_single[::].copy()
    sol.Vm_trans_single[::] = sol.Vm_single[::].copy()
    sol.Cw_priv_trans_single[::] = sol.Cw_priv_single[::].copy()
    sol.Cm_priv_trans_single[::] = sol.Cm_priv_single[::].copy()
    sol.Cw_pub_trans_single[::] = sol.Cw_pub_single[::].copy()
    sol.Cm_pub_trans_single[::] = sol.Cm_pub_single[::].copy()
    sol.Cw_tot_trans_single[::] = sol.Cw_tot_single[::].copy()
    sol.Cm_tot_trans_single[::] = sol.Cm_tot_single[::].copy()

@njit
def simulate_lifecycle(sim,sol,par):     
    

    # unpacking some values
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)#single
    pres=(par.inc_w,par.inc_m,par.R)# resources single
    
    
    for i in range(par.simN):
        for t in range(par.simT):

            # state variables
            if t==0:
                A_lag = sim.init_A[i]
                Aw_lag = sim.init_Aw[i]
                Am_lag = sim.init_Am[i]
                couple_lag = sim.init_couple[i]
                power_idx_lag = sim.init_power_idx[i]
                love = sim.love[i,t] = sim.init_love[i]

            else:
                A_lag = sim.A[i,t-1]
                Aw_lag = sim.Aw[i,t-1]
                Am_lag = sim.Am[i,t-1]
                couple_lag = sim.couple[i,t-1]
                power_idx_lag = sim.power_idx[i,t-1]
                love = sim.love[i,t]


            # first check if they want to remain together and what the bargaining power will be if they do.
            if couple_lag:                   

                # value of transitioning into singlehood
                Vw_single = linear_interp.interp_1d(par.grid_Aw,sol.Vw_trans_single[t],Aw_lag)
                Vm_single = linear_interp.interp_1d(par.grid_Am,sol.Vm_trans_single[t],Am_lag)

                idx = (t,power_idx_lag)
                Vw_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[idx],love,A_lag)
                Vm_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[idx],love,A_lag)

                if ((Vw_couple_i>=Vw_single) & (Vm_couple_i>=Vm_single)):
                    power_idx = power_idx_lag

                else:
                    # value of partnerhip for all levels of power
                    Vw_couple = np.zeros(par.num_power)
                    Vm_couple = np.zeros(par.num_power)
                    for iP in range(par.num_power):
                        idx = (t,iP)
                        Vw_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[idx],love,A_lag)
                        Vm_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[idx],love,A_lag)

                    # check participation constraint 
                    Sw = Vw_couple - Vw_single
                    Sm = Vm_couple - Vm_single
                    power_idx = update_bargaining_index(Sw,Sm,power_idx_lag, par.num_power)

                # infer partnership status
                if power_idx < 0.0: # divorce is coded as -1
                    sim.couple[i,t] = False

                else:
                    sim.couple[i,t] = True

            else: # remain single

                sim.couple[i,t] = False

            # update behavior
            if sim.couple[i,t]:
                
                # optimal consumption allocation if couple
                sol_C_tot = sol.C_tot_couple[t,power_idx] 
                C_tot = linear_interp.interp_2d(par.grid_love,par.grid_A,sol_C_tot,love,A_lag)
                args=(C_tot,par.num_Ctot,par.grid_Ctot,sol.pre_Ctot_Cw_priv[power_idx],sol.pre_Ctot_Cm_priv[power_idx])
                sim.Cw_priv[i,t], sim.Cm_priv[i,t], C_pub = intraperiod_allocation(*args)
                sim.Cw_pub[i,t] = C_pub
                sim.Cm_pub[i,t] = C_pub

                # update end-of-period states
                M_resources = usr.resources_couple(A_lag,*pres) 
                sim.A[i,t] = M_resources - sim.Cw_priv[i,t] - sim.Cm_priv[i,t] - sim.Cw_pub[i,t]
                if t<(par.simT-1):
                    sim.love[i,t+1] = love + par.sigma_love*sim.draw_love[i,t+1]

                # in case of divorce
                sim.Aw[i,t] = par.div_A_share * sim.A[i,t]
                sim.Am[i,t] = (1.0-par.div_A_share) * sim.A[i,t]

                sim.power_idx[i,t] = power_idx
                sim.power[i,t] = par.grid_power[sim.power_idx[i,t]]

            else: # single

                # pick relevant solution for single, depending on whether just became single
                idx_sol_single = t
                sol_single_w = sol.Cw_tot_trans_single[idx_sol_single]
                sol_single_m = sol.Cm_tot_trans_single[idx_sol_single]
                if (power_idx_lag<0):
                    sol_single_w = sol.Cw_tot_single[idx_sol_single]
                    sol_single_m = sol.Cm_tot_single[idx_sol_single]

                # optimal consumption allocations
                Cw_tot = linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw_lag)
                Cm_tot = linear_interp.interp_1d(par.grid_Am,sol_single_m,Am_lag)
                
                sim.Cw_priv[i,t],sim.Cw_pub[i,t] = intraperiod_allocation_single(Cw_tot,woman,*pars)
                sim.Cm_priv[i,t],sim.Cm_pub[i,t] = intraperiod_allocation_single(Cm_tot,man,*pars)

                # update end-of-period states
                Mw = usr.resources_single(Aw_lag,woman,*pres)
                Mm = usr.resources_single(Am_lag,man,*pres) 
                sim.Aw[i,t] = Mw - sim.Cw_priv[i,t] - sim.Cw_pub[i,t]
                sim.Am[i,t] = Mm - sim.Cm_priv[i,t] - sim.Cm_pub[i,t]

                # not updated: nans
                # sim.power[i,t] = np.nan
                # sim.love[i,t+1] = np.nan 
                # sim.A[i,t] = np.nan

                sim.power_idx[i,t] = -1

    # total consumption
    sim.Cw_tot[::] = sim.Cw_priv[::] + sim.Cw_pub[::]
    sim.Cm_tot[::] = sim.Cm_priv[::] + sim.Cm_pub[::]
    sim.C_tot[::] = sim.Cw_priv[::] + sim.Cm_priv[::] + sim.Cw_pub[::]