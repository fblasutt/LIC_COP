from interpolation.splines import prefilter,eval_spline
import numpy as np
from numba import njit,prange,config
from consav import linear_interp, linear_interp_1d
import UserFunctions_numba as usr
import setup

#general configuratiion and glabal variables (common across files)
config.DISABLE_JIT = setup.nojit;parallel=setup.parallel
woman=setup.woman;man=setup.man

###############################################################################
# SINGLE'S FUNCTIONS
###############################################################################
@njit
def integrate_single(sol,par,t):
    Ew=np.ones((par.num_h,par.num_zw,par.num_A));Em=np.ones((1,par.num_zm,par.num_A)) 
     
    # men
    for iA in prange(par.num_A):Em[:,:,iA]  = sol.Vm_single[t+1,:,:,iA].flatten() @ par.Π_zm[t]
    
    # women: always work, is low human (1) capital have a change to improve their position
    for iA in prange(par.num_A):
        for iz in range(par.num_zw):
            for ih in range(par.num_h):
                Ew[ih,iz,iA]  = sol.Vw_single[t+1,:,iz,iA] @ par.Πh_p[ih,:]

    return Ew,Em 

@njit(parallel=parallel)
def solve_single(sol,par,t):

    #Integrate first to get continuation value (0 if terminal period)
    Ew, Em = integrate_single(sol,par,t) if t<par.T-1 else (np.zeros((par.num_h,par.num_zw,par.num_A)),np.zeros((1,par.num_zm,par.num_A)))
     
    # parameters used for optimization: partial unpacking improves speed
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)
            
    #Pre-define outcomes (if update .sol directly, parallelization go crazy)
    vw,cw=np.ones((2,par.num_h,par.num_zw,par.num_A));vm,cm=np.ones((2,1,par.num_zm,par.num_A))
    
    
    for iA in prange(par.num_A):
        
        # Women
        for iz in range(par.num_zw):
            for ih in range(par.num_h):

                Mw = (par.R*par.grid_Aw[iA] + np.exp(np.log(par.grid_zw[t,iz]) + par.grid_h[ih]))#*0.0000001 #resources
    
                # search over optimal total consumption, , and store (OPPOSITE) of value
                argsw=(Mw,Ew[ih,iz,:],*pars,par.β,par.grid_Aw)             
                cw[ih,iz,iA], vw[ih,iz,iA] = usr.optimizer(lambda *argsw:-value_of_choice_single(*argsw),1.0e-8, Mw-1.0e-8,args=argsw)
              
        # Men
        for iz in range(par.num_zm):
            
            Mm = (par.R*par.grid_Am[iA] + par.grid_zm[t,iz])#*0.0000001 # resources
    
            # search over optimal total consumption, and store (OPPOSITE) of value
            argsm=(Mm,Em[0,iz,:],*pars,par.β,par.grid_Am)                
            cm[0,iz,iA], vm[0,iz,iA] = usr.optimizer(lambda *argsm:-value_of_choice_single(*argsm),1.0e-8, Mm-1.0e-8,args=argsm)            
                
    sol.Vw_single[t,:,:,:]=-vw.copy();sol.Vm_single[t,:,:,:]=-vm.copy()
    sol.Cw_tot_single[t,:,:,:] = cw.copy() ; sol.Cm_tot_single[t,:,:,:] = cm.copy()
    
    
@njit
def value_of_choice_single(C_tot,M,EV_next,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,β,grid_A):
   
    # flow-utility
    C_priv, C_pub =  usr.intraperiod_allocation_single(C_tot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb)
    Util = usr.util(C_priv,C_pub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb)
    
    # continuation value
    A = M - C_tot
    
    EVnext = linear_interp.interp_1d(grid_A,EV_next,A)
    
    # return discounted sum
    return Util + β*EVnext



###############################################################################
# COUPLE'S FUNCTIONS
###############################################################################

@njit
def intraperiod_allocation(C_tot,num_Ctot,grid_Ctot,pre_Ctot_Cw_priv,pre_Ctot_Cm_priv):

    # interpolate pre-computed solution
    j1 = linear_interp.binary_search(0,num_Ctot,grid_Ctot,C_tot)
    Cw_priv = linear_interp_1d._interp_1d(grid_Ctot,pre_Ctot_Cw_priv,C_tot,j1)
    Cm_priv = linear_interp_1d._interp_1d(grid_Ctot,pre_Ctot_Cm_priv,C_tot,j1)
    C_pub = C_tot - Cw_priv - Cm_priv 

    return Cw_priv, Cm_priv, C_pub

@njit(parallel=parallel) 
def solve_remain_couple(par,sol,t): 
 
    #Integration if not last period
    if t<(par.T-1): pEVw,pEVm,nEVw,nEVm = integrate_couple(par,sol,t)

    # initialize 
    remain_Vw,remain_Vm,p_remain_Vw,p_remain_Vm,n_remain_Vw,n_remain_Vm,p_C_tot,n_C_tot, remain_wlp\
       =np.zeros((9,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)) 
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)  
   
    #parameters: useful to unpack this to improve speed 
    couple=1.0;ishome=1.0#;k=par.interp
    pars1=(par.grid_love,par.num_Ctot,par.grid_Ctot, par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb,couple, 
           par.T,par.grid_A,par.grid_weight_love,par.β,par.num_shock_love,par.max_A,par.num_A) 
     
    pars2=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb) 
     
    #pen=1.0 if t<par.Tr else 1e-6
     
    for iL in prange(par.num_love): 
         
        #Variables defined in advance to improve speed 
        Vw_plus_vec,Vm_plus_vec=np.ones(par.num_shock_love)+np.nan,np.ones(par.num_shock_love)+np.nan 
        love_next_vec = par.grid_love[iL] + par.grid_shock_love 
        savings_vec = np.ones(par.num_shock_love) 
        
        for iz in range(par.num_z):  
            for ih in range(par.num_h):
                for iA in range(par.num_A): 
                    for iP in range(par.num_power):  
                        
                        idx=(ih,iz,iP,iL,iA)#;izw=iz//par.num_zm;izm=iz%par.num_zw# indexes 
                        # if (t==12) & (iP==1) & (iz==4) & (iA==49):
                        #     import matplotlib.pyplot as plt
                        #     remain_Vm[idx,1,1]=3.0          
                        # continuation values 
                        if t==(par.T-1):#last period 
                      
                            n_C_tot[idx] = usr.resources_couple(par,t,ih,iz,par.grid_A[iA])[0]# usr.resources_couple(t,par.grid_A[iA],izw,izm,ih,par,wlp=0)#usr.resources_couple(t,par.Tr,par.grid_A[iA],par.grid_zw[t,izw],0.0,par.grid_zm[t,izm],par.R) #No savings, it's the last period 
                             
                            # current utility from consumption allocation 
                            remain_Cw_priv, remain_Cm_priv, remain_C_pub =\
                                intraperiod_allocation(n_C_tot[idx],par.num_Ctot,par.grid_Ctot,sol.pre_Ctot_Cw_priv[0,iP],sol.pre_Ctot_Cm_priv[0,iP]) 
                            remain_Vw[idx] = usr.util(remain_Cw_priv,remain_C_pub,*pars2,par.grid_love[iL],couple,ishome) 
                            remain_Vm[idx] = usr.util(remain_Cm_priv,remain_C_pub,*pars2,par.grid_love[iL],couple,ishome) 
                            remain_wlp[idx]=0.0;p_remain_Vm[idx]=p_remain_Vw[idx]=-1e10;n_remain_Vw[idx],n_remain_Vm[idx]=remain_Vw[idx],remain_Vm[idx]
                        else:#periods before the last 
                                     
                            coeffsWp = prefilter(((0.0,par.max_A,par.num_A),), pEVw[ih,iz,iP,iL,:],k=3) 
                            coeffsMp = prefilter(((0.0,par.max_A,par.num_A),), pEVm[ih,iz,iP,iL,:],k=3)
                            
                            coeffsWn = prefilter(((0.0,par.max_A,par.num_A),), nEVw[ih,iz,iP,iL,:],k=3) 
                            coeffsMn = prefilter(((0.0,par.max_A,par.num_A),), nEVm[ih,iz,iP,iL,:],k=3)
                             
                            def M_resources(wlp): return usr.resources_couple(par,t,ih,iz,par.grid_A[iA])[wlp]#usr.resources_couple(t,par.grid_A[iA],izw,izm,ih,par,wlp=wlp)
                            
                            #first find optimal total consumption 
                            p_args=(iL,par.grid_power[iP],pEVw[iz,iP,iL,:],pEVm[iz,iP,iL,:],coeffsWp,coeffsMp, 
                                  sol.pre_Ctot_Cw_priv[1,iP],sol.pre_Ctot_Cm_priv[1,iP],Vw_plus_vec,Vm_plus_vec,*pars1,love_next_vec,savings_vec)  
                            
                            n_args=(iL,par.grid_power[iP],nEVw[iz,iP,iL,:],nEVm[iz,iP,iL,:],coeffsWn,coeffsMn, 
                                  sol.pre_Ctot_Cw_priv[0,iP],sol.pre_Ctot_Cm_priv[0,iP],Vw_plus_vec,Vm_plus_vec,*pars1,love_next_vec,savings_vec)  
                            
                            def obj(x,t,M_resources,ishom,*args):#function to minimize (= maximize value of choice) 
                                return - value_of_choice_couple(x,t,M_resources,*args,ishom)[0]  
                            
                            p_C_tot[idx]=usr.optimizer(obj,1.0e-6, M_resources(1) - 1.0e-6,args=(t,M_resources(1),0.0,*p_args))[0]
                            n_C_tot[idx]=usr.optimizer(obj,1.0e-6, M_resources(0) - 1.0e-6,args=(t,M_resources(0),1.0,*n_args))[0] #if t<par.Tr else 1e-6 
         
                            # current utility from consumption allocation 
                            p_v_couple, p_remain_Vw[idx], p_remain_Vm[idx] = value_of_choice_couple(p_C_tot[idx],t,M_resources(1),*p_args,0.0)
                            n_v_couple, n_remain_Vw[idx], n_remain_Vm[idx] = value_of_choice_couple(n_C_tot[idx],t,M_resources(0),*n_args,1.0)
                            if (t>=par.Tr):p_v_couple=p_remain_Vw[idx]=p_remain_Vm[idx]=-1e10 # before retirement no labor participation
                            
                            # get the probabilit of working, baed on couple utility choices
                            c=np.maximum(p_v_couple/par.σ,n_v_couple/par.σ)
                            v_couple=par.σ*(c+np.log(np.exp(p_v_couple/par.σ-c)+np.exp(n_v_couple/par.σ-c)))
                            remain_wlp[idx]=np.exp(p_v_couple/par.σ-v_couple/par.σ) # probability of optimal labor supply
                            
                            # now the value of making the choice see Shepard (2019), page 11
                            Δp=p_remain_Vw[idx]-p_remain_Vm[idx];Δn=n_remain_Vw[idx]-n_remain_Vm[idx]
                            remain_Vw[idx]=v_couple+(1.0-par.grid_power[iP])*(remain_wlp[idx]*Δp+(1.0-remain_wlp[idx])*Δn)
                            remain_Vm[idx]=v_couple+(-par.grid_power[iP])   *(remain_wlp[idx]*Δp+(1.0-remain_wlp[idx])*Δn)
                            
                    check_participation_constraints(remain_Vw,remain_Vm,p_remain_Vw,p_remain_Vm,n_remain_Vw,n_remain_Vm,p_C_tot,n_C_tot, remain_wlp,par,sol,t,iL,iA,iz,ih)      
    # return objects 
    return (remain_Vw,remain_Vm,p_remain_Vw,p_remain_Vm,n_remain_Vw,n_remain_Vm,p_C_tot,n_C_tot, remain_wlp)


@njit(parallel=parallel)
def integrate_couple(par,sol,t): 
     
    EVw,EVm,pEVw,pEVm,nEVw,nEVm,tEVw,tEVm,EVdw,EVdm=np.ones((10,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)) 
    aw,am,adw,adm = np.ones((4,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A,par.num_shock_love)) 
    love_next_vec=np.ones((par.num_love,par.num_shock_love)) 
     
    for iL in range(par.num_love):love_next_vec[iL,:] = par.grid_love[iL] + par.grid_shock_love 
     
    #Integrate income shocks
    for iL in prange(par.num_love): 
        for ih in range(par.num_h):     
            for iP in range(par.num_power):     
                for iA in range(par.num_A): 
                    id_i=(ih,slice(None),iP,iL,iA);id_ii=(t+1,ih,slice(None),iP,iL,iA)
                    tEVw[id_i] =  sol.Vw_couple[id_ii].flatten() @ par.Π[t] 
                    tEVm[id_i] =  sol.Vm_couple[id_ii].flatten() @ par.Π[t] 

    #Integrate love shocks
    for iL in prange(par.num_love): 
        for ih in range(par.num_h):  
            for iP in range(par.num_power):     
                for iA in range(par.num_A): 
                    for iz in range(par.num_z): 
                        id_i=(ih,iz,iP,slice(None),iA);id_ii=(ih,iz,iP,iL,iA)
                        linear_interp.interp_1d_vec(par.grid_love,tEVw[id_i],love_next_vec[iL,:],aw[id_ii]) 
                        linear_interp.interp_1d_vec(par.grid_love,tEVm[id_i],love_next_vec[iL,:],am[id_ii])
    
                        EVw[id_ii] =  aw[id_ii].flatten() @ par.grid_weight_love 
                        EVm[id_ii] =  am[id_ii].flatten() @ par.grid_weight_love 
           
    #Integrate human capital transitions  
    for iL in prange(par.num_love): 
        for ih in range(par.num_h):
            for iP in range(par.num_power):     
                for iA in range(par.num_A): 
                     for iz in range(par.num_z):      
                         idx=(slice(None),iz,iP,iL,iA);idz=(ih,iz,iP,iL,iA)
                     
                         pEVw[idz]=EVw[idx]@par.Πh_p[ih,:]
                         nEVw[idz]=EVw[idx]@par.Πh_n[ih,:]
                         pEVm[idz]=EVm[idx]@par.Πh_p[ih,:]
                         nEVm[idz]=EVm[idx]@par.Πh_n[ih,:]
                        
    return pEVw,pEVm,nEVw,nEVm



@njit  
def value_of_choice_couple(Ctot,tt,M_resources,iL,power,Eaw,Eam,coeffsW,coeffsM,pre_Ctot_Cw_priviP,  
        pre_Ctot_Cm_priviP,Vw_plus_vec,Vm_plus_vec, grid_love,num_Ctot,grid_Ctot,  
        ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,couple,
        T,grid_A,grid_weight_love,β,num_shock_love,max_A,num_A,love_next_vec,savings_vec,ishom):  
  
    pars= (ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,grid_love[iL],couple,ishom)  
      
    # current utility from consumption allocation     
    Cw_priv, Cm_priv, C_pub = intraperiod_allocation(Ctot,num_Ctot,grid_Ctot,pre_Ctot_Cw_priviP,pre_Ctot_Cm_priviP)  
    Vw = usr.util(Cw_priv,C_pub,*pars)  
    Vm = usr.util(Cm_priv,C_pub,*pars)  
 
    point=np.array([M_resources - Ctot]) 
    grid=((0.0,max_A,num_A),)
    EVw_plus=eval_spline(grid,coeffsW,point, order=3, extrap_mode="linear", diff="None") 
    EVm_plus=eval_spline(grid,coeffsM,point, order=3, extrap_mode="linear", diff="None")  
    
    #EVw_plus=linear_interp.interp_1d(grid_A, Eaw, M_resources - Ctot)  
    #EVm_plus=linear_interp.interp_1d(grid_A, Eam, M_resources - Ctot)  

    Vw += β*EVw_plus  
    Vm += β*EVm_plus  
       
    return power*Vw + (1.0-power)*Vm, Vw,Vm

@njit
def check_participation_constraints(Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_C_tot,n_C_tot,wlp,par,sol,t,iL,iA,iz,ih):
                 
    # check the participation constraints
    idx_s_w = (t,ih,iz//par.num_zm,iA);idx_s_m = (t,0,iz%par.num_zw,iA)
    
    list_couple = (sol.Vw_couple, sol.Vm_couple)       #couple        list
    list_raw    = (Vw[ih,iz,:,iL,iA],Vm[ih,iz,:,iL,iA])#remain-couple list
    list_single = (sol.Vw_single,sol.Vm_single)        #single        list
    iswomen     = (True,False)                         #iswomen? in   list
    
    # surplus of marriage, then its min and max given states
    Sw = Vw[ih,iz,:,iL,iA] - sol.Vw_single[idx_s_w] 
    Sm = Vm[ih,iz,:,iL,iA] - sol.Vm_single[idx_s_m] 
    min_Sw = np.min(Sw);min_Sm = np.min(Sm)
    max_Sw = np.max(Sw);max_Sm = np.max(Sm) 

    ######################
    # rebargaining below
    #####################
    
    #1) all iP values are consistent with marriage
    if (min_Sw >= 0.0) & (min_Sm >= 0.0): 
        for iP in range(par.num_power):

            idx = (t,ih,iz,iP,iL,iA)
            for i,key in enumerate(list_couple): key[idx] = list_raw[i][iP]
            sol.power[idx] = par.grid_power[iP]
            
    #2) no iP values consistent with marriage
    elif (max_Sw < 0.0) | (max_Sm < 0.0): 
        for iP in range(par.num_power):

            idx = (t,ih,iz,iP,iL,iA)
            for i,key in enumerate(list_couple):
                key[idx]=list_single[i][idx_s_w] if iswomen[i] else list_single[i][idx_s_m]                
            sol.power[idx] = -1.0       
            
    #3) some iP are (invidivually) consistent with marriage
    else: 
    
        # find lowest (highest) value with positive surplus for women (men)
        Low_w = 1 #0 # in case there is no crossing, this will be the correct value
        Low_m = par.num_power-1-1 #par.num_power-1 # in case there is no crossing, this will be the correct value
        for iP in range(par.num_power-1):
            if (Sw[iP]<0) & (Sw[iP+1]>=0): Low_w = iP+1                            
            if (Sm[iP]>=0) & (Sm[iP+1]<0): Low_m = iP
                
        # interpolate the surplus of each member at indifference points
        power_at_zero_w, Sm_at_zero_w = interp_barg(par,Low_w-1,Sw,Sm)
        power_at_zero_m, Sw_at_zero_m = interp_barg(par,Low_m  ,Sm,Sw)
        
        # update the outcomes
        for iP in range(par.num_power):
            idx = (t,ih,iz,iP,iL,iA)
    
            # 3.1) woman wants to leave
            if iP<Low_w: 
                if Sm_at_zero_w > 0: # man happy to shift some bargaining power

                    for i,key in enumerate(list_couple):
                        if iP==0: key[idx] = linear_interp_1d._interp_1d(par.grid_power,list_raw[i],power_at_zero_w,Low_w-1) 
                        else:     key[idx] = list_couple[i][(t,ih,iz,0,iL,iA)]; # re-use that the interpolated values are identical                           

                    sol.power[idx] = power_at_zero_w
                                       
                else: # divorce

                    for i,key in enumerate(list_couple):  
                        if iswomen[i]: key[idx] = list_single[i][idx_s_w]
                        else:          key[idx] = list_single[i][idx_s_m]

                    sol.power[idx] = -1.0
                                    
            # 3.2) man wants to leave
            elif iP>Low_m: 
                if Sw_at_zero_m > 0: # woman happy to shift some bargaining power
                    
                    for i,key in enumerate(list_couple):
                        if (iP==(Low_m+1)): key[idx] = linear_interp_1d._interp_1d(par.grid_power,list_raw[i],power_at_zero_m,Low_m)                             
                        else: key[idx] = list_couple[i][(t,ih,iz,Low_m+1,iL,iA)]; # re-use that the interpolated values are identical
                            
                    sol.power[idx] = power_at_zero_m
                                      
                else: # divorce

                    for i,key in enumerate(list_couple):
                        key[idx] = list_single[i][idx_s_w] if iswomen[i] else list_single[i][idx_s_m] 
                    sol.power[idx] = -1.0
                   
            # 3.3) no-one wants to leave
            else:

                for i,key in enumerate(list_couple): key[idx] = list_raw[i][iP]
                sol.power[idx] = par.grid_power[iP]
                
                
@njit
def interp_barg(par,id,Si,So):#i: individual, o: spouse
    
    denom = (par.grid_power[id+1] - par.grid_power[id])
    ratio_i = (Si[id+1] - Si[id])/denom
    ratio_o = (So[id+1] - So[id])/denom
    
    power_at_zero = par.grid_power[id] - Si[id]/ratio_i                     
    So_at_zero=So[id]+ratio_o*(power_at_zero-par.grid_power[id])
        
    return power_at_zero, So_at_zero