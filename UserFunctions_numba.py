from numba import njit
from numba_stats import norm
import numpy as np
from consav import linear_interp
from numba import config 
from consav.grids import nonlinspace
import setup
from scipy.integrate import quad

#general configuratiion and glabal variables  (common across files)
config.DISABLE_JIT = setup.nojit;parallel=setup.parallel;cache=setup.cache
woman=setup.woman;man=setup.man

############################
# User-specified functions #
############################
@njit(cache=cache)
def home_good(x,θ,λ,tb,couple=0.0,ishom=0.0):
    home_time=(2*tb+ishom*(1-tb)) if couple else tb
    return (θ*x**λ+(1.0-θ)*home_time**λ)**(1.0/λ)

@njit(cache=cache)
def util(c_priv,c_pub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=0.0,ishom=0.0,γ=0.0):
    homegood=home_good(c_pub,θ,λ,tb,couple=couple,ishom=ishom)
    return ((α1*c_priv**ϕ1 + α2*homegood**ϕ1)**ϕ2)/(1.0-ρ)+love-γ#-couple*ishom*1000 

 
@njit(cache=cache) 
def resources_couple(par,t,ih,iz,assets):     
     
    izw=iz//par.num_zm;izm=iz%par.num_zw 
     
    #resources depending on employment 
    res_not_work = par.R*assets + np.exp(np.log(par.grid_zw[t,izw])+par.grid_h[ih])*par.grid_wlp[0] + par.grid_zm[t,izm]  
    res_work     = par.R*assets + np.exp(np.log(par.grid_zw[t,izw])+par.grid_h[ih])*par.grid_wlp[1] + par.grid_zm[t,izm]  
     
    # change resources if retired: women should not work!  
    if t>=par.Tr: return res_not_work, res_not_work  
    else:         return res_not_work, res_work  
    
@njit(cache=cache) 
def income_single(par,t,ih,iz,women=True):
    
    iz_i=iz//par.num_zm if women else iz%par.num_zw
    
    labor_income =  par.grid_zw[t,iz_i] if women else par.grid_zm[t,iz_i]#without HC!
    if (women) & (t<par.Tr): return np.exp(np.log(labor_income) + par.grid_h[ih])
    else                   : return labor_income

@njit(cache=cache)
def couple_util(Cpriv,Ctot,power,ishom,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb):#function to minimize
    """
        Couple's utility given private (Cpriv np.array(float,float)) 
        and total consumption Ctot (float). Note that love does
        not matter here, as this fun is used for intra-period 
        allocation of private and home consumption
    """
    Cpub=Ctot-np.sum(Cpriv)
    Vw=util(Cpriv[0],Cpub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=True,ishom=ishom,γ=0.0)
    Vm=util(Cpriv[1],Cpub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=True,ishom=ishom,γ=0.0)
    
    return np.array([power*Vw +(1.0-power)*Vm, Vw, Vm])


@njit(cache=cache)
def marg_util(C_tot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb):
    
    share = 1.0/(1.0 + (α2/α1)**(1.0/(1.0-ϕ1)))
    constant = α1*share**ϕ1+α2*(1.0-share)**ϕ1
    return ϕ1*C_tot**((1.0-ρ)*ϕ1 -1.0)*constant**(1.0 - ρ)
    
    
    
@njit(cache=cache)
def couple_time_utility(Ctot,par,sol,iP,part,love,pars2,γ):
    """
        Couple's utility given total consumption Ctot (float)
    """    
    p_Cw_priv, p_Cm_priv, p_C_pub =\
        intraperiod_allocation(Ctot,par.grid_Ctot,sol.pre_Ctot_Cw_priv[part,iP],sol.pre_Ctot_Cm_priv[part,iP]) 
        
    vw_new = util(p_Cw_priv,p_C_pub,*pars2,love[0],True,1.0-par.grid_wlp[part],γ)                       
    vm_new = util(p_Cm_priv,p_C_pub,*pars2,love[1],True,1.0-par.grid_wlp[part],γ)
     
    return vw_new, vm_new

@njit(cache=cache)
def single_time_util(Ctot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=0.0,ishom=0.0,γ=0.0):
    
    c_priv,c_pub = intraperiod_allocation_single(Ctot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb)
    
    return util(c_priv,c_pub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love,couple,ishom,γ)
    
@njit(cache=cache)
def intraperiod_allocation(C_tot,grid_Ctot,pre_Ctot_Cw_priv,pre_Ctot_Cm_priv):
 
    #if vector: # interpolate pre-computed solution if C_tot is vector      
    lenn=1 if np.isscalar(C_tot) else len(C_tot)
    Cw_priv,Cm_priv=np.ones((2,lenn))    
    linear_interp.interp_1d_vec(grid_Ctot,pre_Ctot_Cw_priv,C_tot,Cw_priv)
    linear_interp.interp_1d_vec(grid_Ctot,pre_Ctot_Cm_priv,C_tot,Cm_priv)

    return Cw_priv, Cm_priv, C_tot - Cw_priv - Cm_priv #returns numpy arrays
        

@njit(cache=cache)
def intraperiod_allocation_single(C_tot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb):
    
    #find private and public expenditure to max util
    args=(ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb)
    C_priv = optimizer(lambda x,y,args:-util(x,y-x,*args),1.0e-6, C_tot - 1.0e-6,args=(C_tot,args))[0]
    
    return C_priv,C_tot - C_priv#=C_pub

def labor_income(t0,t1,t2,t3,T,Tr,sigma_persistent,sigma_init,npts,pension): 
     
 
    X, Pi, Pi0 =rouw_nonst(T,sigma_persistent,sigma_init,npts) 
     
    if sigma_persistent<0.001:     
        for t in range(T):X[t][:]=0.0 
        for t in range(T-1):Pi[t]=np.eye(npts) 
         
    for t in range(Tr-1,T-1): Pi[t][:]=np.eye(npts) 
    for t in range(T):X[t][:]=np.exp(X[t]+t0+t1*t+t2*t**2+t3*t**3) 
    for t in range(Tr,T):X[t][:]=pension*X[Tr-1]
 
    return np.array(X), Pi, Pi0
   


###########################
# Uncertainty below       #
###########################
 
def sd_rw(T,sigma_persistent,sigma_init):
    
    if isinstance(sigma_persistent,np.ndarray):
        return np.sqrt([sigma_init**2 + t*sigma_persistent[t]**2 for t in range(T)])
    else:
        return np.sqrt(sigma_init**2 + np.arange(0,T)*(sigma_persistent**2))
    
def sd_rw_trans(T,sigma_persistent,sigma_init,sigma_transitory):
    return sd_rw(T, sigma_persistent, sigma_init)

    
    
def normcdf_tr(z,nsd=5):
        
        z = np.minimum(z, nsd*np.ones_like(z))
        z = np.maximum(z,-nsd*np.ones_like(z))
            
        pup = norm.cdf(nsd,0.0,1.0)
        pdown = norm.cdf(-nsd,0.0,1.0)
        const = pup - pdown
        
        return (norm.cdf(z,0.0,1.0)-pdown)/const
    
    
def normcdf_ppf(z): return norm.ppf(z,0.0,1.0)       
        
    

def rouw_nonst(T=40,sigma_persistent=0.05,sigma_init=0.2,npts=10):
   
    sd_z = sd_rw(T,sigma_persistent,sigma_init)
    sd_z0 = np.array([sd_z[t]-sigma_init for t in range(T)])
    
    
    Pi = list();Pi0 = list();X = list()
    
    
    for t in range(0,T):
        nsd = np.sqrt(npts-1)
        X = X + [np.linspace(-nsd*sd_z[t],nsd*sd_z[t],num=npts)]
        
        if t >= 1: Pi = Pi +   [rouw_nonst_one(sd_z[t-1],sd_z[t] ,npts).T]
        if t >= 1: Pi0 = Pi0 + [rouw_nonst_one(sd_z0[t-1],sd_z0[t],npts).T]
       
     
    
    return X, Pi, Pi0


def rouw_nonst_one(sd0,sd1,npts):
   
    # this generates one-period Rouwenhorst transition matrix
    assert(npts>=2)
    pi0 = 0.5*(1+(sd0/sd1))
    Pi = np.array([[pi0,1-pi0],[1-pi0,pi0]])
    assert(pi0<1)
    assert(pi0>0)
    for n in range(3,npts+1):
        A = np.zeros([n,n])
        A[0:(n-1),0:(n-1)] = Pi
        B = np.zeros([n,n])
        B[0:(n-1),1:n] = Pi
        C = np.zeros([n,n])
        C[1:n,1:n] = Pi
        D = np.zeros([n,n])
        D[1:n,0:(n-1)] = Pi
        Pi = pi0*A + (1-pi0)*B + pi0*C + (1-pi0)*D
        Pi[1:n-1] = 0.5*Pi[1:n-1]
        
        assert(np.all(np.abs(np.sum(Pi,axis=1)-1)<1e-5 ))
    
    return Pi


# def normcdf_tr(z,nsd=5):
        
#         z = np.minimum(z, nsd*np.ones_like(z))
#         z = np.maximum(z,-nsd*np.ones_like(z))
            
#         pup = norm.cdf(nsd,0.0,1.0)
#         pdown = norm.cdf(-nsd,0.0,1.0)
#         const = pup - pdown
        
#         return (norm.cdf(z,0.0,1.0)-pdown)/const
    
    
# def normcdf_ppf(z): return norm.ppf(z,0.0,1.0)       
        
      
def addaco_nonst(T=40,sigma_persistent=0.05,sigma_init=0.2,npts=50,sd_z=None,intt=False):
  
    # start with creating list of points
    if sd_z is None:  sd_z = sd_rw(T,sigma_persistent,sigma_init)
        
    Int = list()
    X = list()
    Pi = list()


    #Probabilities per period
    Pr=normcdf_ppf((np.cumsum(np.ones(npts+1))-1)/npts)
    
    #Create interval limits
    for t in range(0,T):Int = Int + [Pr*sd_z[t]]
        
    
    #Create gridpoints
    for t in range(T):
        line=np.zeros(npts)
        for i in range(npts):
            line[i]= sd_z[t]*npts*(norm.pdf(Int[t][i]  /sd_z[t],0.0,1.0)-\
                                   norm.pdf(Int[t][i+1]/sd_z[t],0.0,1.0))
            
        X = X + [line]

    def integrand(x,e,e1,sd,sds):
        return np.exp(-(x**2)/(2*sd**2))*(norm.cdf((e1-x)/sds,0.0,1.0)-\
                                          norm.cdf((e- x)/sds,0.0,1.0))
            
            
    #Fill probabilities
    for t in range(1,T):
        Pi_here = np.zeros([npts,npts])
        for i in range(npts):
            for jj in range(npts):
                
                Pi_here[i,jj]=npts/np.sqrt(2*np.pi*sd_z[t-1]**2)\
                    *quad(integrand,Int[t-1][i],Int[t-1][i+1],
                     args=(Int[t][jj],Int[t][jj+1],sd_z[t],sigma_persistent))[0]
                
                if T==2:
                    Pi_here[i,jj]= norm.cdf(Int[t][jj+1],0.0,1.0)-\
                                   norm.cdf(Int[t][jj ],0.0,1.0)
                                   
            #Adjust probabilities to get exactly 1: the integral is an approximation
            Pi_here[i,:]=Pi_here[i,:]/np.sum(Pi_here[i,:])
                
        Pi = Pi + [Pi_here.T]
        
  
    #if intt:return X, Pi, Int
    return X, Pi


@njit(fastmath=True)
def mc_simulate(statein,Piin,shocks):
    """This simulates transition one period ahead for a Markov chain
    
    Args: 
        Statein: scalar giving the initial state
        Piin: transition matrix n*n [post,initial]
        shocks: scalar in [0,1]
    
    """ 
    return  np.sum(np.cumsum(Piin[:,statein])<shocks)

@njit(fastmath=True)
def rescale_matrix(Π,tol=1e-4):
    """Rescale transition matrix by eliminating close to zero elements
    
    Args:
        Π: n*n square tranistion matrix
        tol: elements of Π below tol are set to 0
    
    """
    
    zeros = Π < tol
    for i in range(len(Π)):
        
        Π[zeros[:,i],i] = 0.0
        Π[:,i] = Π[:,i] / np.sum(Π[:,i])
        
    return Π 
    
    
##########################
# Other routines below
##########################

@njit
def deriv(x,f,ϵ=1e-8):
    """
    Create derivative for array f defined on the x array x
    """
    
    forward,backward=np.empty((2,len(x)))
    linear_interp.interp_1d_vec(x,f,x+ϵ,forward)
    linear_interp.interp_1d_vec(x,f,x-ϵ,backward)
    
    return (forward-backward)/(2*ϵ)
    

def grid_fat_tails(gmin,gmax,gridpoints):
    """Create a grid with fat tail, centered and symmetric around gmin+gmax
    
    Args: 
        gmin (float): min of grid
        gmax(float): max of grids
        gridpoints(int): number of gridpoints (odd number)
    
    """ 
    odd_num = np.mod(gridpoints,2)
    mid=(gmax+gmin)/2.0
    summ=gmin+gmax
    first_part = nonlinspace(gmin,mid,(gridpoints+odd_num)//2,1.3)
    last_part = np.flip(summ - nonlinspace(gmin,mid,(gridpoints-odd_num)//2 + 1,1.3))[1:]
    return np.append(first_part,last_part)

@njit 
def optimizer(obj,a,b,args=(),tol=1e-6): 
    """ golden section search optimizer 
     
    Args: 
 
        obj (callable): 1d function to optimize over 
        a (double): minimum of starting bracket 
        b (double): maximum of starting bracket 
        args (tuple): additional arguments to the objective function 
        tol (double,optional): tolerance 
 
    Returns: 
 
        float,float: optimization result, function at argmin
     
    """ 
     
    inv_phi = (np.sqrt(5) - 1) / 2 # 1/phi                                                                                                                 
    inv_phi_sq = (3 - np.sqrt(5)) / 2 # 1/phi^2      
         
    # a. distance 
    dist = b - a 
    if dist <= tol:  
        return (a+b)/2,  obj((a+b)/2,*args)
 
    # b. number of iterations 
    n = int(np.ceil(np.log(tol/dist)/np.log(inv_phi))) 
 
    # c. potential new mid-points 
    c = a + inv_phi_sq * dist 
    d = a + inv_phi * dist 
    yc = obj(c,*args) 
    yd = obj(d,*args) 
 
    # d. loop 
    for _ in range(n-1): 
        if yc < yd: 
            b = d 
            d = c 
            yd = yc 
            dist = inv_phi*dist 
            c = a + inv_phi_sq * dist 
            yc = obj(c,*args) 
        else: 
            a = c 
            c = d 
            yc = yd 
            dist = inv_phi*dist 
            d = a + inv_phi * dist 
            yd = obj(d,*args) 
 
    # e. return 
    if yc < yd: 
        return (a+d)/2, obj((a+d)/2,*args)
    else: 
        return (c+b)/2, obj((c+b)/2,*args)
    
####################################################################################
# Upper envelop alogorithm - Adapted from Consav to accomodate for couple_decisions
###################################################################################
def create(ufunc):
    """ create upperenvelope function from the utility function ufunc
    
    Args:

        ufunc (callable): utility function with *args (must be decorated with @njit)

    Returns:

        upperenvelope (callable): upperenvelope called as (grid_a,m_vec,c_vec,inv_w_vec,use_inv_w,grid_m,c_ast_vec,v_ast_vec,*args)
    
    """

    @njit
    def upperenvelope(grid_a,m_vec,c_vec,inv_w_vec_w,inv_w_vec_m,power,grid_m,c_ast_vec,v_ast_vec_w,v_ast_vec_m,v_ast_vec_c,v_guess_all_division,v_given_division,*args):
        """ upperenvelope function
        
        Args:

            grid_a (numpy.ndarray): input, end-of-period asset vector of length Na
            m_vec (numpy.ndarray): input, cash-on-hand vector from egm of length Na
            c_vec (numpy.ndarray): input, consumption vector from egm of length Na
            inv_w_vec (numpy.ndarray): input, post decision value-of-choice vector from egm of length Na
            grid_m (numpy.ndarray): input, common grid for cash-on-hand of length Nm
            c_ast_vec (numpy.ndarray): input-output, consumption on common grid for cash-on-hand of length Nm, *for any asset division*
            v_ast_vec_w (numpy.ndarray): input-output, women's value-of-choice on common grid for cash-on-hand of length Nm, *for any asset division*
            v_ast_vec_m (numpy.ndarray): input-output,   men's value-of-choice on common grid for cash-on-hand of length Nm, *for any asset division*
            v_ast_vec_c (numpy.ndarray): input-output, coupl's value-of-choice on common grid for cash-on-hand of length Nm, *for any asset division*
            v_guess_all_division  (numpy.ndarray): input-output, *guess*, or *current best* coupl's value-of-choice on common grid for cash-on-hand of length Nm, *for specific asset division*
            v_given_division  (numpy.ndarray):     input-output, coupl's value-of-choice on common grid for cash-on-hand of length Nm, *for specific asset division*
            *args: additional arguments to the utility function
                    
        """

        # for given m_vec, c_vec and w_vec (coming from grid_a)
        # find the optimal consumption choices (c_ast_vec) at the common grid (grid_m) 
        # using the upper envelope + also value the implied values-of-choice (v_ast_vec)

        Na = grid_a.size 
        Nm = grid_m.size 
 
        #c_ast_vec[:] = 0 
        v_ast_vec = -np.inf*np.ones(c_ast_vec.shape) 
 
        # constraint 
        # the constraint is binding if the common m is smaller 
        # than the smallest m implied by EGM step (m_vec[0]) 
 
        im = 0  
        while im < Nm and grid_m[im] <= m_vec[0]:  
              
            # a. value of choice, consume cash on hand  
            u_w,u_m = ufunc(grid_m[im:im+1],*args)   
            temp_v_ast_vec_w = u_w[0] + inv_w_vec_w[0]   
            temp_v_ast_vec_m = u_m[0] + inv_w_vec_m[0]   
   
            # update the asset *division specific* couple's value function  
            v_ast_vec[im] = power*temp_v_ast_vec_w + (1.0-power)*temp_v_ast_vec_m  
            v_given_division[im] =  v_ast_vec[im] 
             
            # if better couple's U with current util, update  
            # *general* (not division specific) value functions and consumption  
            if (v_given_division[im]>=v_guess_all_division[im]):  
                  
                # b. store results  
                c_ast_vec[im] = grid_m[im]  
                v_ast_vec_w[im] = temp_v_ast_vec_w  
                v_ast_vec_m[im] = temp_v_ast_vec_m                      
                v_ast_vec_c[im] = v_given_division[im] 
                v_guess_all_division[im] = v_given_division[im] 
                  
            im += 1  
             
        # upper envellope 
        # apply the upper envelope algorithm 
         
        for ia in range(Na-1): 
 
            # a. a inteval and w slope 
            a_low  = grid_a[ia] 
            a_high = grid_a[ia+1] 
             
            inv_w_low_w  = inv_w_vec_w[ia] 
            inv_w_high_w = inv_w_vec_w[ia+1] 
             
            inv_w_low_m  = inv_w_vec_m[ia] 
            inv_w_high_m = inv_w_vec_m[ia+1] 
 
            if a_low > a_high: 
                continue 
 
            inv_w_slope_w = (inv_w_high_w-inv_w_low_w)/(a_high-a_low) 
            inv_w_slope_m = (inv_w_high_m-inv_w_low_m)/(a_high-a_low) 
             
            # b. m inteval and c slope 
            m_low  = m_vec[ia] 
            m_high = m_vec[ia+1] 
 
            c_low  = c_vec[ia] 
            c_high = c_vec[ia+1] 
 
            c_slope = (c_high-c_low)/(m_high-m_low) 
 
            # c. loop through common grid 
            for im in range(Nm): 
 
                # i. current m 
                m = grid_m[im] 
 
                # ii. interpolate? 
                interp = (m >= m_low) and (m <= m_high)             
                extrap_above = ia == Na-2 and m > m_vec[Na-1] 
 
                # iii. interpolation (or extrapolation) 
                if interp or extrap_above: 
 
                    # o. implied guess 
                    c_guess = np.array([c_low + c_slope * (m - m_low)]) 
                    a_guess = m - c_guess[0] 
 
                    # oo. implied post-decision value function 
                    inv_w = inv_w_low_w + inv_w_slope_w * (a_guess - a_low)      
                    inv_m = inv_w_low_m + inv_w_slope_m * (a_guess - a_low)        
 
                    # ooo. value-of-choice 
                    u_w,u_m = ufunc(c_guess,*args)    
                    v_guess_w = u_w[0] + inv_w 
                    v_guess_m = u_m[0] + inv_m 
 
                    v_guess=power*v_guess_w+(1.0-power)*v_guess_m 
                     
                    # oooo. update 
                    if (v_guess >  v_ast_vec[im]):  
                        v_given_division[im] = v_guess 
                        v_ast_vec[im] = v_guess 
                        if (v_guess>=v_guess_all_division[im]): 
                            
                            c_ast_vec[im] = c_guess[0] 
                             
                            # update utility for the couple 
                            v_ast_vec_w[im] = v_guess_w 
                            v_ast_vec_m[im] = v_guess_m                       
                            v_ast_vec_c[im]=  v_guess 
                            v_guess_all_division[im]      =  v_guess 
     
    return upperenvelope