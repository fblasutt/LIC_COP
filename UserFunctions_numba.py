from numba import njit,prange
from numba_stats import norm
from scipy.integrate import quad
import numpy as np#import autograd.numpy as np#
from consav import linear_interp, linear_interp_1d,quadrature
from numba import config 
config.DISABLE_JIT = False
cache=True
# set gender indication as globals
woman = 1
man = 2

############################
# User-specified functions #
############################
@njit(cache=cache)
def home_good(x,θ,λ,tb,couple=0.0,ishom=0.0):
    home_time=(2*tb+ishom*(1-tb)) if couple else tb
    return (θ*x**λ+(1.0-θ)*home_time**λ)**(1.0/λ) 

@njit(cache=cache)
def util(c_priv,c_pub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=0.0,ishom=0.0):
    homegood=home_good(c_pub,θ,λ,tb,couple=couple,ishom=ishom)
    return np.log(c_priv)-1.1*(1.0-ishom)#((α1*c_priv**ϕ1 + α2*homegood**ϕ1)**ϕ2)/(1.0-ρ)++ love 

@njit(cache=cache)
def resources_couple(t,Tr,A,inc_w,wlp,inc_m,R):   
    # change resources if retired: women should not work!
    if t>=Tr: return R*A + inc_w     + inc_m
    else:     return R*A + inc_w*wlp + inc_m

@njit(cache=cache)
def couple_util(x,ct,pw,ishom,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,couple):#function to minimize
    pubc=ct*(1.0-np.sum(x))
    return (pw*util(x[0]*ct,pubc,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,couple,ishom,True) +\
      (1.0-pw)*util(x[1]*ct,pubc,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,couple,ishom,True))
        
@njit(cache=cache)
def couple_util_ind(x,ct,pw,ishom,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,couple):#function to minimize
    pubc=ct*(1.0-np.sum(x))
    Vw=util(x[0]*ct,pubc,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,couple,ishom,True)
    Vm=util(x[1]*ct,pubc,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,couple,ishom,True)
    return Vw, Vm


@njit(cache=cache)
def value_of_choice(cons,par,sol,iP,EVw_i,EVm_i,part,power,love,pars2):
    
    p_Cw_priv, p_Cm_priv, p_C_pub =\
        intraperiod_allocation(cons,par.num_Ctot,par.grid_Ctot,sol.pre_Ctot_Cw_priv[part,iP],sol.pre_Ctot_Cm_priv[part,iP]) 
        
    #p_Ew=linear_interp.interp_1d(par.grid_A,EVw_i,savings)
    #p_Em=linear_interp.interp_1d(par.grid_A,EVm_i,savings)
     
    vw_new = util(p_Cw_priv,p_C_pub,*pars2,love,True,1.0-par.grid_wlp[part])#+par.β*p_Ew                        
    vm_new = util(p_Cm_priv,p_C_pub,*pars2,love,True,1.0-par.grid_wlp[part])#+par.β*p_Em
     
    return vw_new, vm_new
     
    
@njit(cache=cache)
def intraperiod_allocation(C_tot,num_Ctot,grid_Ctot,pre_Ctot_Cw_priv,pre_Ctot_Cm_priv):

    # interpolate pre-computed solution
    j1 = linear_interp.binary_search(0,num_Ctot,grid_Ctot,C_tot)
    Cw_priv = linear_interp_1d._interp_1d(grid_Ctot,pre_Ctot_Cw_priv,C_tot,j1)
    Cm_priv = linear_interp_1d._interp_1d(grid_Ctot,pre_Ctot_Cm_priv,C_tot,j1)
    C_pub = C_tot - Cw_priv - Cm_priv 

    return Cw_priv, Cm_priv, C_pub
        
@njit(cache=cache)    
def obj_s(x,Ctot,pρ,pϕ1,pϕ2,pα1,pα2,pθ,pλ,ptb):#=-utility of consuming x
    return -util(x,Ctot-x,pρ,pϕ1,pϕ2,pα1,pα2,pθ,pλ,ptb)

@njit(cache=cache)
def intraperiod_allocation_single(C_tot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb):
    
    args=(C_tot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb)
 
    #find private and public expenditure to max util
    C_priv = optimizer(obj_s,1.0e-6, C_tot - 1.0e-6,args=args)[0]
    
    C_pub = C_tot - C_priv
    return C_priv,C_pub

def labor_income(t0,t1,t2,T,Tr,sigma_persistent,sigma_init,npts):
    

    X, Pi =rouw_nonst(T,sigma_persistent,sigma_init,npts)
    
    if sigma_persistent<0.001:    
        for t in range(T-1):X[t][:]=0.0
        for t in range(T-1):Pi[t]=np.eye(npts)
        
    for t in range(T):X[t][:]=0.4#np.exp(X[t]+t0+t1*t+t2*t**2)
    for t in range(Tr,T):X[t][:]=0.1

    return np.array(X), Pi
   
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
        
    

def rouw_nonst(T=40,sigma_persistent=0.05,sigma_init=0.2,npts=10,sd_z=None):
   
    if sd_z is None: sd_z = sd_rw(T,sigma_persistent,sigma_init)
    assert(npts>=2)
    Pi = list()
    X = list()
    
    for t in range(0,T):
        nsd = np.sqrt(npts-1)
        X = X + [np.linspace(-nsd*sd_z[t],nsd*sd_z[t],num=npts)]
        if t >= 1: Pi = Pi + [rouw_nonst_one(sd_z[t-1],sd_z[t],npts).T]
            
    
    return X, Pi


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

@njit(fastmath=True)
def mc_simulate(statein,Piin,shocks):
    """This simulates transition one period ahead for a Markov chain
    
    Args: 
        Statein: scalar giving the initial state
        Piin: transition matrix n*n [post,initial]
        shocks: scalar in [0,1]
    
    """ 
    return  np.sum(np.cumsum(Piin[:,statein])<shocks)


##########################
# Other routines below
##########################
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
# Upper envelop alogorithm - Adapted from COnsav to accomodate for couple_decisions
###################################################################################
def create(ufunc,use_inv_w=False):
    """ create upperenvelope function from the utility function ufunc
    
    Args:

        ufunc (callable): utility function with *args (must be decorated with @njit)

    Returns:

        upperenvelope (callable): upperenvelope called as (grid_a,m_vec,c_vec,inv_w_vec,use_inv_w,grid_m,c_ast_vec,v_ast_vec,*args)
        use_inv_w (bool,optional): assume that the post decision value-of-choice vector is a negative inverse
    
    """

    @njit
    def upperenvelope(grid_a,m_vec,c_vec,inv_w_vec_w,inv_w_vec_m,power,grid_m,c_ast_vec,v_ast_vec_w,v_ast_vec_m,*args):
        """ upperenvelope function
        
        Args:

            grid_a (numpy.ndarray): input, end-of-period asset vector of length Na
            m_vec (numpy.ndarray): input, cash-on-hand vector from egm of length Na
            c_vec (numpy.ndarray): input, consumption vector from egm of length Na
            inv_w_vec (numpy.ndarray): input, post decision value-of-choice vector from egm of length Na
            grid_m (numpy.ndarray): input, common grid for cash-on-hand of length Nm
            c_ast_vec (numpy.ndarray): output, consumption on common grid for cash-on-hand of length Nm
            v_ast_vec (numpy.ndarray): output, value-of-choice on common grid for cash-on-hand of length Nm
            *args: additional arguments to the utility function
                    
        """

        # for given m_vec, c_vec and w_vec (coming from grid_a)
        # find the optimal consumption choices (c_ast_vec) at the common grid (grid_m) 
        # using the upper envelope + also value the implied values-of-choice (v_ast_vec)

        Na = grid_a.size
        Nm = grid_m.size

        c_ast_vec[:] = 0
        v_ast_vec = -np.inf*np.ones(c_ast_vec.shape)

        # constraint
        # the constraint is binding if the common m is smaller
        # than the smallest m implied by EGM step (m_vec[0])

        im = 0
        while im < Nm and grid_m[im] <= m_vec[0]:
            
            # a. consume all
            c_ast_vec[im] = grid_m[im] 

            # b. value of choice
            u_w,u_m = ufunc(c_ast_vec[im],*args)
            if use_inv_w:
                v_ast_vec_w[im] = u_w + (-1.0/inv_w_vec_w[0])
                v_ast_vec_m[im] = u_m + (-1.0/inv_w_vec_m[0])
            else:
                v_ast_vec_w[im] = u_w + inv_w_vec_w[0]
                v_ast_vec_m[im] = u_m + inv_w_vec_m[0]

            v_ast_vec[im] = power*v_ast_vec_w[im] + (1.0-power)*v_ast_vec_m[im]
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
                    c_guess = c_low + c_slope * (m - m_low)
                    a_guess = m - c_guess

                    # oo. implied post-decision value function
                    inv_w = inv_w_low_w + inv_w_slope_w * (a_guess - a_low)     
                    inv_m = inv_w_low_m + inv_w_slope_m * (a_guess - a_low)       

                    # ooo. value-of-choice
                    u_w,u_m = ufunc(c_guess,*args)
                    if use_inv_w:
                        
                        v_guess_w = u_w + (-1/inv_w)
                        v_guess_m = u_m + (-1/inv_m)
                     
                    else:
                        v_guess_w = u_w + inv_w
                        v_guess_m = u_m + inv_m

                    v_guess=power*v_guess_w+(1.0-power)*v_guess_m
                    
                    # oooo. update
                    if v_guess > v_ast_vec[im]:
                        v_ast_vec[im] = v_guess
                        c_ast_vec[im] = c_guess
                        
                        # update utility for the couple
                        v_ast_vec_w[im] = v_guess_w
                        v_ast_vec_m[im] = v_guess_m
    
    return upperenvelope