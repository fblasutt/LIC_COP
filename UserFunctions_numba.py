from numba import njit,prange
from numba_stats import norm
from scipy.integrate import quad
import numpy as np

from numba import config 
config.DISABLE_JIT = False

# set gender indication as globals
woman = 1
man = 2

############################
# User-specified functions #
############################
@njit
def home_good(x,θ,λ,tb,couple=0.0,ishom=0.0):
    home_time=(2*tb+ishom*(1-tb)) if couple else tb
    return (θ*x**λ+(1.0-θ)*home_time**λ)**(1.0/λ)

@njit
def util(c_priv,c_pub,gender,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=0.0,ishom=0.0):
    homegood=home_good(c_pub,θ,λ,tb,couple=couple,ishom=ishom)
    return ((α1*c_priv**ϕ1 + α2*homegood**ϕ2)**(1.0-ρ))/(1.0-ρ)+ love

@njit
def resources_couple(A,inc_w,inc_m,R):
    # resources of the couple
    return R*A + inc_w + inc_m

@njit
def resources_single(A,gender,inc_w,inc_m,R):
    # resources of single individual of gender "gender"
    income = inc_m if gender ==man else inc_w
    return R*A + income


###########################
# Income shocks below     #
###########################

def labor_income(t0,t1,t2,T,sigma_persistent,sigma_init,npts):
    

    X, Pi = addaco_nonst(T,sigma_persistent,sigma_init,npts)
    
    if sigma_persistent<0.001:    
        for t in range(T):X[t][:]=0.0
        for t in range(T):Pi[t]=np.eye(npts)
        
    for t in range(T):X[t][:]=np.exp(X[t]+t0+t1*t+t2*t**2)
        
    return np.array(X), Pi
    
      
def addaco_nonst(T,sigma_persistent,sigma_init,npts):
    #Create 
    
    # start with creating list of points
    sd_z = np.sqrt(sigma_init**2 + np.arange(0,T)*(sigma_persistent**2))
        
    Int = list()
    X = list()
    Pi = list()

    #Probabilities per period
    Pr=norm.ppf(((np.cumsum(np.ones(npts+1))-1)/npts),0.0,1.0)
    
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
        
        
    Pi = Pi + [None] # last matrix is not definednsd    
   
    return X, Pi

# @njit(fastmath=True,parallel=True)
# def mc_simulate(statein,Piin,shocks):
#     """This simulates transition one period ahead for a Markov chain
    
#     Args: 
#         Statein: numpy array of length n giving the initial state
#         Piin:    
#         shocks:
        
#     Equivalent to:
#     for i in prange(len(statein)):
#        stateout[i]= np.sum(np.cumsum(Piin[statein[i],:])<shocks[i])
    
#     """ 
    
#     stateout = np.full(statein.size,Piin.shape[1]-1,dtype=np.int16)
#     for i in prange(len(statein)):
        
#         prob=0.0
#         for j in range(Piin.shape[1]):
#             if prob<shocks[i]:
#                 prob=prob+Piin[statein[i],j]
#             else:             
#                 stateout[i]=j-1
#                 break
    
#     return stateout 


@njit(fastmath=True)
def mc_simulate(statein,Piin,shocks):
    """This simulates transition one period ahead for a Markov chain
    
    Args: 
        Statein: numpy array of length n giving the initial state
        Piin:    
        shocks:
        
    Equivalent to:
    for i in prange(len(statein)):
       stateout[i]= np.sum(np.cumsum(Piin[statein[i],:])<shocks[i])
    
    """ 
    
    # prob=0.0
    # for j in range(Piin.shape[1]):
    #     if prob<shocks:
    #         prob=prob+Piin[statein,j]
    #     else:             
    #         stateout=j-1
    #         break
    
    # return stateout 

    return  np.sum(np.cumsum(Piin[:,statein])<shocks)

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
 
        (float): optimization result 
     
    """ 
     
    inv_phi = (np.sqrt(5) - 1) / 2 # 1/phi                                                                                                                 
    inv_phi_sq = (3 - np.sqrt(5)) / 2 # 1/phi^2      
         
    # a. distance 
    dist = b - a 
    if dist <= tol:  
        return (a+b)/2 
 
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
        return (a+d)/2 
    else: 
        return (c+b)/2 