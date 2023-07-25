
#############################################################################
# Code for class 2.1 of "Inequality, household behavior and the macroeconomy"
#
# Life cycle model of consumption and savings with (or without) borrowing
# and endogenous female labor force participation with or without
# DC-EGM method as in Iskhakov, Jørgensen, Rust, and Schjerning 2017
# *WITH* task shocks
#
# Author: Fabio Blasutto
#
############################################################################

#--------------------------------#
#         Initialization         #
#--------------------------------#

#Load libraries
import time                      #Measure time
import numpy as np               #Load numpy
import matplotlib.pyplot as plt  #Load matplotlib for graphs
import quantecon as qe           #Useful tools for economics
from consav.grids import nonlinspace # grids
from numba import njit,prange
from consav import upperenvelope
#--------------------------------#
#        Parameters              #
#--------------------------------#

# Data structure of state and exogenous variables
class modelState(object):
    def __init__(self,
                 NA=300,         #Gridpoints of assets grid
                 T=20,            #Age at death
                 Tr=13,           #Age at retirement
                 β=1.00,           #Discount factor
                 r=0.00,          #Interest rate
                 amin=0.0,        #Lower bound of assets'grid
                 amax=6.0,       #Higher bound of assets'grid
                 Nag=10000,       #Number of agents to simulate
                 σ=0.0000005,           #Taste shocks
                 σ_z=0.01**0.5,  #Standard dev. of income shock (same for H, W)
                 ρ=0.95,          #Persistence of income shock
                 ϵ=0.0000005,   #Avoid wrong participation because of float #
                 ψ=0.0,           #Utility cost of female labor participation
                 Nz=3,            #Gridpoints for persistent income shock
                 dcegm=True,      #Use DC-EGM method, otherwise egm only
                 diagnostic=True):#Check for non convexities
        
        #Initialize arrays: value functions, policy funtcion for saving,cons..
        #...,female labor force participation + (log) income grid for H and W
        V_p,V_np,V,pr,opt_a,opt_a_np,opt_a_p,opt_c,opt_c_np,opt_c_p,part=np.zeros((11,T,NA,Nz,Nz))
        log_yH,log_yW,yH,yW=np.zeros((4,T,Nz))          
        
        A=np.linspace(amin,amax,NA) #nonlinspace(amin,amax,NA,1.2)#             #Build grid for assets
        
        #Construct the grid for productivity (z) using Rouwenhorst (1995)
        markov_chain=qe.markov.approximation.rouwenhorst(Nz,ρ,σ_z,0.0)
        Π = markov_chain.P                       #Transition matrix for z
        zgrid = markov_chain._state_values       #Grid for z        
           
        #Add shocks to the second dimension of log income array for H    
        for z in range(Nz):log_yH[:,z]+=zgrid[z]
        
        #Add age trend to the array of log income for H and W  
        skew=-σ_z**2/2#This is to correct avg. income when σ_z increases         
        for t in range(T):log_yH[t,:]+=skew-0.75+.042*t-.00165*t**2+.000018*t**3
        
        #Create income from log income. Also create a gender wage gap of 80%
        yH=np.exp(log_yH)            #Create income of H
        yW=0.8*yH                    #W's income is 80% of H's inocme
        
        #Adjus at retirement: H and W pensions  of value 0.45 each stored in...
        #yH: trick to avoid participation in labor market of W at retirement
        yH[:,:],yW[:,:]=0.4,0.3#.045+.045,0.0
        yH[Tr:,:],yW[Tr:,:]=0.1,0.#.045+.045,0.0
        
        
        #grid of marginal utilities
        grid_Ctot=nonlinspace(1.0e-6,amax*2,200,1.1)#np.linspace(1e-6,amax*2,NA)
        
        grid_marg_u=np.ones(grid_Ctot.shape)
        ϵ=grid_Ctot[0]/2
        for i,Ctot in enumerate(grid_Ctot):
            backward=np.log(Ctot-ϵ)
            forward=np.log(Ctot+ϵ)
            grid_marg_u[i]=(forward-backward)/(2*ϵ)
            
        grid_marg_u=0.5/grid_Ctot
        grid_marg_u_for_inv=np.flip(grid_marg_u)
        grid_inv_marg_u = np.flip(grid_Ctot)        
        Vd=np.zeros((T,NA,Nz,Nz))
        
        # Store grids and parameters
        self.NA = NA;self.T = T;self.Tr=Tr;self.A=A;self.β=β;self.Π=Π;self.dcegm=dcegm;
        self.yH = yH;self.yW = yW;self.r = r;self.opt_a=opt_a;self.Nag=Nag;self.σ=σ;
        self.Nz=Nz;self.opt_c=opt_c;self.zgrid=zgrid;self.V=V;self.pr=pr;self.part=part;
        self.ψ=ψ;self.V_p=V_p;self.V_np=V_np;self.σ_z=σ_z ;self.ϵ=ϵ;self.diagnostic=diagnostic;
        self.opt_a_np=opt_a_np;self.opt_a_p=opt_a_p;self.opt_c_np=opt_c_np;self.opt_c_p=opt_c_p
        
        self.grid_Ctot=grid_Ctot;self.grid_marg_u=grid_marg_u;
        self.grid_marg_u_for_inv=grid_marg_u_for_inv;self.grid_inv_marg_u=grid_inv_marg_u;        
        self.Vd=Vd
#--------------------------------#
#     Structure and function     #
#--------------------------------#

# Import all the parameters
st=modelState()          

# Extraoplate

def ext(x,xp,yp):
    
    return yp[-1]+(yp[-1]-yp[-2])/(xp[-1]-xp[-2])*(x-xp[-1])
# Define the utility function
def u(c,st):
    
    u=np.log(np.zeros(c.shape)+1e-6)
    where=(c>0)
    u[where]=np.log(c[where]/2.0)

    
    return u
    
# Define the utility function
@njit
def uu(c):
    
    if c>1e-6:
        return np.log(c/2.0)
    else:
        return np.log(1e-6/2.0)
    

uppere=upperenvelope.create(uu)



#  Function that stores in st the policy functions for (t,izh,izw, ia=0,...,NA)
def pol_func(st,t,izh,izw):                      #max util given age t,shocks izh,izw
                                              
    ce,ae_p,ae_np,c_p,c_np,a_p,a_np,Vp,V_np,EV,EVd=np.zeros((11,st.NA))#initialize 
    
    #Find policy function (optimal savings and consumption) if t==T-1
    if(t>=st.T-1):                               #decisions if last period...        
        st.opt_a_np[st.T-1,:,izh,izw] = np.zeros(st.NA);          #optimal savings
        st.opt_c_np[st.T-1,:,izh,izw] = \
                 st.A*(1+st.r) +st.yH[st.T-1,izh]              #optimal consumption  
        st.V[st.T-1,:,izh,izw]=st.V_p[st.T-1,:,izh,izw]=u(st.opt_c_np[st.T-1,:,izh,izw],st)#get the value function   
        st.Vd[st.T-1,:,izh,izw] = np.interp(st.opt_c_np[st.T-1,:,izh,izw],st.grid_Ctot,st.grid_marg_u)
        st.V_p[st.T-1,:,izh,izw]=-np.inf
        return
      
    #Find policy functions (optimal savings, cons. labor part.) if t<T-1
    for ia in range(st.NA):            #Loop over assets ia
        
        #Obtain expected marginal utility of consumption and utility
        for izw1 in range(st.Nz):
            for izh1 in range(st.Nz):
                
                #Marginal utility
                EVd[ia]+=(1+st.r)*st.β*st.Vd[t+1,ia,izh1,izw1]*st.Π[izh,izh1]*st.Π[izw,izw1] 
                                       
                                       
                #Utility                    
                EV[ia]+=st.V[t+1,ia,izh1,izw1]*st.Π[izh,izh1]*st.Π[izw,izw1]   
                
    #Obtain consumption
    ce=np.interp(EVd,st.grid_marg_u_for_inv,st.grid_inv_marg_u)
        
    #How much assets in t rationalizes ce? Find out using the BC for 2 cases:..
    #1) Participation (_p) and no participation (_np)!
    ae_p=(st.A-st.yH[t,izh]-st.yW[t,izw]+ce)/(1+st.r)#Participation
    ae_np=(st.A-st.yH[t,izh]+ce)/(1+st.r)            #No participation
    
    
    #1) participation                                 
    st.opt_a_p[t,:,izh,izw]=np.interp(st.A, ae_p,st.A)    
    st.opt_c_p[t,:,izh,izw]=(1+st.r)*st.A+st.yH[t,izh]+st.yW[t,izw]-st.opt_a_p[t,:,izh,izw]
    st.V_p[t,:,izh,izw]=u(st.opt_c_p[t,:,izh,izw],st)-st.ψ+st.β*np.interp(st.opt_a_p[t,:,izh,izw],st.A,EV)#+st.β*EVp
       
    #Nonconvexities handled below: apply upper envelop to eliminate FOCs
    #that are not global optimum
    
    #if t==11:st.V_p[t,:,izh,izw,0,0]=3
        
    if 1>0:#(np.min(np.diff(ae_p))<0) & (st.dcegm):
        
        uppere(st.A,ae_p*(1+st.r)+st.yH[t,izh]+st.yW[t,izh],ce,st.β*EV,st.A*(1+st.r)+st.yH[t,izh]+st.yW[t,izh],
                     st.opt_c_p[t,:,izh,izw],st.V_p[t,:,izh,izw])

        st.opt_a_p[t,:,izh,izw]=(1+st.r)*st.A+st.yH[t,izh]+st.yW[t,izw]-st.opt_c_p[t,:,izh,izw]
        st.V_p[t,:,izh,izw]=st.V_p[t,:,izh,izw]-st.ψ
    #2) no participation in the market  (next 3 lines)
    st.opt_a_np[t,:,izh,izw]=np.interp(st.A, ae_np,st.A)
    st.opt_c_np[t,:,izh,izw]=(1+st.r)*st.A+st.yH[t,izh]-st.opt_a_np[t,:,izh,izw]
    st.V_np[t,:,izh,izw]=u(st.opt_c_np[t,:,izh,izw],st)+st.β*np.interp(st.opt_a_np[t,:,izh,izw],st.A,EV)##+st.β*EVnp

    #Nonconvexities handled below: apply upper envelop to eliminate FOCs
    #that are not global optimum
    if 1>0:#(np.min(np.diff(ae_np))<0) & (st.dcegm):
              
        uppere(st.A,ae_np*(1+st.r)+st.yH[t,izh],ce,st.β*EV,st.A*(1+st.r)+st.yH[t,izh],
                    st.opt_c_np[t,:,izh,izw],st.V_np[t,:,izh,izw])

        st.opt_a_np[t,:,izh,izw]=(1+st.r)*st.A+st.yH[t,izh]-st.opt_c_np[t,:,izh,izw]
            
    #Choose whether to participate in the mkt: pick the choice that gives the...
    #...highest utility! st.V is the envolop of st.V_p and st.V_np!
    #if t>=st.Tr:st.V_p[t,:,izh,izw]=-np.inf
    c= st.V_np[t,:,izh,izw]/st.σ if t>=st.Tr else np.maximum(st.V_p[t,:,izh,izw]/st.σ,st.V_np[t,:,izh,izw]/st.σ)
    st.V[t,:,izh,izw]=st.σ*(c+np.log(np.exp(st.V_p[t,:,izh,izw]/st.σ-c)+np.exp(st.V_np[t,:,izh,izw]/st.σ-c)))#part*st.V_p[t,:,izh,izw]+(1-part)*st.V_np[t,:,izh,izw]
    st.pr[t,:,izh,izw]=np.exp(st.V_p[t,:,izh,izw]/st.σ-st.V[t,:,izh,izw]/st.σ)#np.exp(st.V_p[t,:,izh,izw]/st.σ)/np.exp(st.V[t,:,izh,izw]/st.σ)


    #Compute marginal utility of consumption:
    p_Vd = np.interp(st.opt_c_p[t,:,izh,izw],st.grid_Ctot,st.grid_marg_u)
    np_Vd = np.interp(st.opt_c_np[t,:,izh,izw],st.grid_Ctot,st.grid_marg_u)
    st.Vd[t,:,izh,izw] = p_Vd*st.pr[t,:,izh,izw]+np_Vd*(1.0-st.pr[t,:,izh,izw])
    
    # if t==st.T-2:
    #     print(111)
    #Check DC_EGM against a brute force algorithm
    if st.diagnostic:
        
        cons=np.linspace(0.0,(1+st.r)*st.A+st.yH[t,izh],1000)
        ass=(1+st.r)*st.A+st.yH[t,izh]-cons
        util=u(cons,st)+st.β*np.interp(ass,st.A,EV)
        
        consp=np.linspace(0.0,(1+st.r)*st.A+st.yW[t,izh]+st.yH[t,izh],1000)
        assp=(1+st.r)*st.A+st.yH[t,izh]+st.yW[t,izh]-consp
        utilp=u(consp,st)+st.β*np.interp(assp,st.A,EV)-st.ψ
        
        #Where to check
        where=st.A>=0
        if (t<=st.Tr-2) & np.any(((np.max(util,axis=0)[where]-st.V_np[t,:,izh,izw][where])>0.001)):
            print("We have a problem...")
            #import matplotlib.pyplot as plt
            #plt.plot(st.A,np.max(util,axis=0)-st.V_np[t,:,izh,izw])
            inde=np.argmax(util,axis=0)
            optc=np.zeros(st.A.shape)
            for i in range(len(optc)):optc[i]=cons[inde[i],i]
            
            inde=np.argmax(utilp,axis=0)
            optcp=np.zeros(st.A.shape)
            for i in range(len(optcp)):optcp[i]=consp[inde[i],i]
            
            optce=optcp.copy()
            wherenp=(np.max(util,axis=0)-np.max(utilp,axis=0)>0)
            optce[wherenp]=optc[wherenp]

###############################################################################
############## MODEL SOLUTION AND SIMULATIONS START BELOW!  ###################
###############################################################################
#--------------------------------#
#     Life-cycle computation     #
#--------------------------------#

#start the timer
start = time.time()

#Compute value and policy functions for (t,izh,zhw) by backward induction
for t in reversed(range(st.T)):         #Loop over (reversed) age...
    for izh in range(st.Nz):            #...H's income shocks izh...
        for izw in range(st.Nz):        #...W's income shocks izw!
        
            #Invoke"pol_func" to obtain policy functions, using the method..
            #..of endogenous gridpoints. It solves for all assets ia=0,..,NA
            pol_func(st,t,izh,izw)

#compute elapsed time for solving the model
finish = time.time() - start
print("TOTAL ELAPSED TIME: ", round(finish, 4), " seconds. \n")

#----------------------------#
#     Simulate the model     #
#----------------------------#
#Initial condition (same for all) on yH and yW in t=0. It equals (st.Nz-1)/2
initial_states=np.ones(st.Nag,dtype=np.int32)*int((st.Nz-1)/2)

#Simulate the markov chain Π using the routines of quant econ for H and W
mc = qe.MarkovChain(st.Π)               
         
#H's case below!! Note: random_state sets the "seed", which manages randomness..
#..on the computer. This avoid H and W having exactly the same shocks"        
Xh = mc.simulate(ts_length=st.T,init=initial_states,random_state=7).T

#W's case below     
Xw = mc.simulate(ts_length=st.T,init=initial_states,random_state=8).T

#Shock for participation
Xs=np.random.rand(st.T,st.Nag)

#Initialize path of assets (this will also work as initial condition),...
#..consumption, income and labor force participation
A_path,c_path,yh_path,yw_path,part_path=np.zeros((5,st.T,st.Nag))

#Get (potential) income given the shocks for H and W
for t in range(st.T):yh_path[t,:]=st.yH[t,Xh[t,:]]
for t in range(st.T):yw_path[t,:]=st.yW[t,Xw[t,:]]
    

#Get female labor force participation in the two rows below.
for t in range(st.T):
    for i in range(st.Nag):
        
        #participation choice
        Vp=np.interp(A_path[t,i],st.A,st.V_p[t,:,Xh[t,i],Xw[t,i]])
        Vnp=np.interp(A_path[t,i],st.A,st.V_np[t,:,Xh[t,i],Xw[t,i]])
        
        c= np.maximum(Vp/st.σ,Vnp/st.σ)
        den=st.σ*(c+np.log(np.exp(Vp/st.σ-c)+np.exp(Vnp/st.σ-c)))#part*st.V_p[t,:,izh,izw]+(1-part)*st.V_np[t,:,izh,izw]       
        part_i=np.exp(Vp/st.σ-den/st.σ)
        part_path[t,i]=(part_i>Xs[t,i])
        
        #Assets choice
        app=st.opt_a_p[t,:,Xh[t,i],Xw[t,i]].copy() if part_path[t,i] else st.opt_a_np[t,:,Xh[t,i],Xw[t,i]].copy()
        if (t<st.T-1):
            A_path[t+1,i]=np.interp(A_path[t,i],st.A,app) 
            

#Use the budget constraint to get the consumption path
for t in range(st.T):
    if(t<st.T-1):                   #If working or retired (before last period)
        c_path[t,:]=(1+st.r)*A_path[t,:]+yh_path[t,:]+ \
                    yw_path[t,:]*part_path[t,:]-A_path[t+1,:]
    else:                           #If last period T!  
        c_path[t,:]=(1+st.r)*A_path[t,:]+yh_path[t,:]        

#----------------------------#
#     Results                #
#----------------------------#

print("For σ_z={}, average wealth is {}".format(st.σ_z,np.mean(A_path)))
print("For σ_z={}, average FLP is    {}".format(st.σ_z,np.mean(part_path)))

#Correlation of income shocks and changes in labor participation
corr=np.corrcoef(np.stack((np.diff(yh_path,axis=0).flatten(),\
                           np.diff(part_path,axis=0).flatten())))[0,1]
    
print("The correlation between shocks and participation is {}".format(corr))


########################
#Policy/Value Functions
#######################

#Graph the value of participating or not in the labor market 2 periods before retirement
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(st.A,st.V_np[st.Tr-29,:,0,0], label="Value of FLP=0") 
ax.plot(st.A,st.V_p[st.Tr-29,:,0,0], label="Value of FLP=1") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Utility')                  #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph

#Second derivatives of the utility of working and not, 2 periods before retirement
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(st.A[:],np.gradient(np.gradient(st.V_np[st.Tr-20,:,0,0])), label="FLP=0") 
ax.plot(st.A[:],np.gradient(np.gradient(st.V_p[st.Tr-20,:,0,0])), label="FLP=1") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Utility')                  #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph

#Graph difference in the value of participating and not in the mkt
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(st.A,st.V_np[0,:,0,0]-st.V_p[0,:,0,0], label="Value(FLP=0)-Value(FLP=1)") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Utility')                  #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph


#Consumption under participation and not participation
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(st.A,st.opt_c_p[0,:,0,0], label="Cons if FLP=1") 
ax.plot(st.A,st.opt_c_np[0,:,0,0], label="Cons if FLP=0") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Consumption')              #Label of y axis
plt.legend()                              #Plot the legend
plt.show()     

#Savings under participation and not participation
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(st.A,st.opt_a_p[0,:,0,0], label="FLP=1") 
ax.plot(st.A,st.opt_a_np[0,:,0,0], label="FLP=0") 
ax.grid()
ax.set_xlabel('Assets in t')              #Label of x axis
ax.set_ylabel('Assets in t+1')            #Label of y axis
plt.legend()                              #Plot the legend
plt.show()    


####################################
#Simulations 
###################################

#...of one individual...

#Simulated Assets
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(np.cumsum(np.ones(st.T)),A_path[:,0], label="A") 
ax.grid()
ax.set_xlabel('Age')                      #Label of x axis
ax.set_ylabel('Individual Assets')        #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph


#Graph the value of participating or not in the labor market
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(np.cumsum(np.ones(st.T)),c_path[:,0], label="c") 
ax.grid()
ax.set_xlabel('Age')                      #Label of x axis
ax.set_ylabel('Individual Consumption')   #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph

#...and aveages!

#Simulated Assets
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(np.cumsum(np.ones(st.T)),A_path.mean(axis=1), label="A") 
ax.grid()
ax.set_xlabel('Age')                      #Label of x axis
ax.set_ylabel('Average Assets')           #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph


#Graph the value of participating or not in the labor market
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(np.cumsum(np.ones(st.T)),c_path.mean(axis=1), label="c") 
ax.grid()
ax.set_xlabel('Age')                      #Label of x axis
ax.set_ylabel('Average Consumption')      #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph




