import numba as nb

# set gender indication as globals
woman = 1
man = 2

############################
# User-specified functions #
############################
@nb.njit
def util(c_priv,c_pub,gender,rho,phi,alpha1,alpha2,love=0.0):

    return ((alpha1*c_priv**phi + alpha2*c_pub**phi)**(1.0-rho))/(1.0-rho) + love

@nb.njit
def resources_couple(A,R,inc_w,inc_m):
    # resources of the couple
    return R*A + inc_w + inc_m

@nb.njit
def resources_single(A,gender,inc_w,inc_m,R):
    # resources of single individual of gender "gender"
    income = inc_m if gender ==man else inc_w
    return R*A + income

@nb.njit
def cons_priv_single(C_tot,gender,rho,phi,alpha1,alpha2):
    # closed form solution for intra-period problem of single.
    return C_tot/(1.0 + (alpha2/alpha1)**(1.0/(1.0-phi)) )




