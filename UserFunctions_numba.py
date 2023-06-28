from numba import njit

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


