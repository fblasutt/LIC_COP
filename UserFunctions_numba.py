import numba as nb

# set gender indication as globals
woman = 1
man = 2

############################
# User-specified functions #
############################
@nb.njit
def home_good(x,θ,λ,tb,couple=False,ishom=False):
    home_time=(2*tb+ishom*(1-tb)) if couple else tb
    return (θ*x**λ+(1.0-θ)*home_time**λ)**(1.0/λ)

@nb.njit
def util(c_priv,c_pub,gender,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=False,ishom=False):
    homegood=home_good(c_pub,θ,λ,tb,couple=couple,ishom=ishom)
    return ((α1*c_priv**ϕ1 + α2*homegood**ϕ2)**(1.0-ρ))/(1.0-ρ)+ love

@nb.njit
def resources_couple(A,inc_w,inc_m,R):
    # resources of the couple
    return R*A + inc_w + inc_m

@nb.njit
def resources_single(A,gender,inc_w,inc_m,R):
    # resources of single individual of gender "gender"
    income = inc_m if gender ==man else inc_w
    return R*A + income

@nb.njit
def cons_priv_single(C_tot,gender,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb):
    # closed form solution for intra-period problem of single.
    return C_tot/(1.0 + (α2/α1)**(1.0/(1.0-ϕ1)) )




