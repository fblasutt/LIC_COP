from numba import njit
from interpolation.splines import prefilter,eval_spline
from interpolation.splines import filter_cubic,eval_cubic
import numpy as np

@njit
def cube(x):
    return x**3.0


grid=((0.0,1.0,100),)
fgrid=cube(np.linspace(*grid[0])) 
point=np.array([0.33333333]) 



@njit
def interp1(grid1,fgrid1,point1):

    coeffs1 = prefilter(grid1, fgrid1,k=3)
    return eval_spline(grid1,coeffs1,point1,out=None, k=3, diff="None", extrap_mode="linear") 

    
    
@njit
def interp2(grid2,fgrid2,point2):

    coeffs2 = filter_cubic(grid2, fgrid2) 
    return eval_cubic(grid2,coeffs2,point2)  



fpoint2=interp2(grid,fgrid,point)
fpoint1=interp1(grid,fgrid,point)
    
    
