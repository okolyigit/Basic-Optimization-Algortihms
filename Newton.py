"""
Optimization with Newton Method for Powell Function

"""
from scipy import linalg
import numpy as np


#Objective Function (Powell Function)
def f(x,y,z,t):
    return (x+10*y)**2+5*(z-t)**2+(y-2*z)**4+10*(x-t)**4

#Initialisation
(x0,y0,z0,t0)=(3,-1,0,1)
maxiter=30

#First Derivative of Objective Function
def fprime(x,y,z,t):
    return np.array([2*(x+10*y)+40*(x-t)**3,
            20*(x+10*y)+4*(y-2*z)**3,
            10*(z-t)-8*(y-2*z)**3,
            -10*(z-t)-40*(x-t)**3,]);
    
#Second Derivative of Objective Function
def fsecond(x,y,z,t):
   return np.array(([2+120*(x-t)**2, 20, 0, -120*(x-t)**2],
             [20, 200+12*(y-2*z)**2, -24*(y-2*z)**2, 0],
            [0, -24*(y-2*z)**2, 10+48*(y-2*z)**2, -10],
            [-120*(x-t)**2, 0, -10, 10+120*(x-t)**2]));

#main		                     
print('f(0)=',f(x0,y0,z0,t0),'\n')
print('g(0)=',fprime(x0,y0,z0,t0),'\n')
print('F(0)=',fsecond(x0,y0,z0,t0),'\n')
print('inv F(0)', linalg.inv(fsecond(x0,y0,z0,t0)),'\n')

for i in range(maxiter):
      ([x1,y1,z1,t1])= ([x0,y0,z0,t0])- linalg.inv(fsecond(x0,y0,z0,t0)).dot(fprime(x0,y0,z0,t0))
      if (fprime(x1,y1,z1,t1)[0]==0 and fprime(x1,y1,z1,t1)[1]==0 and fprime(x1,y1,z1,t1)[2]==0 and fprime(x1,y1,z1,t1)[3]==0):
           print('Reached to local minimum.')
           break
      print(i+1,'. iterasyon',f(x1,y1,z1,t1))
      [x0,y0,z0,t0]=[x1,y1,z1,t1]
print('The point that Minimum value is obtained:')
print((round(x0,4),round(y0,4),round(z0,4),round(t0,4)))
