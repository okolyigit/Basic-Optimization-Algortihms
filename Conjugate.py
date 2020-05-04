"""
Optimization with Conjugate Direction Mothods for Multi-Variable Functions

"""
import numpy as np

#Objective Function 
def f(x,y,z):
    return (3/2)*(x**2)+2*(y**2)+(3/2)*(z**2)+x*z+2*y*z-3*x-z

#First Derivative of Objective Function
def fprime(x,y,z):
    return np.array((3*x+z-3,4*y+2*z,x+2*y+3*z-1));
    
#Second Derivative of Objective Function
def fsecond(x,y,z):
   return np.array(([3, 0, 1],
                    [0, 4, 2],
                    [1, 2, 3]));

#Initialisation
(x0,y0,z0)=(0,0,0)
maxiter=3
b= [3,0,1]
d=-fprime(x0,y0,z0)

#main		                     
print('f(0)=',f(x0,y0,z0),'\n')
print('g(0)=',fprime(x0,y0,z0),'\n')
print('Q=',fsecond(x0,y0,z0),'\n')

for i in range(maxiter):
       alpha =-(np.transpose(fprime(x0,y0,z0)).dot(d)/
              (np.transpose(d).dot(fsecond(x0,y0,z0))).dot(d))
       ([x1,y1,z1])= ([x0,y0,z0]) + (alpha)*d
       if (fprime(x1,y1,z1)[0]==0 and fprime(x1,y1,z1)[1]==0 and fprime(x1,y1,z1)[2]==0):
           print('Reached to local minimum.')
           break
       beta=(np.transpose(fprime(x1,y1,z1).dot(fsecond(x0,y0,z0)).dot(d))/
                   (np.transpose(d)).dot(fsecond(x0,y0,z0)).dot(d))
       d=-fprime(x1,y1,z1)+beta*d
       print(i+1,'.iteration')
       print('alpha=:',alpha)
       print([round(x1,4),round(y1,4),round(z1,4)])
       print(f(x1,y1,z1))
       print('beta=',round(beta,4))
       print('d=',d,'\n')
       [x0,y0,z0]=[x1,y1,z1]
print('The point that Minimum value is obtained:')
print((round(x0,4),round(y0,4),round(z0,4)))