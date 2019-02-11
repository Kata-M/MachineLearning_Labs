import numpy as np
import random as rand
import math as math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def kernel(xVector,yVector):
#linear kernel function for
  scalar = np.dot(xVector, yVector)
  return scalar

def zerofun(a,t):
#implements equality constrain of function (10)
#in lab2 instructions
  result = np.dot(a,t)
  if(result == 0):
      return True
  else:
      return False

#minimize function call here
#ret =minimize(objective,start,bounds=B,constrains=XC)
#alpha = ret['x']
#print(alpha)


#test kernel function
y = np.array([0, 2, 0])
x = np.array([2, 0, 4])
kernel(x,y)

print("kernel function returns :",kernel(x,y))

#test zerofun
print("zerofun returns :",zerofun(x,y))
