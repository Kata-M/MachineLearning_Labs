import numpy as np

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


#test kernel function
y = np.array([0, 2, 0])
x = np.array([2, 0, 4])
kernel(x,y)

print("kernel function returns :",kernel(x,y))

#test zerofun
print("zerofun returns :",zerofun(x,y))
