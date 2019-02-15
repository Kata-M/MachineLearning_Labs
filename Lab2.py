import numpy as np

# datapoints user for training and test
x_datapoints = np.array([1,2,3])

# TODO: find how to compute t_array
t_array = np.array([1,2,3])

matrix_p = np.matrix([[]])


def objective(alpha_array):

    sum_of_alpha = np.sum(alpha_array)
    sum = 1/2 * compute_sum(alpha_array) - sum_of_alpha
    return sum


def pre_compute_matrix_p():

    matrix_p = np.outer(t_array, t_array)

    for i in range(t_array.size):
        for j in range(t_array.size):
            matrix_p[i][j] *= kernel(x_datapoints[i], x_datapoints[j])


def compute_sum(alpha_array):
    resulted_matrix = np.outer(alpha_array, alpha_array)

    for i in range(len(alpha_array)):
        for j in range(len(alpha_array[0])):
            resulted_matrix[i][j] *= matrix_p[i][j]

    # TODO: compute the sum of all the elements in the matrix resulted_matrix

    return resulted_matrix


def kernel(x_vector, y_vector):

    # linear kernel function for
    scalar = np.dot(x_vector, y_vector)
    return scalar

def zerofun(a,t):
#implements equality constrain of function (10)
#in lab2 instructions
  result = np.dot(a,t)
  if(result == 0):
      return True
  else:
      return False

#pre-compute matrix P (call the function only once at the beginning)
pre_compute_matrix_p()


#test kernel function
y = np.array([0, 2, 0])
x = np.array([2, 0, 4])
kernel(x,y)

print("kernel function returns :",kernel(x,y))

#test zerofun
print("zerofun returns :",zerofun(x,y))
