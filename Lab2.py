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


def extract_non_zero_alpha_positions(alpha_array):

    non_zero_positions = []
    threshold = 10 ^ (-5)

    for i in range(len(alpha_array)):
        if alpha_array[i] > threshold:
            non_zero_positions.append(i)

    return non_zero_positions


def extract_non_zero_alphas(alpha_array, t_array, x_datapoints):
    non_zero_positions = extract_non_zero_alpha_positions(alpha_array)

    non_zero_alphas = []

    for i in range(len(non_zero_positions)):
        non_zero_alphas.append([alpha_array[i], t_array[i], x_datapoints[i]])

    return non_zero_alphas


def indication_function(non_zero_alphas, b_value, s):
    indicator_val = 0

    for i in range(len(non_zero_alphas)):
        alpha = non_zero_alphas[i][0]
        t_val = non_zero_alphas[i][1]
        x_val = non_zero_alphas[i][2]
        indicator_val += alpha*t_val*kernel(x_val, s)

    indicator_val -= b_value

    return indicator_val


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
