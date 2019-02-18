import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt


def compute_sum(alpha_array):
    resulted_matrix = np.outer(alpha_array, alpha_array)

    for i in range(len(alpha_array)):
        for j in range(len(alpha_array)):
            resulted_matrix[i, j] *= matrix_p[i, j]

    return resulted_matrix.sum()


def objective(alpha_array):

    sum_of_alpha = np.sum(alpha_array)
    sum_dual_formulation = (1/2) * compute_sum(alpha_array) - sum_of_alpha

    return sum_dual_formulation


def pre_compute_matrix_p():

    global matrix_p
    matrix_p = np.outer(t_array, t_array)

    for i in range(t_array.size):
        for j in range(t_array.size):
            matrix_p[i][j] *= linear_kernel(data_points[i], data_points[j])


def extract_non_zero_alpha_positions(alpha_array):

    non_zero_positions = []
    threshold = pow(10, -5)



    for i in range(len(alpha_array)):
        if alpha_array[i] > threshold:
            non_zero_positions.append(i)

    print(len(non_zero_positions))

    return non_zero_positions


def extract_non_zero_alphas(alpha_array):
    non_zero_positions = extract_non_zero_alpha_positions(alpha_array)

    non_zero_alphas = []

    for i in range(len(non_zero_positions)):
        non_zero_alphas.append([alpha_array[non_zero_positions[i]],
                                t_array[non_zero_positions[i]],
                                data_points[non_zero_positions[i]]])

    print(len(non_zero_alphas))

    return non_zero_alphas


def indicator_function(non_zero_alphas, b_value, s):
    indicator_val = 0

    for i in range(len(non_zero_alphas)):
        alpha = non_zero_alphas[i][0]
        t_val = non_zero_alphas[i][1]
        x_val = non_zero_alphas[i][2]
        indicator_val += alpha * t_val * linear_kernel(x_val, s)

    indicator_val -= b_value

    return indicator_val


def linear_kernel(x_vector, y_vector):
    # linear kernel function for
    scalar = np.dot(x_vector, y_vector)
    return scalar


def polynomial_kernel(x_vector, y_vector, p):
    # linear kernel function for
    scalar = pow((np.dot(x_vector, y_vector) + 1), p)
    return scalar


def zerofun(alpha_array):
#implements equality constrain of function (10)
#in lab2 instructions

    dot_product = np.dot(alpha_array, t_array)
    return dot_product

    # TODO : Not needed to return boolean values. Only the scalar value
    # if dot_product == 0:
    #     for i in range(len(alpha_array)):
    #         if alpha_array[i] > C:
    #             return False
    #         if alpha_array[i] < 0:
    #             return False
    #     return True
    # else:
    #     return False


def calculate_b(s_vec, x_vec, t_s, slack, C):
#compute b according to sum (ai ti K(s,xi) -ts
#slack is boolean, true if slack is used, false otherwise

    # print(len(x_vec))

    if slack:
        result = 0
        for i in range(len(x_vec)):
            if 0 < x_vec[i][0] < C:
                result += x_vec[i][0] * x_vec[i][1] * linear_kernel(s_vec, x_vec[i][2])
            else:
                print("alpha value is not acceptable")

        result = result - t_s
        # print("result of calculate_b (slack variables used):", result)
        return result
    else:
        result = 0
        for i in range(len(x_vec)):
                result += x_vec[i][0] * x_vec[i][1] * linear_kernel(s_vec, x_vec[i][2])

        result = result - t_s
        # print("result of calculate_b (no slack variables used):", result)

        return result


def plot_generated_data(classA, classB):
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b.')

    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')

    plt.axis('equal')
    # plt.savefig('svmplot.pdf')
    # plt.show()


def main_method():
    classA = np.concatenate(
        (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
         np.random.randn(10, 2) + 0.2 + [-1.5, 0.5])
    )

    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]),
         -np.ones(classB.shape[0]))
    )

    global N
    N = inputs.shape[0]

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    global t_array
    t_array = targets

    global data_points
    data_points = inputs

    # print(inputs)

#     For test purpose
    plot_generated_data(classA, classB)


N = 0
C = 10
t_array = np.array([])
matrix_p = np.matrix([[]])
data_points = np.array([])

main_method()
pre_compute_matrix_p()


#TODO radial basis function kernel (RBF)

# pre_compute_matrix_p()

#print(matrix_p)

start = np.zeros(N)
B = [(0, C) for b in range(N)]
XC = {'type': 'eq', 'fun': zerofun}

# print(XC)
ret = minimize(objective, start, bounds=B, constraints=XC)

alphas = ret['x']

non_zero_alphas = extract_non_zero_alphas(alphas)

b_value = -100

for i in range(len(non_zero_alphas)):

    b_value = calculate_b(non_zero_alphas[i][2], non_zero_alphas, non_zero_alphas[i][1], False, C)
    print(b_value)

# Plotting the Decision Boundary

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)

grid = np.array([[indicator_function(non_zero_alphas, b_value, [x, y])
                  for x in xgrid]
                 for y in ygrid])

plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))

plt.show()

#test kernel functions
#y = np.array([5, 3 , 1])
#x = np.array([4, 2 , 6])
#linear_kernel(x, y)

#objective(alpha)

#print("linear kernel function returns :", linear_kernel(x, y))

#print("polynimial kernel function returns :", polynomial_kernel(x,y,2))

#test calculate_b
#calculate_b(alpha,t_array,s_vec,x_datapoints,t_s, True ,1)

#pre-compute matrix P (call the function only once at the beginning)
#pre_compute_matrix_p()


#test zerofun
#print("zerofun returns :",zerofun(x,y,8))


