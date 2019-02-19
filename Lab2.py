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


def precompute_matrix_p():

    global matrix_p
    matrix_p = np.outer(t_array, t_array)

    for i in range(t_array.size):
        for j in range(t_array.size):
            matrix_p[i][j] *= kernel(data_points[i], data_points[j])


def extract_non_zero_alpha_positions(alpha_array):

    non_zero_positions = []

    for i in range(len(alpha_array)):
        if alpha_array[i] > threshold:
            non_zero_positions.append(i)

    return non_zero_positions


def extract_non_zero_alphas(alpha_array):
    non_zero_positions = extract_non_zero_alpha_positions(alpha_array)

    global non_zero_alphas

    non_zero_alphas = []

    for i in range(len(non_zero_positions)):
        non_zero_alphas.append([alpha_array[non_zero_positions[i]],
                                t_array[non_zero_positions[i]],
                                data_points[non_zero_positions[i]]])

    return non_zero_alphas


def indicator_function(non_zero_alphas, b_value, s):
    indicator_val = 0

    for i in range(len(non_zero_alphas)):
        alpha = non_zero_alphas[i][0]
        t_val = non_zero_alphas[i][1]
        x_val = non_zero_alphas[i][2]
        indicator_val += alpha * t_val * kernel(x_val, s)

    indicator_val -= b_value

    return indicator_val


def kernel(x1, x2):
    if option == 0:
        return linear_kernel(x1, x2)
    else:
        if option == 1:
            return polynomial_kernel(x1, x2, p)
        else:
            if option == 2:
                return RBF_kernel(x1, x2, sigma)
            else:
                return linear_kernel(x1, x2)


def linear_kernel(x1, x2):
    scalar = np.dot(x1, x2)
    return scalar


def polynomial_kernel(x1, x2, p):
    scalar = pow((np.dot(x1, x2) + 1), p)
    return scalar


def RBF_kernel(x1, x2, sigma):
    # Radial Basis Function kernel for
    subs = x1-x2
    xylength = np.linalg.norm(subs)
    scalar = np.exp(- (pow(xylength, 2)) / (2 * pow(sigma, 2)))
    print("RBF kernel test : ", scalar)
    return scalar


def zerofun(alpha_array):
#implements equality constrain of function (10)
#in lab2 instructions

    dot_product = np.dot(alpha_array, t_array)
    return dot_product


def calculate_b(s_vec, x_vec, t_s, slack, C):
#compute b according to sum (ai ti K(s,xi) -ts
#slack is boolean, true if slack is used, false otherwise

    result = 0

    if slack:
        for i in range(len(x_vec)):
            if 0 <= x_vec[i][0] <= C:
                result += x_vec[i][0] * x_vec[i][1] * kernel(s_vec, x_vec[i][2])
    else:

        for i in range(len(x_vec)):
                result += x_vec[i][0] * x_vec[i][1] * kernel(s_vec, x_vec[i][2])

    result -= t_s

    return result


def plot_generated_data(classA, classB):
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'bo')

    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'ro')

    plt.axis('equal')

    # plt.show()


def generate_data():

    class_a = np.concatenate(
        (np.random.randn(int(N/4), 2) * 0.6 + [1.5, 0.5],
         np.random.randn(int(N/4), 2) * 0.6 + [2, 1.5])
    )

    class_b = np.random.randn(int(N/2), 2) * 0.6 + [0.9, 0.80]

    inputs = np.concatenate((class_a, class_b))
    targets = np.concatenate(
        (np.ones(class_a.shape[0]),
         -np.ones(class_b.shape[0]))
    )

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    global t_array
    t_array = targets

    global data_points
    data_points = inputs

    plot_generated_data(class_a, class_b)


def plot_svm():
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[indicator_function(non_zero_alphas, b_value, [x, y])
                      for x in xgrid]
                     for y in ygrid])

    plt.contour(xgrid, ygrid, grid,
                (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1, 3, 1))

    plt.plot([p[2][0] for p in non_zero_alphas],
             [p[2][1] for p in non_zero_alphas],
             'gx')

    plt.savefig('svmplot.pdf')

    plt.show()


def compute_mean_b_value():
    global b_value, non_zero_alphas
    b_sum = 0
    for i in range(len(non_zero_alphas)):
        b_value = calculate_b(non_zero_alphas[i][2],
                              non_zero_alphas, non_zero_alphas[i][1], use_slack, C)
        print("B val : " + str(b_value))
        b_sum += b_value

    b_value = b_sum / len(non_zero_alphas)

    print("B mean " + str(b_value))

    return b_value


def compute_alphas():
    global alphas
    start = np.zeros(N)
    B = [(0, C) for b in range(N)]
    XC = {'type': 'eq', 'fun': zerofun}

    ret = minimize(objective, start, bounds=B, constraints=XC)

    alphas = ret['x']


def main_method():

    # np.random.seed(100)

    print()

    generate_data()
    precompute_matrix_p()
    compute_alphas()
    extract_non_zero_alphas(alphas)
    compute_mean_b_value()
    plot_svm()


# Instantiate empty structures
t_array = np.array([])
matrix_p = np.matrix([[]])
data_points = np.array([])
alphas = np.array([])
non_zero_alphas = np.array([])

# Instantiate parameters
# Polynomial
p = 2

# Number of points
N = 80

# Do we use slack? If false, don't forget to change C to None
use_slack = True

# Slack value; if we don't want to set higher boundary, we make C equal to None
C = 20

# Sigma for RBF kernel
sigma = 2

# Type of kernel used
# Where : 0 - linear; 1 - polynomial; 2 - RBF; else - linear
option = 0

# threshold for alphas
threshold = pow(10, -5)

b_value = 0

# Method calls
main_method()





