def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    if len(a[0]) != len(b):
        return -1
    result = []
    for row in a:
        total = 0
        for i in range(len(row)):
            total += row[i] * b[i]
        result.append(total)
    return result

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return [list(i) for i in zip(*a)]

import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int|float]) -> list[list[int|float]]:
    # Not compatible case
    if len(a)*len(a[0]) != new_shape[0]*new_shape[1]:
        return []
    return np.array(a).reshape(new_shape).tolist()

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'column':
        return [sum(col) / len(matrix) for col in zip(*matrix)]
    elif mode == 'row':
        return [sum(row) / len(row) for row in matrix]
    else:
        raise ValueError("Mode must be 'row' or 'column'")

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return [[element * scalar for element in row] for row in matrix]

def calculate_eigenvalues(matrix: list[list[float]]) -> list[float]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    determinant = a * d - b * c
    # Calculate the discriminant of the quadratic equation
    discriminant = trace**2 - 4 * determinant
    # Solve for eigenvalues
    lambda_1 = (trace + discriminant**0.5) / 2
    lambda_2 = (trace - discriminant**0.5) / 2
    return [lambda_1, lambda_2]

import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    # Convert to numpy arrays for easier manipulation
    A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)
    
    # Check if the matrices T and S are invertible
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        # raise ValueError("The matrices T and/or S are not invertible.")
        return -1
    
    # Compute the inverse of T
    T_inv = np.linalg.inv(T)

    # Perform the matrix transformation; use @ for better readability
    transformed_matrix = np.round(T_inv @ A @ S, 3)
    
    return transformed_matrix.tolist()
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    determinant = a * d - b * c
    if determinant == 0:
        return None
    inverse = [[d/determinant, -b/determinant], [-c/determinant, a/determinant]]
    return inverse

import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    d_a = np.diag(A)
    nda = A - np.diag(d_a)
    x = np.zeros(len(b))
    x_hold = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(A)):
            x_hold[i] = (1/d_a[i]) * (b[i] - sum(nda[i]*x))
        x = x_hold.copy()
    return np.round(x,4).tolist()

import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    X_transpose = X.T
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    theta = np.round(theta, 4).flatten().tolist()
    return theta
import numpy as np 


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
   a = A
   a_t = np.transpose(a)
   a_2 = a_t @ a
   v = np.eye(2)
   for _ in range(1):
       if a_2[0,0] == a_2[1,1]:
           theta = np.pi/4
       else:
           theta = 0.5 * np.arctan2(2 * a_2[0,1], a_2[0,0] - a_2[1,1])
       r = np.array(
           [
               [np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]
               ]
           )
       d = np.transpose(r) @ a_2 @ r
       a_2 = d
       v = v @ r
   s = np.sqrt([d[0,0], d[1,1]])
   s_inv = np.array([[1/s[0], 0], [0, 1/s[1]]])
   u = a @ v @ s_inv
   return (u, s, v.T)
    
