from qiskit.quantum_info.operators import Operator
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.circuit import Gate
import numpy as np
from qiskit.providers.fake_provider import FakeManilaV2
from qiskit.circuit.library import RYGate
import math
from scipy import linalg

#helper function for computing tensor products
def tensor_product(A, B):
   #Input:  matrix A and matrix B (square matricies)
   #Output: tensor product of A and B
   tensor_product = np.zeros((len(A)*len(B), len(A[0])*len(B[0])), dtype=np.complex_)
   for i in range(len(A)):
      for j in range(len(A[0])):
         a = A[i][j]
         a_x_B = a*B
         for m in range(len(B)):
            for n in range(len(B[0])):
               tensor_product[i*len(B) + m][j*len(B[0]) + n] = a_x_B[m][n]
   return tensor_product

#returns the hilbert-schmidt distance with CCNOT
def hilbert_schmidt_distance_CCNOT(V):
   norm_sq_tr_U_dag_V = abs(np.trace(CCNOT.dot(V)))**2
   return math.sqrt(1 - norm_sq_tr_U_dag_V/(64))

#Toffoli Approximation

#Want: Hilbert-Schmidt Distance of 0.38268
#Hilbert-Schmidt Distance: d(V, U) = sqrt(1 - (|Tr(U^{dagger} * V)|^2)/(2^{2n}) ) for circuits U, V and number of qubits n.

#We calculate the distance between our approximation circuit V and U := CCNOT

CCNOT = np.array([
   [1 + 0j, 0 + 0j, 0 + 0j,  0 + 0j,  0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j ],
   [0 + 0j, 1 + 0j, 0 + 0j,  0 + 0j,  0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j ],
   [0 + 0j, 0 + 0j, 1 + 0j,  0 + 0j,  0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j ],
   [0 + 0j, 0 + 0j, 0 + 0j,  1 + 0j,  0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j ],
   [0 + 0j, 0 + 0j, 0 + 0j,  0 + 0j,  1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j ],
   [0 + 0j, 0 + 0j, 0 + 0j,  0 + 0j,  0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j ],
   [0 + 0j, 0 + 0j, 0 + 0j,  0 + 0j,  0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j ],
   [0 + 0j, 0 + 0j, 0 + 0j,  0 + 0j,  0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j  ],
])

outer_prod_0 = np.array([
   [1 + 0j, 0 + 0j],
   [0 + 0j, 0 + 0j]
])

outer_prod_1 = np.array([
   [0 + 0j, 0 + 0j],
   [0 + 0j, 1 + 0j]
])


I = np.array([
   [1 + 0j, 0 + 0j],
   [0 + 0j, 1 + 0j]
])

X = np.array([
   [0 + 0j, 1 + 0j],
   [1 + 0j, 0 + 0j]
])

CX = np.array([
   [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
   [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
   [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
   [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j],
])

T = np.array([ 
   [1 + 0j, 0 + 0j],
   [0 + 0j, 1/(math.sqrt(2)) + 1j/(math.sqrt(2))]  
])

# Computing square root of NOT and inverse of square root of NOT
evalues, evectors = np.linalg.eig(X)
sqrt_X = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
negative_sqrt_X = np.linalg.inv(sqrt_X)

# Computing fourth root of NOT
evalues, evectors = np.linalg.eig(sqrt_X)
fourth_root_X = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)

C_sqrt_X = tensor_product(outer_prod_1, sqrt_X) + tensor_product(outer_prod_0, I)
C_negative_sqrtX = tensor_product(outer_prod_1, negative_sqrt_X) + tensor_product(outer_prod_0, I)

#computing the Toffoli approximation:
comp_one = tensor_product(I, C_sqrt_X)
comp_two = tensor_product(CX, I)
comp_three = tensor_product(I, C_negative_sqrtX)
comp_four = tensor_product(CX, I)
comp_five = tensor_product(tensor_product(T, I), fourth_root_X)

V = comp_one.dot(comp_two.dot(comp_three.dot(comp_four.dot(comp_five))))

print(hilbert_schmidt_distance_CCNOT(V))

