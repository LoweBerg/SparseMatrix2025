# Hello, this is the main file

import Sparse
import numpy as np

print("Grattis på födelsedagen Freja")
for i in range(5):
    print("Ja må hon leva")

A = np.array([[0, 0, 0, 0],
              [5, 8, 0, 0],
              [0, 0, 3, 0],
              [0, 6, 0, 0]])

B = np.array([[10, 20, 0, 0, 0, 0],
              [0, 30, 0, 4, 0, 0],
              [0, 0, 50, 60, 70, 0],
              [0, 0, 0, 0, 0, 80]])

C = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0],
              [19, 0, 0, 0, 0, 0, 5, 0],
              [0, 0, 10, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 2, 0, 0, 0, 17, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0]
              ])

D = np.array([[0, 0, 0, 0, 1],
              [5, 8, 0, 0, 0],
              [0, 0, 3, 0, 0],
              [0, 6, 0, 0, 1]])

E = np.array([[1e-8, 1e-8, 1e-8, 1e-8, 1],
              [5, 8, 1e-8, 1e-8, 0],
              [0, 0, 3, 0, 0],
              [0, 6, 0, 0, 1]])

SA = Sparse.SparseMatrix(A)
SB = Sparse.SparseMatrix(B)
SC = Sparse.SparseMatrix(C)
SD = Sparse.SparseMatrix(D)
SE = Sparse.SparseMatrix(E)

print(SA)
print(SB)
print(SC)
print(SD)
print(SE)