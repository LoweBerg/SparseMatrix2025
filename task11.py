# Written by Lowe Berg

import timeit
import matplotlib.pyplot as plt


def compareinsert():
    setup = """
import Sparse
import numpy as np
A = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0],
              [19, 0, 0, 0, 0, 0, 5, 0],
              [0, 0, 10, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 2, 0, 0, 0, 17, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0]
              ])
SA = Sparse.SparseMatrix(A)
    """

    x = range(1, 201)

    y = [timeit.timeit("SA.edit(3, 2, 4)", setup=setup, number=n)*1000 for n in x]

    plt.scatter(x, y, label='Our implementation')

    setup = """
from scipy.sparse import csr_array
import numpy as np
A = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0],
              [19, 0, 0, 0, 0, 0, 5, 0],
              [0, 0, 10, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 2, 0, 0, 0, 17, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0]
              ])
SA = csr_array(A)     
    """

    y = [timeit.timeit("SA[2, 4] = 3", setup=setup, number=n)*1000 for n in x]

    plt.scatter(x, y, color="r", label="Scipy CSR matrix")

    plt.title("Runtime for performing n edits")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Edits (n)")
    plt.legend()
    plt.grid()

    plt.show()


def compareaddition():
    setup = """
import Sparse
import numpy as np
A = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0],
              [19, 0, 0, 0, 0, 0, 5, 0],
              [0, 0, 10, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 2, 0, 0, 0, 17, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0]
              ])
SA = Sparse.SparseMatrix(A)
B = np.array([[0, 2, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0],
              [19, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 4, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 3, 0, 1, 0],
              [0, 2, 0, 0, 0, 17, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 20, 0, 0, 0, 1, 0, 0]
              ])
SB = Sparse.SparseMatrix(B)
    """

    x = range(1, 201)

    y = [timeit.timeit("SA + SB", setup=setup, number=n) * 1000 for n in x]

    plt.scatter(x, y, label='Our implementation')

    setup = """
from scipy.sparse import csr_array
import numpy as np
A = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0],
              [19, 0, 0, 0, 0, 0, 5, 0],
              [0, 0, 10, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 2, 0, 0, 0, 17, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0]
              ])
SA = csr_array(A)
B = np.array([[0, 2, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0],
              [19, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 4, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 3, 0, 1, 0],
              [0, 2, 0, 0, 0, 17, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 20, 0, 0, 0, 1, 0, 0]
              ])
SB = csr_array(B)
    """

    y = [timeit.timeit("SA + SB", setup=setup, number=n) * 1000 for n in x]

    plt.scatter(x, y, color="r", label="Scipy CSR matrix")

    plt.title("Runtime for performing n additions")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Additions (n)")
    plt.legend()
    plt.grid()

    plt.show()


def comparemultiplication():
    setup = """
import Sparse
import numpy as np
A = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0],
              [19, 0, 0, 0, 0, 0, 5, 0],
              [0, 0, 10, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 2, 0, 0, 0, 17, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0]
              ])
SA = Sparse.SparseMatrix(A)
vec = np.array([5, 3, 11, 0, -6, 2, 10, -14])
    """

    x = range(1, 201)

    y = [timeit.timeit("SA.vec_mul(vec)", setup=setup, number=n) * 1000 for n in x]

    plt.scatter(x, y, label="Our implementation")

    setup = """
from scipy.sparse import csr_array
import numpy as np
A = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0],
              [19, 0, 0, 0, 0, 0, 5, 0],
              [0, 0, 10, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 2, 0, 0, 0, 17, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0]
              ])
SA = csr_array(A)
vec = np.array([5, 3, 11, 0, -6, 2, 10, -14])
    """

    y = [timeit.timeit("SA.dot(vec)", setup=setup, number=n) * 1000 for n in x]

    plt.scatter(x, y, color="r", label="Scipy CSR matrix")

    plt.title("Runtime for performing n multiplications")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Multiplications (n)")
    plt.legend()
    plt.grid()

    plt.show()


comparemultiplication()
