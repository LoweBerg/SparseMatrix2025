# Written by Lowe Berg

import timeit
import matplotlib.pyplot as plt

x = range(1, 1000)

y = [timeit.timeit(f'toeplitz({n})', 'from Sparse import SparseMatrix; toeplitz = SparseMatrix.short_toeplitz',
                   number=100) for n in x]

plt.scatter(x, y, label='Short method')

y = [timeit.timeit(f'toeplitz({n})', 'from Sparse import SparseMatrix; toeplitz = SparseMatrix.toeplitz',
                   number=100) for n in x]

plt.scatter(x, y, color='r', label='Long method')

plt.title("Runtime comparison of different toeplitz generation methods")
plt.ylabel("Runtime for 100 executions (s)")
plt.xlabel("Number of rows in toeplitz matrix")
plt.legend()

plt.show()
