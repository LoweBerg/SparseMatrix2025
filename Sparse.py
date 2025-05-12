# Contains the SparseMatrix object
import numpy as np


class SparseMatrix:

    def __init__(self, arr: np.ndarray):  # Sparsify matrix
        temp_V = []
        temp_col_index = []
        temp_row_counter = [0]
        temp_number_of_nonzero = 0  # can be further simplified later

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] != 0:
                    temp_number_of_nonzero += 1
                    temp_V.append(arr[i, j])
                    temp_col_index.append(j)

            temp_row_counter.append(temp_number_of_nonzero)

        self._V = np.array(temp_V)
        self._col_index = np.array(temp_col_index)
        self._row_counter = np.array(temp_row_counter)
        self._number_of_nonzero = temp_number_of_nonzero
        self._intern_represent = 'CSR'

    def __repr__(self):
        return(
                f"""
                NNZ: {self._number_of_nonzero}
                Values: {self._V}
                Row Counter: {self._row_counter}
                Columns: {self._col_index}
                """
               )

    def edit(self, a, x, y):  # Will not work yet
        in_matrix = False
        index = None
        for i in range(np.size(self._col_index)):  # Check if indices exist in matrix
            index = i
            if (self._col_index[i], self._row_counter[i]) == (x, y):
                in_matrix = True
                break

            if self._col_index[i] > x or (self._col_index[i] == x and self._row_counter[i] > y):
                break

        if in_matrix:
            if a != 0:  # simple edit
                self._V[index] = a
                self._col_index[index] = x
                self._row_counter[index] = y
            else:  # remove from matrix
                self._number_of_nonzero -= 1
                self._V = np.concat(self._V[:index], self._V[index+1:])
                self._col_index = np.concat(self._col_index[:index], self._col_index[index+1:])
                self._row_counter = np.concat(self._col_index[:index], self._col_index[index + 1:])

        elif not in_matrix and a != 0:
            self._number_of_nonzero += 1
            self._V = np.concat(self._V[:index], a, self._V[index:])

