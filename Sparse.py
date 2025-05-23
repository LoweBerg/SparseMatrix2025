# Contains the SparseMatrix object
import numpy as np
tol = 1e-8  # Tolerance for zero values in the matrix

class SparseMatrix:
    def __init__(self, arr: np.ndarray):  # Sparsify matrix
        """
        Generates a Sparse Matrix using the Compressed Sparse Row (CSR) format
        with the following properties: \n
        V: Values of nonzero elements in matrix. \n
        col_index: Column indices for each nonzero element in matrix. \n
        row_counter: Number of nonzero elements in a row such that row i contains
        row_counter[i+1] - row_counter[i] elements (0-indexed). \n
        number_of_nonzero: Total number of nonzero elements in matrix. \n
        intern_represent: Sparse matrix compression format.

        :param arr: Numpy sparse matrix to be compressed
        """
        temp_V = []
        temp_col_index = []
        temp_row_counter = [0]
        temp_number_of_nonzero = 0  # can be further simplified later

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if abs(arr[i, j]) > tol:
                    temp_number_of_nonzero += 1
                    temp_V.append(arr[i, j])
                    temp_col_index.append(j)

            temp_row_counter.append(temp_number_of_nonzero)

        self._V = np.array(temp_V)
        self._col_index = np.array(temp_col_index)
        self._row_counter = np.array(temp_row_counter)
        self._number_of_nonzero = temp_number_of_nonzero
        self._intern_represent = 'CSR'
        self._shape = arr.shape # jag lade till denna /RS

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
    
    def __add__(self, other):
        if self._intern_represent != other._intern_represent:
            raise ValueError("Cannot add matrices with different representations")
        if self._shape != other._shape:
            raise ValueError("Cannot add matrices with different dimensions")
        
        sum_V = []
        sum_col_index = []
        sum_row_counter = [0]
        
        for i in range(self._shape[0]):
            row_start_self = self._row_counter[i]
            row_end_self = self._row_counter[i + 1]
            row_start_other = other._row_counter[i]
            row_end_other = other._row_counter[i + 1]
            
            row_self = {sel.col_index[i]: self._V[i] for i in range(row_start_self, row_end_self)}
            row_other = {other.col_index[i]: other._V[i] for i in range(row_start_other, row_end_other)}
            
            row_sum = {}
            for j in row_self:
                row_sum[j] = row_self[j]
            for j in row_other:
                if j in row_sum:
                    row_sum[j] += row_other[j]
                else:
                    row_sum[j] = row_other[j]
            for j in row_sum:
                if abs(row_sum[j]) > tol:
                    sum_V.append(row_sum[j])
                    sum_col_index.append(j)
            sum_row_counter.append(sum_V)
        
        sum = SparseMatrix(np.zeros((self._shape[0], self._shape[1])))
        sum._V = np.array(sum_V)
        sum._col_index = np.array(sum_col_index) 
        sum._row_counter = np.array(sum_row_counter)
        sum._number_of_nonzero = len(sum_V)
        sum._intern_represent = 'CSR'
        sum._shape = self._shape
        return sum



