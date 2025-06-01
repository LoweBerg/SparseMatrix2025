# Contains the SparseMatrix object
import numpy as np
tol = 1e-8  # Tolerance for zero values in the matrix


class SparseMatrix:
    def __init__(self, arr: np.ndarray = None):  # Written by Lowe Berg
        """
        Generates a Sparse Matrix using the Compressed Sparse Row (CSR) format
        with the following properties: \n
        V: Values of nonzero elements in matrix. \n
        col_index: Column indices for each nonzero element in matrix. \n
        row_counter: Number of nonzero elements in a row such that row i contains
        row_counter[i+1] - row_counter[i] elements (0-indexed). \n
        number_of_nonzero: Total number of nonzero elements in matrix. \n
        intern_represent: Sparse matrix compression format. \n
        shape: shape of the uncompressed matrix

        :param arr: Numpy sparse matrix to be compressed
        """
        temp_V = []
        temp_col_index = []
        temp_row_counter = [0]
        temp_number_of_nonzero = 0  # can be further simplified later

        if arr is None:  # Create empty object
            self._V = np.array(temp_V)
            self._col_index = np.array(temp_col_index)
            self._row_counter = np.array(temp_row_counter)
            self._number_of_nonzero = temp_number_of_nonzero
            self._intern_represent = 'CSR'
            self._shape = (0,)
            return

        if arr.shape[1] is None:
            rows = 1
            cols = arr.shape[0]
        else:
            rows, cols = arr.shape

        for i in range(rows):
            for j in range(cols):
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
        self._shape = arr.shape  # jag lade till denna /RS

    def __repr__(self):  # Written by Lowe Berg and Amanda Bjerge Andersson
        if self._intern_represent == 'CSR':
            return(
                    f"""
                    NNZ: {self._number_of_nonzero}
                    Values: {self._V}
                    Row Counter: {self._row_counter}
                    Column index: {self._col_index}
                    """
                )
        if self._intern_represent == 'CSC':
            return(
                    f"""
                    NNZ: {self._number_of_nonzero}
                    Values: {self._V}
                    Column Counter: {self._col_counter}
                    Row index: {self._row_index}
                    """
                )

    @property
    def number_of_nonzero(self):
        return self._number_of_nonzero

    @property
    def V(self):
        return self._V

    @property
    def col_index(self):
        return self._col_index
        
    @property
    def row_index(self):
        return self._row_index
        
    @property
    def col_counter(self):
        return self._col_counter
           
    @property
    def row_counter(self):
        return self._row_counter

    @property
    def intern_represent(self):
        return self._intern_represent

    @property
    def shape(self):
        return self._shape

    def convert_for_transition(self):      # separate transition format that contains both col- and row indices/counters
        """
        Converts a given sparse matrix from CSR (Compressed Sparse Row) to
        our own transition format. This contains the information found in both
        the CSR and the CSC formats, so as to make conversion between them 
        simpler. It also allows for the potential implementation of conversions
        to other formats. 
        """
        # create col_counter
        n = self._shape[1]+1              # column length +1 to account for intial 0
        temp_col_counter = [0]*n          
        for i in range(len(self._col_index)):
            for col in range(self._col_index[i]+1, n):
                temp_col_counter[col] += 1         # adds one count to each element from and including the current index

        # create row_index
        temp_row_index = [0]*len(self._V)
        positions = temp_col_counter[:]       # slicing to make a copy to not change col_counter

        for row in range(len(self._row_counter)-1):
            for sans in range(self._row_counter[row], self._row_counter[row + 1]):
                col = self._col_index[sans]
                pos = positions[col]
                temp_row_index[pos] = row
                positions[col] += 1

        # reshuffle values to csc order
        temp_remapping = []         # list for remapped values to respective column indices
        temp_resorted_val = []
        for i in range(self._number_of_nonzero): 
            temp_remapping.append((self._V[i], self._col_index[i]))

        temp_remapping = sorted(temp_remapping, key= lambda x: x[1])
                # sorts the remapped values by number, referring to the column indexes

        for i in range(len(temp_remapping)):
            temp_resorted_val.append(temp_remapping[i][0])  # separates the now reshuffled values

        # compiling new arrays
        self._V = np.array(temp_resorted_val)
        self._row_index = np.array(temp_row_index)
        self._col_counter = np.array(temp_col_counter)
        self._intern_represent = 'transition'

    def CSC_conversion(self):
        """
        Checks if the given sparse matrix has been converted to the transition
        format. If not, it converts it using the convert_for_transition() 
        method. 
        Changes the internal representation to CSC so that the representation
        will match the CSC (Compressed Sparse Column) format.
        """
        if self._intern_represent != 'transition' :
            self.convert_for_transition()

        self._intern_represent = 'CSC'

        return self
        
      
      
    def same_matrix (self, other):
        
        
        """
        Compares two sparse matrices to check if they are the same. 
        If the two matrices are of different representations i.e. 
        CSR and CSC it converts the CSR to CSC.
        """
        
        
        if not isinstance(other, SparseMatrix):
            raise ValueError("Input needs to be a sparse matrix")
        
        self_copy = copy.deepcopy(self)
        other_copy = copy.deepcopy(other)
       
        if self_copy.intern_represent == other.intern_represent:
            if self_copy.intern_represent == 'CSR':
                
                List_self_V = self_copy._V.tolist() 
                List_other_V = other_copy._V.tolist()
                
                List_self_r_c = self_copy._row_counter.tolist()
                List_other_r_c = other_copy._row_counter.tolist()
                
                List_self_c_i = self_copy._col_index.tolist()
                List_other_c_i = other_copy._col_index.tolist()
                
                return (List_self_V == List_other_V and List_self_r_c == List_other_r_c and List_self_c_i == List_other_c_i)
            
            else:
                
                List_self_V = self_copy._V.tolist() 
                List_other_V = other_copy._V.tolist()
                
                List_self_c_c = self_copy._col_counter.tolist()
                List_other_c_c = other_copy._col_counter.tolist()
                
                List_self_r_i = self_copy._row_index.tolist()
                List_other_r_i = other._row_index.tolist()
                
                return (List_self_V == List_other_V and List_self_c_c == List_other_c_c and List_self_r_i == List_other_r_i)
        
        elif self_copy.intern_represent == "CSR" and other.intern_represent == "CSC":
            self_copy.CSC_conversion()
            
        elif self_copy.intern_represent == "CSC" and other.intern_represent == "CSR":
            other_copy.CSC_conversion()
                    
        else:
            raise ValueError("Neither are in a CSR or CSC sparse array, I cant compare these")
        
        List_self_V = self_copy._V.tolist() 
        List_other_V = other_copy._V.tolist()
        
        List_self_c_c = self_copy._col_counter.tolist()
        List_other_c_c = other_copy._col_counter.tolist()
        
        List_self_r_i = self_copy._row_index.tolist()
        List_other_r_i = other_copy._row_index.tolist()
       
        return (List_self_V == List_other_V and List_self_c_c == List_other_c_c and List_self_r_i == List_other_r_i)
       
        
    def __add__(self, other):
        if self._intern_represent != other.intern_represent:
            raise ValueError("Cannot add matrices with different representations")
        if self._shape != other.shape:
            raise ValueError("Cannot add matrices with different dimensions")
        
        sum_V = []
        sum_col_index = []
        sum_row_counter = [0]
        
        for i in range(self._shape[0]):
            row_start_self = self._row_counter[i]
            row_end_self = self._row_counter[i + 1]
            row_start_other = other.row_counter[i]
            row_end_other = other.row_counter[i + 1]
            
            row_self = {self._col_index[i]: self._V[i] for i in range(row_start_self, row_end_self)}
            row_other = {other.col_index[i]: other.V[i] for i in range(row_start_other, row_end_other)}
            
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
            sum_row_counter.append(len(sum_V))
        
        sum = SparseMatrix(np.zeros((self._shape[0], self._shape[1])))
        sum._V = np.array(sum_V)
        sum._col_index = np.array(sum_col_index) 
        sum._row_counter = np.array(sum_row_counter)
        sum._number_of_nonzero = len(sum_V)
        sum._intern_represent = 'CSR'
        sum._shape = self._shape
        return sum
        
    def vec_mul(self, arr: np.array):

        """
        Makes it possible to multiply a CSR sparse matrix with a 1 dimensional vector. 
        """
        
        Pre_mul = []    #every columb index has a corresponding value in the multiplier vector which is stored in Pre_mul
        New_matrix_list = []    #slicelist post multiplication
        
        for i in range(len(self._col_index)):
            Vec = arr[self._col_index[i]]
            Pre_mul.append(Vec)

        Post_mul = self._V * Pre_mul   #unsliced array containing the values V times their matching part of the vector

        for i in range(len(self._row_counter)-1):
            if self._row_counter[i+1] - self._row_counter[i] != 0:
                Vslice = Post_mul[self._row_counter[i]:self._row_counter[i+1]]  #splits up the multiplied values to their corresponding row
                Vlist = Vslice.tolist()                 #and makes every slice be a list so that the contents of every slice can be summed up
            else:
                Vlist = [0]
            V_sum = sum(Vlist)              #sums up each list giving the total worth of one row in the new array
            New_matrix_list.append(V_sum)
        
        New_matrix_array = np.array(New_matrix_list)                 #makes the list an array again
            
        return New_matrix_array   

    def edit(self, x, i, j): #Written by Herman Plank and ELis Eles
        """
        Takes in a value x, and a desired position row i and column j in matrix A, where said 
        value x is to replace whatever value was in that cell before. If x is smaller than the
        given tolerance tol, it is considered as a zero. If a nonzero is put in a cell containing
        a zero, or vice versa, the number of nonzero counter is updated accordingly.

        Values can also be added to a cell 'outside' of a matrix A. If the input i or j or both
        are bigger than matrix A's current shape, the matrix turns into a new matrix B of the 
        bigger shape with zeroes in the new rows and columns. Also, if a row or column is edited 
        such that the resulting matrix has an empty (zero filled) outer row or column, its shape
        is reduced to no store unnecessary zero rows.
        """

        if i < 0 or j < 0:
            raise ValueError('Indices must be positive integers')

        if i % 1 != 0 or j % 1 != 0:
            raise TypeError('Indices must be positive integers')
        
        isOccupied = False 
        nonZero = False
        
        if abs(x) > tol:
            nonZero = True
        
        if (i > self._shape[0] - 1): #Handles cases where a value is placed on a row outside of the current shape.
            if not nonZero:
                return self
            else:
                self._col_index = np.append(self._col_index, j)
                self._V = np.append(self._V, x)
                while np.size(self._row_counter) - 1 <= i:
                    self._row_counter = np.append(self._row_counter, self._row_counter[-1])
                self._row_counter[-1] += 1
                self._number_of_nonzero += 1
        
        else:
            row_start  = self._row_counter[i]
            row_end  = self._row_counter[i+1]
            
            if j in self._col_index[row_start:row_end]: #Checks if the cell being changed currently contains a nonzero or not.
                isOccupied  = True                      #If it contains a nonzero, the cell is considered 'occupied'.
            
            if isOccupied and nonZero:     #Occupied cell being changed to a nonzero x
                workCol = self._col_index[row_start:row_end] #workCol and workV contains all the values and their respective columns in row i.
                workV = self._V[row_start:row_end]
                n = 0
                while j > workCol[n]:     #Finds the correct spot in workV to replace the value.
                    n += 1
                workV[n] = x
                self._V = np.concatenate((self._V[0:row_start], workV, self._V[row_end:len(self._V)])) #Puts the entire value array back togehter
                
            elif not isOccupied and nonZero:    #Unoccupied cell changed to a nonzero x
                workCol = self._col_index[row_start:row_end]
                workV = self._V[row_start:row_end]
                
                if (workCol.size == 0) or (j > np.max(workCol)): #Case if the row is empty or j is the largest index
                    workCol = np.append(workCol, j)
                    workV = np.append(workV, x)
                else:
                    n = 0
                    while j > workCol[n]: #Finds the correct spot in workCol and workV to add the value and its index.
                        n += 1
                    workCol = np.insert(workCol, n, j)
                    workV = np.insert(workV, n, x)
                
                for a in range(i + 1, len(self._row_counter)):     #Corrects the row_counter
                    self._row_counter[a] += 1
                    
                self._col_index = np.concatenate((self._col_index[0:row_start], workCol, self._col_index[row_end:len(self._col_index)]))
                self._V = np.concatenate((self._V[0:row_start], workV, self._V[row_end:len(self._V)]))
                self._number_of_nonzero += 1      #Corrects the NNZ-counter
            
            elif isOccupied and not nonZero:    #Occupied cell being chaged to a zero
                workCol = self._col_index[row_start:row_end]
                workV = self._V[row_start:row_end]
                n = 0
                while j > workCol[n]: #Finds the correct spot in workCol and workV to remove the value and its index.
                    n += 1
                workCol = np.delete(workCol, n)
                workV = np.delete(workV, n)
                for a in range(i + 1, len(self._row_counter)):
                    self._row_counter[a] -= 1
                self._col_index = np.concatenate((self._col_index[0:row_start], workCol, self._col_index[row_end:len(self._col_index)]))
                self._V = np.concatenate((self._V[0:row_start], workV, self._V[row_end:len(self._V)]))
                self._number_of_nonzero -= 1
                
            while self._row_counter[-1] == self._row_counter[-2]:   #Shortens row_counter if the last row(s) only has zeros.
                self._row_counter = np.delete(self._row_counter, -1)
              
        self._shape = (np.size(self._row_counter) - 1, int(np.max(self._col_index)) + 1)    #Reshapes the matrix and removes any column that is empty
        
        if self._intern_represent == 'CSC':
            self.convert_for_transition() #Updates CSC representation 
        
    @staticmethod
    def toeplitz(n: int):  # Written by Lowe Berg
        """
        Generates a CSR matrix with n rows where the first diagonal is filled with 2 and the adjacent
        diagonals contain -1
        :param n: number of rows in matrix
        :return: SparseMatrix object
        """
        if n < 0:
            raise ValueError("Number of rows must be a positive integer!")

        result = SparseMatrix()  # Create empty object to fill and return

        if n == 1:  # Trivial case
            result._V = np.array([2])
            result._col_index = np.array([0])
            result._row_counter = np.array([0, 1])
            result._number_of_nonzero = 1
            result._intern_represent = 'CSR'
            result._shape = (1,)
            return result

        # Cases where n > 1 are generated procedurally

        temp_V = []
        temp_col_index = []
        temp_row_counter = [0]
        temp_number_of_nonzero = 0

        head = [2, -1]
        spine = [-1, 2, -1]
        tail = [-1, 2]

        columnpointer = 0

        # Create head of matrix
        temp_number_of_nonzero += 2
        temp_V += head
        temp_col_index += [columnpointer, columnpointer + 1]
        temp_row_counter.append(temp_number_of_nonzero)

        for i in range(0, n-2):  # Create spine of matrix
            temp_number_of_nonzero += 3
            temp_V += spine
            temp_col_index += [columnpointer, columnpointer+1, columnpointer+2]
            temp_row_counter.append(temp_number_of_nonzero)
            columnpointer += 1

        # Create tail of matrix
        temp_number_of_nonzero += 2
        temp_V += tail
        temp_col_index += [columnpointer, columnpointer+1]
        temp_row_counter.append(temp_number_of_nonzero)

        result._V = np.array(temp_V)
        result._col_index = np.array(temp_col_index)
        result._row_counter = np.array(temp_row_counter)
        result._number_of_nonzero = np.array(temp_number_of_nonzero)
        result._intern_represent = 'CSR'
        result._shape = (n, n)
        return result

    @staticmethod
    def short_toeplitz(n: int):  # alternative solution. Written by Lowe Berg
        """
            Generates a CSR matrix with n rows where the first diagonal is filled with 2 and the adjacent
            diagonals contain -1. Shorter version of the toeplitz function to perform the same task.
            :param n: number of rows in matrix
            :return: SparseMatrix object
        """
        if n < 0:
            raise ValueError("Number of rows must be a positive integer!")

        result = SparseMatrix()  # Create empty object to fill and return

        if n == 1:  # Trivial but strange case
            result._V = np.array([2])
            result._col_index = np.array([0])
            result._row_counter = np.array([0, 1])
            result._number_of_nonzero = 1
            result._intern_represent = 'CSR'
            result._shape = (1, 1)
            return result

        # Freaky generation method
        temp_V = [2] + [-1, -1, 2] * (n-1)
        temp_col_index = [0] + [x for xs in ([i+1, i, i+1] for i in range(n-1)) for x in xs]
        temp_row_counter = [0] + [2+3*i for i in range(n-1)] + [4+3*(n-2)]
        temp_number_of_nonzero = temp_row_counter[-1]

        result._V = np.array(temp_V)
        result._col_index = np.array(temp_col_index)
        result._row_counter = np.array(temp_row_counter)
        result._number_of_nonzero = np.array(temp_number_of_nonzero)
        result._intern_represent = 'CSR'
        result._shape = (n, n)
        return result
