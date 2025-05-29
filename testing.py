import unittest
import numpy as np
import Sparse


class TaskTests(unittest.TestCase):
    def testtask1(self):
        A = np.array([[0, 0, 0, 0],
                      [5, 8, 0, 0],
                      [0, 0, 3, 0],
                      [0, 6, 0, 0]])
        SA = Sparse.SparseMatrix(A)
        self.assertEqual(SA.number_of_nonzero, 4)
        self.assertIsNone(np.testing.assert_array_equal(SA.V, np.array([5, 8, 3, 6])))
        self.assertIsNone(np.testing.assert_array_equal(SA.col_index, np.array([0, 1, 2, 1])))
        self.assertIsNone(np.testing.assert_array_equal(SA.row_counter, np.array([0, 0, 2, 3, 4])))
        self.assertEqual(SA.intern_represent, 'CSR')

    def testtask3add(self):
        A = np.array([[0, 0, 0, 2, 0],
                      [0, -9, 0, 0, 1],
                      [0, 0, 13, 0, 4],
                      [0, 0, -3, 0, 7],
                      [5, 0, 0, 0, 0]])

        SA = Sparse.SparseMatrix(A)

        SA.edit(8, 0, 1)  # Stoppar in en 8a i position [0, 1]
        self.assertEqual(SA.number_of_nonzero, 9)
        self.assertIsNone(np.testing.assert_array_equal(SA.V, np.array([8, 2, -9, 1, 13, 4, -3, 7, 5])))
        self.assertIsNone(np.testing.assert_array_equal(SA.col_index, np.array([1, 3, 1, 4, 2, 4, 2, 4, 0])))
        self.assertIsNone(np.testing.assert_array_equal(SA.row_counter, np.array([0, 2, 4, 6, 8, 9])))

    def testtask3remove(self):
        A = np.array([[0, 0, 0, 2, 0],
                      [0, -9, 0, 0, 1],
                      [0, 0, 13, 0, 4],
                      [0, 0, -3, 0, 7],
                      [5, 0, 0, 0, 0]])

        SA = Sparse.SparseMatrix(A)

        SA.edit(0, 3, 4)  # Stoppar in en 0a i position [3, 4]
        self.assertEqual(SA.number_of_nonzero, 7)
        self.assertIsNone(np.testing.assert_array_equal(SA.V, np.array([2, -9, 1, 13, 4, -3, 5])))
        self.assertIsNone(np.testing.assert_array_equal(SA.col_index, np.array([3, 1, 4, 2, 4, 2, 0])))
        self.assertIsNone(np.testing.assert_array_equal(SA.row_counter, np.array([0, 1, 3, 5, 6, 7])))

    def testtask3edit(self):
        A = np.array([[0, 0, 0, 2, 0],
                      [0, -9, 0, 0, 1],
                      [0, 0, 13, 0, 4],
                      [0, 0, -3, 0, 7],
                      [5, 0, 0, 0, 0]])

        SA = Sparse.SparseMatrix(A)

        SA.edit(-30, 1, 4)  # Stoppar in -30 i position [1, 4]
        self.assertEqual(SA.number_of_nonzero, 8)
        self.assertIsNone(np.testing.assert_array_equal(SA.V, np.array([2, -9, -30, 13, 4, -3, 7, 5])))
        self.assertIsNone(np.testing.assert_array_equal(SA.col_index, np.array([3, 1, 4, 2, 4, 2, 4, 0])))
        self.assertIsNone(np.testing.assert_array_equal(SA.row_counter, np.array([0, 1, 3, 5, 7, 8])))


if __name__ == '__main__':
    unittest.main()
    print("All tests passed!")
