import unittest
import numpy as np
import Sparse


class TestTask1(unittest.TestCase):
    def test1(self):
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


if __name__ == '__main__':
    unittest.main()
    print("All tests passed!")
