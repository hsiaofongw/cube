import numpy as np
from helper import Helper
from scipy.sparse.linalg import splu

class LUMultipleEquationsSolver:

    def solve(
        self,
        coeff_A: np.ndarray,
        coeff_B: np.ndarray
    ) -> np.ndarray:

        coeff_A = Helper.make_block_diagonal(coeff_A, 3)
        coeff_A = coeff_A.tocsc()

        solver = splu(coeff_A)
        solution = solver.solve(coeff_B)

        return solution
