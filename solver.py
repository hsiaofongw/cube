import numpy as np
import cupy as cp
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

class CUDAMultipleEquationsSolver:

    def __init__(self):

        gaussian_elimination_kernel_source = str()
        back_substitution_kernel_source = str()

        with open('./cuda_kernels/gaussian_elimination_kernel.cu', 'r') as f:
            gaussian_elimination_kernel_source = f.read()
        
        with open('./cuda_kernels/back_substitution_kernel.cu', 'r') as f:
            back_substitution_kernel_source = f.read()

        gaussian_elimination_kernel = cp.RawKernel(
            gaussian_elimination_kernel_source,
            'gaussian_elimination'
        )

        back_substitution_kernel = cp.RawKernel(
            back_substitution_kernel_source,
            'back_substitution'
        )