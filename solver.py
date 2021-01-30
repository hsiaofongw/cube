import numpy as np
from numpy.core.shape_base import block
import cupy as cp
from helper import Helper
from scipy.sparse.linalg import splu
from math import floor, ceil, sqrt

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

        self.gaussian_elimination_kernel = cp.RawKernel(
            gaussian_elimination_kernel_source,
            'gaussian_elimination'
        )

        self.back_substitution_kernel = cp.RawKernel(
            back_substitution_kernel_source,
            'back_substitution'
        )
    
    def solve(self, coeff_A: cp.ndarray, coeff_B: cp.ndarray) -> np.ndarray:

        coeff_B = cp.asarray(coeff_B)
        coeff_A = cp.asarray(coeff_A)
        coeff_B = cp.atleast_2d(coeff_B).T
        coeff_A = cp.concatenate((coeff_A, coeff_B,), axis=1).astype(cp.float32)
        n_rows = coeff_A.shape[0]
        n_cols = coeff_A.shape[1]
        n_matrices = floor(n_rows/3 + 1e-9)
        BLOCK_SIZE = 16
        block_dim = (BLOCK_SIZE, BLOCK_SIZE,)
        grid_dim = (
            ceil(sqrt(n_matrices/(BLOCK_SIZE*BLOCK_SIZE))), 
            ceil(sqrt(n_matrices/(BLOCK_SIZE*BLOCK_SIZE))), 
        )
        self.gaussian_elimination_kernel(
            grid_dim, 
            block_dim,
            (coeff_A, n_rows, n_cols, n_matrices,)
        )
    
        solutions = cp.zeros(shape=(n_matrices * n_rows,), dtype=cp.float32)
        self.back_substitution_kernel(
            grid_dim,
            block_dim,
            (coeff_A, solutions, n_rows, n_cols, n_matrices,)
        )

        return solutions.get()