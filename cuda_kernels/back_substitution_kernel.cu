extern "C" __global__
void back_substitution(float *global_data, float *global_solution, int n_rows, int n_cols, int n_matrices)
{
    int thread_row = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_col = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_idx = blockDim.x * gridDim.x * thread_row + thread_col;
    
    if (thread_idx >= n_matrices)
    {
        return;
    }
    
    if (sizeof(float) != 4)
    {
        return;
    }
    
    float *data = &(global_data[thread_idx * n_rows * n_cols]);
    float *solution = &(global_solution[thread_idx * n_rows]);

    int i = n_rows - 1;
    while (i >= 0)
    {

        int j = n_cols-2;
        int j_nonzero = j;
        float eps = 1e-8;
        while (j >= 0)
        {
            if (fabsf(data[i * n_cols + j]) > eps)
            {
                j_nonzero = j;
            }
            j -= 1;
        }


        float sum_part = 0.0;
        j = n_cols-2;
        while (j > j_nonzero)
        {
            sum_part = sum_part + data[i * n_cols + j] * solution[j];
            j = j - 1;
        }

        solution[i] = (data[i * n_cols + n_cols-1] - sum_part)/data[i * n_cols + j_nonzero];

        i -= 1;
    }
    
    
}
