extern "C" __global__
void gaussian_elimination(float *global_data, int n_rows, int n_cols, int n_matrices)
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
    
    int pivot_row = 0;
    int pivot_col = 0;
    float eps = 1e-10;
    while ((pivot_row <= (n_rows-1)) && (pivot_col <= (n_cols-1)))
    {
        int i_max = pivot_row;
        for (int i = i_max; i < n_rows; ++i)
        {
            if (data[i * n_cols + pivot_col] >= data[i_max * n_cols + pivot_col])
            {
                i_max = i;
            }
        }

        if (fabsf(data[i_max * n_cols + pivot_col] - 0) < eps)
        {
            pivot_col += 1;
        }
        else
        {
            for (int j = 0; j < n_cols; ++j)
            {
                float temp = data[pivot_row * n_cols + j];
                data[pivot_row * n_cols + j] = data[i_max * n_cols + j];
                data[i_max * n_cols + j] = temp;
            }

            for (int i = pivot_row+1; i < n_rows; ++i)
            {
                float factor = data[i * n_cols + pivot_col] / data[pivot_row * n_cols + pivot_col];
                data[i * n_cols + pivot_col] = 0;

                for (int j = pivot_col+1; j < n_cols; j++)
                {
                    data[i * n_cols + j] = data[i * n_cols + j] - data[pivot_row * n_cols + j] * factor;
                }
            }

            pivot_row += 1;
            pivot_col += 1;
        }
    }
}