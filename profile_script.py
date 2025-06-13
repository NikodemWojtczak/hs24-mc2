import numpy as np
import numba
from numba import cuda
import math
import time

# --- KERNEL DEFINITIONS (Copied from your notebook) ---

TILE_DIM = 16


@cuda.jit
def reconstruct_svd_numba_kernel(reco, u, s, vt, k):
    """Basic CUDA kernel for SVD reconstruction."""
    i, j = cuda.grid(2)
    m, n = reco.shape
    if i < m and j < n:
        temp_sum = 0.0
        for l in range(k):
            temp_sum += u[i, l] * s[l] * vt[l, j]
        reco[i, j] = temp_sum


@cuda.jit
def reconstruct_svd_shared_mem_kernel(reco, u, s, vt, k):
    """SVD reconstruction kernel optimized with shared memory."""
    s_u = cuda.shared.array(shape=(TILE_DIM, TILE_DIM), dtype=numba.float32)
    s_vt = cuda.shared.array(shape=(TILE_DIM, TILE_DIM), dtype=numba.float32)
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tmp = 0.0
    num_tiles = math.ceil(k / TILE_DIM)
    for i in range(num_tiles):
        tile_col_u = i * TILE_DIM + ty
        tile_row_vt = i * TILE_DIM + tx
        if x < u.shape[0] and tile_col_u < k:
            s_u[tx, ty] = u[x, tile_col_u] * s[tile_col_u]
        else:
            s_u[tx, ty] = 0.0
        if y < vt.shape[1] and tile_row_vt < k:
            s_vt[tx, ty] = vt[tile_row_vt, y]
        else:
            s_vt[tx, ty] = 0.0
        cuda.syncthreads()
        for j in range(TILE_DIM):
            tmp += s_u[tx, j] * s_vt[j, ty]
        cuda.syncthreads()
    if x < reco.shape[0] and y < reco.shape[1]:
        reco[x, y] = tmp


# --- MAIN EXECUTION ---
def main():
    if not cuda.is_available():
        print("GPU not available. Exiting.")
        return

    matrix_size = 1024
    k = matrix_size
    print(f"Setting up for profiling with a {matrix_size}x{matrix_size} matrix...")

    # Prepare data on the GPU
    u_gpu = cuda.to_device(np.random.rand(matrix_size, k).astype(np.float32))
    s_gpu = cuda.to_device(np.random.rand(k).astype(np.float32))
    vt_gpu = cuda.to_device(np.random.rand(k, matrix_size).astype(np.float32))
    reco_gpu = cuda.device_array((matrix_size, matrix_size), dtype=np.float32)

    threads_per_block = (TILE_DIM, TILE_DIM)
    blocks_per_grid_x = math.ceil(matrix_size / threads_per_block[0])
    blocks_per_grid_y = math.ceil(matrix_size / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # --- Profile the Basic Kernel ---
    print("Running Basic Kernel...")
    reconstruct_svd_numba_kernel[blocks_per_grid, threads_per_block](
        reco_gpu, u_gpu, s_gpu, vt_gpu, k
    )
    cuda.synchronize()  # Wait for kernel to finish

    # --- Profile the Shared Memory Kernel ---
    print("Running Shared Memory Kernel...")
    reconstruct_svd_shared_mem_kernel[blocks_per_grid, threads_per_block](
        reco_gpu, u_gpu, s_gpu, vt_gpu, k
    )
    cuda.synchronize()  # Wait for kernel to finish

    print("Profiling script finished.")


if __name__ == "__main__":
    main()
