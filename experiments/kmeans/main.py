import numpy as np
import time
import torch
import triton
import triton.language as tl


def kmeans_cpu(X, k, iterations):
    # Random initialization of centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(iterations):
        # Compute squared distances and assign clusters
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        # Update centroids
        for j in range(k):
            if np.any(labels == j):
                centroids[j] = X[labels == j].mean(axis=0)

    return centroids, labels


@triton.jit
def kmeans_kernel(
    points_ptr,
    centroids_ptr,
    result_ptr,
    point_count: tl.constexpr,
    centroid_count: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current program ID (block index)
    pid = tl.program_id(0)
    # Calculate starting index for this block
    start = pid * BLOCK_SIZE

    # Create array of offsets for parallel processing
    offsets = start + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid points in memory
    mask = offsets < point_count

    # Initialize arrays for tracking best distances and indices
    best_distances = tl.full([BLOCK_SIZE], float("inf"), dtype=tl.float32)
    best_indices = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for j in range(centroid_count):
        # Initialize sum array for distance calculation
        sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for i in range(d):
            # Load point coordinate
            point = tl.load(points_ptr + offsets * d + i, mask=mask)
            # Load centroid coordinate
            centroid = tl.load(centroids_ptr + j * d + i)
            # Calculate difference and add squared to sum
            diff = point - centroid
            sum += diff * diff

        # Update current best distances and indices
        is_best = sum < best_distances
        best_distances = tl.where(is_best, sum, best_distances)
        best_indices = tl.where(is_best, j, best_indices)

    # Store the best centroid index for each point
    tl.store(result_ptr + offsets, best_indices, mask=mask)


def kmeans_gpu(X, k, epochs, BLOCK_SIZE=1024):
    point_count, d = X.shape

    # Transfer data to GPU
    X_t = torch.tensor(X, dtype=torch.float32, device="cuda")

    # Initialize centroids randomly
    indices = torch.randperm(point_count)[:k]
    centroids_t = X_t[indices].clone()
    result_t = torch.empty(point_count, dtype=torch.int32, device="cuda")

    for _ in range(epochs):
        grid = ((point_count + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        # Launch Triton kernel
        kmeans_kernel[grid](
            X_t, centroids_t, result_t, point_count, k, d, BLOCK_SIZE=BLOCK_SIZE
        )
        # Update centroids on GPU
        for j in range(k):
            mask = result_t == j
            if mask.sum() > 0:
                centroids_t[j] = X_t[mask].mean(dim=0)

    return centroids_t.cpu().numpy(), result_t.cpu().numpy()


def main():
    # Generate random data: 10k points in 32 dimensions
    np.random.seed(42)
    X = np.random.rand(20_000, 64).astype(np.float32)
    k = 10
    iterations = 500

    # CPU Benchmark
    start_time = time.time()
    _centroids_cpu, _labels_cpu = kmeans_cpu(X, k, iterations)
    cpu_time = time.time() - start_time

    # GPU Benchmark
    start_time = time.time()
    _centroids_gpu, _labels_gpu = kmeans_gpu(X, k, iterations)
    gpu_time = time.time() - start_time

    print(f"CPU Time: {cpu_time:.4f} sec")
    print(f"GPU Time: {gpu_time:.4f} sec")


if __name__ == "__main__":
    main()
