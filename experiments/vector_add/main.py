import torch
import triton
import triton.language as tl
import time


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask prevents out-of-bounds memory access
    mask = tid < N
    x = tl.load(x_ptr + tid, mask=mask)
    y = tl.load(y_ptr + tid, mask=mask)

    z = x + y

    tl.store(z_ptr + tid, z, mask=mask)


def vector_add_gpu(x, y):
    N = x.size(0)

    z = torch.empty_like(x)

    BLOCK_SIZE = 256

    # Calculate the grid size (number of blocks)
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # Launch the kernel with the specified grid
    vector_add_kernel[grid](x, y, z, N, BLOCK_SIZE)

    return z


def vector_add_cpu(x, y):
    return x + y


def add_basic():
    N = 8
    x = torch.full((N,), 5, device="cuda")
    y = torch.full((N,), 3, device="cuda")

    z_gpu = vector_add_gpu(x, y)

    z_torch = x + y

    # Verify
    assert torch.allclose(z_gpu, z_torch), "Oh, no!"
    print("All good!")


def add_bench():
    print("Benchmarking...")

    N = 1024 * 1024 * 32
    RUN_COUNT = 10

    x_cpu = torch.randn(N)
    y_cpu = torch.randn(N)

    x_gpu = x_cpu.to("cuda")
    y_gpu = y_cpu.to("cuda")

    # Warm-up
    for _ in range(10):
        _ = x_cpu + y_cpu
        _ = vector_add_gpu(x_gpu, y_gpu)

    # Benchmark CPU
    start_time = time.time()
    for _ in range(RUN_COUNT):
        _ = vector_add_cpu(x_cpu, y_cpu)
    cpu_time = (time.time() - start_time) / RUN_COUNT

    # Benchmark GPU with Triton
    start_time = time.time()
    for _ in range(RUN_COUNT):
        _ = vector_add_gpu(x_gpu, y_gpu)
    gpu_time = (time.time() - start_time) / RUN_COUNT

    print(f"Vector size: {N}")
    print(f"CPU time: {cpu_time:.6f} seconds")
    print(f"GPU time (Triton): {gpu_time:.6f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")


if __name__ == "__main__":
    add_basic()
    add_bench()

    print("Success!")
