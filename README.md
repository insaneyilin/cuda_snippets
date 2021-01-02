# cuda_snippets

CUDA code snippets from [*CUDA by Example: An Introduction to General-Purpose GPU Programming*](https://developer.nvidia.com/cuda-example).

[set_get_gpu_info](./set_get_gpu_info): show how to get GPU information.

[vec_add_simple](./vec_add_simple): use 1 block to perform parallel vector addition.

[vec_add_blocks_long](./vec_add_blocks_long): show how to use multiple blocks to implement vector addition on long vectors.

[ripple_animation](./ripple_animation): 2D image example, show how to map (BlockIdx, threadIdx) to pixel position; each thread processes one pixel.

[julia_set](./julia_set): 2D image example, show how to map blockIdx to pixel position; each block process one pixel

[ray_tracing](./ray_tracing): simple ray tracing example, storing scene data with constant memory to improve performance.

[histogram](./histogram): histogram example, "global memory + atomicAdd" or "shared memory + atomicAdd".

[dot_product](./dot_product): show how to use shared memory to perform reduction inside a block.

[dot_product_with_atomic_lock](./dot_product_with_atomic_lock): use atomic_lock and `atomicAdd` to avoid CPU summing in the last step of dot production.

[cuda_stream_example](./cuda_stream_example): cuda stream example; improper usage of cuda streams may not benefit to improve performance.

Tested env:

- Ubuntu 18.04
- NVIDIA-SMI 440.100      Driver Version: 440.100      CUDA Version: 10.2
