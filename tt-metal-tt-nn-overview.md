# TT-Metal and TTNN Documentation

## Introduction

TT-Metal (TT-Metalium) and TTNN are powerful frameworks developed by Tenstorrent for programming their AI accelerator hardware, specifically designed for high-performance neural network and tensor operations. TT-Metalium is a low-level programming model that enables kernel development directly on Tenstorrent hardware, providing fine-grained control over the Tensix processor architecture. TTNN is a high-level Python and C++ neural network operations library built on top of TT-Metalium, offering a PyTorch-like interface for tensor operations with seamless device management and hardware acceleration.

The framework targets Tenstorrent's unique architecture featuring Tensix cores - specialized compute nodes with 5 RISC-V CPUs, dedicated matrix (FPU) and vector (SFPU) units, 1.5MB of local SRAM, and Network-on-Chip (NoC) interfaces for efficient data movement. The architecture operates natively on 32×32 tiles optimized for deep learning workloads. TTNN abstracts this complexity while providing access to the hardware's full performance potential, supporting multiple device configurations including single-chip (Grayskull, Wormhole, Blackhole), multi-chip systems (QuietBox, Galaxy), and distributed training setups. The framework is Apache 2.0 licensed and includes pre-optimized implementations of popular models like Llama, Whisper, ResNet, and BERT.

## APIs and Key Functions

### Device Management

Opening and managing Tenstorrent devices for tensor operations.

```python
import ttnn

# Open a device by ID
device_id = 0
device = ttnn.open_device(device_id=device_id)

# Perform operations on the device
torch_tensor = torch.rand(64, 128, dtype=torch.float32)
ttnn_tensor = ttnn.from_torch(
    torch_tensor,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# Close the device when done
ttnn.close_device(device)
```

### Tensor Conversion from PyTorch

Converting PyTorch tensors to TTNN tensors with device placement and data type specification.

```python
import torch
import ttnn

# Basic conversion without device
torch_input = torch.zeros(2, 4, dtype=torch.float32)
ttnn_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)

# Conversion with device and layout
device = ttnn.open_device(device_id=0)
torch_tensor = torch.rand(4, 7, dtype=torch.float32)
ttnn_tensor = ttnn.from_torch(
    torch_tensor,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# Convert back to PyTorch
torch_output = ttnn.to_torch(ttnn_tensor)
print(f"Output shape: {torch_output.shape}, dtype: {torch_output.dtype}")
ttnn.close_device(device)
```

### Matrix Multiplication

Performing matrix multiplication operations with various batch dimensions and configurations.

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Simple 2D matrix multiplication
tensor_a = torch.rand(64, 32, dtype=torch.float32)
tensor_b = torch.rand(32, 128, dtype=torch.float32)

ttnn_a = ttnn.from_torch(tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_b = ttnn.from_torch(tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Using @ operator
result = ttnn_a @ ttnn_b
torch_result = ttnn.to_torch(result)
print(f"Matmul result shape: {torch_result.shape}")

# Using ttnn.matmul function
result_alt = ttnn.matmul(ttnn_a, ttnn_b)

# Batched matrix multiplication
batch_a = torch.rand(10, 64, 32, dtype=torch.float32)
batch_b = torch.rand(10, 32, 128, dtype=torch.float32)
ttnn_batch_a = ttnn.from_torch(batch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_batch_b = ttnn.from_torch(batch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
batch_result = ttnn_batch_a @ ttnn_batch_b
print(f"Batch matmul shape: {ttnn.to_torch(batch_result).shape}")

ttnn.close_device(device)
```

### Element-wise Operations

Applying element-wise mathematical functions to tensors on device.

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Create input tensor
torch_input = torch.rand(4, 7, dtype=torch.float32)
input_tensor = ttnn.from_torch(
    torch_input,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# Apply exponential function
exp_output = ttnn.exp(input_tensor)
torch_exp_result = ttnn.to_torch(exp_output)
print(f"Exp output:\n{torch_exp_result}")

# Other element-wise operations supported:
# ttnn.relu, ttnn.sigmoid, ttnn.tanh, ttnn.sqrt, ttnn.log, etc.

ttnn.close_device(device)
```

### Linear Transformation

Performing linear layer operations with weight matrices and optional bias.

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Define layer dimensions
batch_size, seq_len, input_dim = 32, 128, 512
output_dim = 768

# Create input activations
activations = torch.rand(batch_size, seq_len, input_dim, dtype=torch.float32)
ttnn_activations = ttnn.from_torch(
    activations,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# Create weight matrix (transposed for linear layer)
weight = torch.rand(output_dim, input_dim, dtype=torch.float32)
ttnn_weight = ttnn.from_torch(
    weight,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# Optional bias
bias = torch.rand(output_dim, dtype=torch.float32)
ttnn_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16, device=device)

# Perform linear transformation
output = ttnn.linear(ttnn_activations, ttnn_weight, bias=ttnn_bias)
torch_output = ttnn.to_torch(output)
print(f"Linear output shape: {torch_output.shape}")

ttnn.close_device(device)
```

### Tensor Device Movement

Moving tensors between host memory and device memory.

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Create tensor on host
torch_tensor = torch.rand(128, 256, dtype=torch.float32)
ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16)

# Move tensor to device
tensor_on_device = ttnn.to_device(ttnn_tensor, device=device)
print(f"Tensor is on device: {tensor_on_device.is_on_device()}")

# Perform operations on device
result = ttnn.exp(tensor_on_device)

# Move tensor back to host
tensor_on_host = ttnn.from_device(result)
torch_result = ttnn.to_torch(tensor_on_host)
print(f"Result shape: {torch_result.shape}")

ttnn.close_device(device)
```

### Sparse Matrix Multiplication

Performing sparse matrix operations with sparsity masks and custom configurations.

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Create dense input tensors
m, k, n = 1024, 512, 2048
tensor_a = torch.rand(m, k, dtype=torch.float32)
tensor_b = torch.rand(k, n, dtype=torch.float32)

ttnn_a = ttnn.from_torch(tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_b = ttnn.from_torch(tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Create sparsity bitmask (1 = keep, 0 = sparse)
sparsity_mask = torch.randint(0, 2, (m // 32, k // 32), dtype=torch.int32)
ttnn_sparsity = ttnn.from_torch(sparsity_mask, device=device)

# Configure sparse matmul
config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=2,
    out_subblock_h=4,
    out_subblock_w=2,
    per_core_M=4,
    per_core_N=4,
)

# Perform sparse matrix multiplication
sparse_result = ttnn.sparse_matmul(
    ttnn_a,
    ttnn_b,
    sparsity=ttnn_sparsity,
    is_input_a_sparse=True,
    is_input_b_sparse=False,
    program_config=config
)

torch_sparse_result = ttnn.to_torch(sparse_result)
print(f"Sparse matmul result shape: {torch_sparse_result.shape}")

ttnn.close_device(device)
```

### Random Tensor Generation

Creating random tensors directly on device for testing and initialization.

```python
import ttnn

device = ttnn.open_device(device_id=0)

# Generate random tensor on device
random_tensor = ttnn.rand(
    (64, 128, 256),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# Use in operations
scaled_tensor = random_tensor * 0.5
result = ttnn.exp(scaled_tensor)

torch_result = ttnn.to_torch(result)
print(f"Random tensor result shape: {torch_result.shape}")
print(f"Mean: {torch_result.mean():.4f}, Std: {torch_result.std():.4f}")

ttnn.close_device(device)
```

### Configuration Management

Managing TTNN runtime configuration and performance settings.

```python
import ttnn
import os

# Set configuration via environment variable before import
os.environ['TTNN_CONFIG_OVERRIDES'] = '{"enable_fast_runtime_mode": true, "enable_logging": false}'

# Access configuration
print(f"Cache path: {ttnn.CONFIG.cache_path}")
print(f"Throw exception on fallback: {ttnn.CONFIG.throw_exception_on_fallback}")

# Temporarily modify configuration
with ttnn.manage_config("enable_logging", True):
    # Operations here will have logging enabled
    device = ttnn.open_device(device_id=0)
    tensor = ttnn.rand((32, 64), dtype=ttnn.bfloat16, device=device)
    result = ttnn.exp(tensor)
    ttnn.close_device(device)
# Logging automatically restored to previous state

# Save current configuration
config_path = "/tmp/ttnn_config.json"
ttnn.save_config_to_json_file(config_path)
print(f"Configuration saved to {config_path}")
```

### Multi-Device Tensor Distribution

Distributing tensors across multiple devices for parallel computation.

```python
import torch
import ttnn

# Open multiple devices
device_ids = [0, 1, 2, 3]
devices = [ttnn.open_device(device_id=device_id) for device_id in device_ids]
mesh_device = ttnn.MeshDevice(device_ids)

# Create tensor to distribute
batch_size, hidden_dim = 128, 2048
torch_tensor = torch.rand(batch_size, hidden_dim, dtype=torch.float32)

# Distribute tensor with sharding strategy
distributed_tensor = ttnn.distribute_tensor(
    ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16),
    mesh_device,
    placement=ttnn.PlacementShard(dim=0)  # Shard along batch dimension
)

# Perform operations on distributed tensor
result = ttnn.matmul(distributed_tensor, distributed_tensor.transpose(-2, -1))

# Aggregate results back
aggregated = ttnn.aggregate_tensor(result, mesh_device)
torch_result = ttnn.to_torch(aggregated)
print(f"Distributed computation result shape: {torch_result.shape}")

# Close devices
for device in devices:
    ttnn.close_device(device)
```

### TT-Metalium Kernel Programming

Writing low-level data movement and compute kernels for custom operations.

```cpp
// Data movement kernel (reader)
#include <dataflow_api.h>

void kernel_main() {
    // Get runtime arguments
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    // Configure circular buffer for output
    constexpr uint32_t cb_id = 0;

    // Read tiles from DRAM to circular buffer
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        uint32_t write_addr = get_write_ptr(cb_id);
        noc_async_read_tile(i, input_addr, write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}

// Compute kernel
#include <compute_kernel_api.h>

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;

    // Initialize compute engine
    init_sfpu(cb_in);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // Wait for input tile
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        // Acquire destination register
        acquire_dst();

        // Perform computation (e.g., exponential)
        exp_tile_init();
        exp_tile(cb_in);

        // Pack result to output circular buffer
        pack_tile(0, cb_out);
        release_dst();

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }
}

// Writer kernel
#include <dataflow_api.h>

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = 16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t read_addr = get_read_ptr(cb_id);
        noc_async_write_tile(i, read_addr, output_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
```

### Trace Capture and Replay

Capturing and replaying operation sequences for optimized repeated execution.

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Prepare inputs
input_tensor = ttnn.from_torch(
    torch.rand(32, 64, dtype=torch.float32),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# Begin trace capture
trace_id = ttnn.begin_trace_capture(device)

# Operations to capture
temp1 = ttnn.exp(input_tensor)
temp2 = ttnn.sqrt(temp1)
output = temp2 * 2.0

# End trace capture
ttnn.end_trace_capture(device, trace_id)

# Execute traced operations (much faster for repeated execution)
for iteration in range(100):
    # Update input if needed
    input_tensor = ttnn.from_torch(
        torch.rand(32, 64, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device
    )

    # Execute entire captured sequence
    ttnn.execute_trace(device, trace_id)

# Release trace when done
ttnn.release_trace(device, trace_id)
ttnn.close_device(device)
```

## Summary and Integration

TT-Metal and TTNN provide a comprehensive software stack for leveraging Tenstorrent's specialized AI accelerator hardware, offering both high-level Python APIs for rapid development and low-level kernel programming for maximum performance. The primary use cases include training and inference of large language models (LLMs like Llama 3.3 70B, Qwen 2.5), computer vision models (ResNet50, ViT, YOLO), audio processing (Whisper), and custom neural network architectures. The framework excels at batched inference workloads with its native 32×32 tile processing, achieving high throughput through efficient data movement patterns and hardware utilization. Users can start with the high-level TTNN Python API for PyTorch-like tensor operations and progressively optimize by writing custom TT-Metalium kernels when needed.

Integration patterns follow a typical workflow: convert PyTorch models and data to TTNN format using `ttnn.from_torch()`, move tensors to device with `ttnn.to_device()`, execute operations on the specialized hardware, and convert results back with `ttnn.to_torch()`. The framework supports multi-device configurations for scaling to larger models through tensor and pipeline parallelism, with built-in support for mesh topologies across 8-device QuietBox and 32-device Galaxy systems. Advanced features include trace capture for optimizing repeated execution, sparse operations for efficiency, and fine-grained memory configuration for managing the 1.5MB per-core SRAM. The ecosystem includes debugging tools (Watcher, Inspector, DPRINT), profiling with Tracy, visualization tools, and pre-optimized model implementations that demonstrate best practices for achieving peak hardware performance.