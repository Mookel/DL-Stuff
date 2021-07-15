- [1. Docs](#1-docs)
- [2. VTA Introduction](#2-vta-introduction)
  * [2.1 VTA Hardware-Software Stack Overview](#21-vta-hardware-software-stack-overview)
  * [2.2 VTA Architecture and JIT Runtime](#22-vta-architecture-and-jit-runtime)
    + [2.2.1 VTA Architecture](#221-vta-architecture)
    + [2.2.2 JIT Runtime](#222-jit-runtime)
- [3. Tutorials](#3-tutorials)
  * [3.1 Get Started with VTA](#31-get-started-with-vta)
  * [3.2 Simple GEMM](#32-simple-gemm)
    + [Tiling计算示意图](#tiling计算示意图)
    + [Schedule步骤](#schedule步骤)
    + [改进](#改进)
    + [遗留issue](#遗留issue)
  * [3.3 Matrix Multiply Blocking](#33-matrix-multiply-blocking)
    + [Blocking分类](#blocking--)
    + [On-Chip Buffer Blocking Scheduling步骤](#on-chip-buffer-blocking-scheduling--)
  * [3.4 2D Convolution Optimization](#34-2d-convolution-optimization)
  * [3.5 End2End](#35-end2end)
- [4. TVM Support](#4-tvm-support)
  * [4.1 VTA Low-Level Build Flow](#41-vta-low-level-build-flow)
    + [4.1.1 函数入口和build_config](#411-函数入口和build_config)
    + [4.1.2 VTA Low-Level Build Flow](#412-vta-low-level-build-flow)
    + [4.1.3 codegen.build_module实现](#413-codegenbuild-module实现)
    + [4.1.4 相关问题](#414-相关问题)
    + [4.1.5 总结](#415-总结)
  * [4.2 VTA Runtime](#42-vta-runtime)
    + [4.2.1 runtime](#421-runtime)
    + [4.2.2 device](#422-device)
  * [4.3 Relay + VTA build Flow](#43-relay-+-vta-build-flow)
  * [4.4 Tensorization](#44-tensorization)
- [5 Appenix](#5-appenix)
  * [Lower实现](#lower实现)
  * [Reference](#reference)

# 1. Docs

- VTA Technical Report: https://tvm.apache.org/2018/07/12/vta-release-announcement.html
- VTA Install Guide: https://tvm.apache.org/docs/vta/install.html

- VTA Hardware Guide: https://tvm.apache.org/docs/vta/dev/hardware.html

# 2. VTA Introduction

## 2.1 VTA Hardware-Software Stack Overview

![image-20210709095647687](\VTA\vta hardware-software stack.png)

- **Framework**: Support different DL frameworks.
- **Relay Graph Optimizer**: When targeting VTA we quantize inputs to match VTA’s low precision data types, transform data layout, maximize data reuse, and transform input and weight data layouts to utilize VTA’s tensor intrinsics.
- **TVM Operator Optimizer**: Automates the tedious process of scheduling workloads onto VTA accelerator variants. Scheduling is important for multiple reasons. First, it tiles the computation to maximize data reuse. Second, it inserts thread parallelism that VTA’s runtime can translate into task-level pipeline parallelism. Third, it partitions operators into sub-computations which can be mapped to high-level hardware intrinsics such as bulk DMA load or GEMM.
- **JIT Compiler and Runtime**: The runtime performs JIT compilation of the accelerator binaries and manages heterogeneous execution between the CPU and VTA.
- **Hardware architecture**: VTA is a parameterizable accelerator and VTA is explicitly programmed by the compiler stack using a two-level programming interface.

## 2.2 VTA Architecture and JIT Runtime

### 2.2.1 VTA Architecture 

![hardware](.\VTA\hardware.PNG)

- Architecture

  - VTA is composed of four modules: fetch, load, compute, and store.
  - The VTA architecture is fully parameterizable, including:
    - the shape of the GEMM tensor intrinsic;
    - shape of the input, weight, and accumulator tensors;
    - each data type can customized to a different integer precision: weight and input types can be 8-bits or fewer, while the accumulation type can be 32-bits or fewer.
  - ***Task-Level Pipeline Parallelism (TLPP)*** is an important feature in the VTA architecture,because it enables simultaneous use of compute and memory resources to maximize their utilization. TLPP is based on the paradigm of ***access-execute decoupling***. To extract TLPP, we partition tasks into two mutually-exclusive execution contexts, so that concurrent load, compute, and store operations do not interfere with one another. This partitioning is easily achieved in TVM using ***virtual threads***. To guarantee timely and correct execution for decoupled access-execute instruction streams, we encode dependency information into instructions.
  - Two functional units perform operations on the register file: the tensor ALU and the GEMM core. The tensor ALU performs element-wise tensor operations such as addition, activation, normalization, and pooling tasks. The GEMM core performs high-arithmetic intensity matrix multiplication over input and weight tensors.
  - The GEMM core defines a low-level tensor hardware intrinsic which is exposed to the TVM compiler stack. TVM uses tensorization to map deep learning operators such as 2d convolution down to fixed tensor hardware intrinsics.

- ISAs

  - **Task-Level ISA:** VTA supports a high-level task ISA that encodes multi-cycle compute and memory operations, including **LOAD, GEMM, ALU**, and **STORE** instructions. LOAD and STORE instructions describe how data from DRAM is loaded and stored into on-chip SRAMs. Strided memory access is supported to load tensor tiles without modifying memory layout. GEMM and ALU instructions invoke micro-coded kernels, based on micro-op instructions, which describe the data-access patterns that define a given deep learning operator.
  - **Microcode ISA**: The compute core reads instructions from the micro-op cache, which describe how computation is performed over data. These micro-ops provide no control flow. Therefore, instructions need to be unrolled to express repeatable data access stencils. To minimize the footprint of micro-op kernels while avoiding the need for control-flow instructions, the compute core executes micro-op sequences inside a two-level nested loop that computes the location of each tensor register via an affine function.

  ![Gemm_core](\VTA\Gemm_core.PNG)

- Instructions Format

![Instructions_format](\VTA\Instructions_format.PNG)

### 2.2.2 JIT Runtime

- The JIT runtime design follows five objectives: (1) enable heterogeneous execution, (2) lower compiler design complexity, (3) overcome physical limitations, (4) reduce binary bloat, (5) future proofing.
- Compiler Design: By adding a level of indirection, code JIT-ting eliminates the need to write compiler code-generation backends which can be tedious to maintain for different programmable accelerators. ***The JIT compiler exposes a high-level API to TVM to lower schedules onto, abstracting away VTA variant-specific architectural details.*** This lets us extend the TVM compiler support we built for VTA to cover future variants of different shapes and sizes.
- Physical Limitations: ***The JIT runtime generates and manages micro-kernels on the fly. It controls when to load kernels from DRAM into the accelerator limited micro-op cache. This eliminates micro-op memory physical limitations and lets us support large models, even if all micro-kernels for all layers do not fit in SRAM all at once***. It also lets us trade area used by the micro-op cache for other resources such as data storage, or compute units.
- Binary bloat: Delaying micro-kernel generation to the JIT compilation stage minimizes binary bloat. Since VTA’s architecture has limited support for control flow, micro-kernels have to be unrolled which can produce fairly large binaries. ***In addition, micro-kernel code JIT-ting expresses binaries for heterogeneous execution in a single-ISA: instead of shipping a hybrid binary, we just ship one CPU binary that performs accelerator binary JIT-ting***.

# 3. Tutorials

## 3.1 Get Started with VTA

- LINK: https://tvm.apache.org/docs/vta/tutorials/vta_get_started.html

## 3.2 Simple GEMM

- LINK: https://tvm.apache.org/docs/vta/tutorials/matrix_multiply.html

### Tiling计算示意图

![blocking_figure](\VTA\blocking_figure.jpeg)

![blocking_code](\VTA\blocking_code.jpeg)

### Schedule步骤

- 未经schedule的原始计算

```c++
# Outer input feature reduction axis
ko = te.reduce_axis((0, n), name="ko")
# Inner input feature reduction axis
ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
# Describe the in-VTA matrix multiplication
C_buf = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT),
    lambda bo, co, bi, ci: te.sum(
        A_buf[bo, ko, bi, ki].astype(env.acc_dtype) * B_buf[co, ko, ci, ki].astype(env.acc_dtype),
        axis=[ko, ki],
    ),
    name="C_buf",
)

# Cast to output type, and send to main memory
C = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: C_buf(*i).astype(env.inp_dtype), name="C"
)

# Let's take a look at the generated schedule
s = te.create_schedule(C.op)
print("original=>\n", tvm.lower(s, [A, B, C], simple_mode=True))
// ----------------------------------------------------
// 生成的代码如下：
// 注：bo和bi维度都是1，所以被忽略了。
for (co: int32, 0, 16) {   //co : 0 ~ m
      for (ci: int32, 0, 16) { // ci : 0 ~ env.BLOCK_OUT
        C_buf[((co*16) + ci)] = 0
        // ko和ki是reduce axis；
        for (ko: int32, 0, 16) { // ko : 0 ~ n
          for (ki: int32, 0, 16) { // ki: 0 ~ env.BLOCK_IN
            C_buf[((co*16) + ci)] = ((int32*)C_buf[((co*16) + ci)] + (cast(int32, (int8*)A_buf[((ko*16) + ki)])*cast(int32, (int8*)B_buf[((((co*4096) + (ko*256)) + (ci*16)) + ki)])))
          }
        }
      }
    }
    for (i1_2: int32, 0, 16) {
      for (i3_2: int32, 0, 16) {
        C_2[((i1_2*16) + i3_2)] = cast(int8, (int32*)C_buf[((i1_2*16) + i3_2)])
      }
    }
```

- 设置buffer范围
  - 这里是关键，如果设置不对可能导致A_buf和B_buf过大而产生on-chip buffer不够的问题，因为这个demo里面只用到了on-chip tensor blocking；
  - 虽然这个schedule可以避免A_buf和B_buf的问题，但是C_buf仍然可能超过on-chip buffer的限制（例如把m和n都设置为10240，编译就会出错）；

```c++
# Set the intermediate tensor's scope to VTA's on-chip buffers
s[A_buf].set_scope(env.inp_scope)
s[B_buf].set_scope(env.wgt_scope)
s[C_buf].set_scope(env.acc_scope)

# Move buffer copy into matrix multiply loop
s[A_buf].compute_at(s[C_buf], ko)
s[B_buf].compute_at(s[C_buf], ko)

# Tag the buffer copies with the DMA pragma to insert a DMA transfer
s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
s[C].pragma(s[C].op.axis[0], env.dma_copy)

// ----------------------------------------------------
// 生成的代码如下：
  attr [C_buf: Pointer(int32)] "storage_scope" = "local.acc_buffer";
  allocate(C_buf, int32, [256]);
  attr [A_buf: Pointer(int8)] "storage_scope" = "local.inp_buffer";
  allocate(A_buf, int8, [16]);
  attr [B_buf: Pointer(int8)] "storage_scope" = "local.wgt_buffer";
  allocate(B_buf, int8, [16]) {
  for (co: int32, 0, 16) { // co : 0 ~ m
      for (ci: int32, 0, 16) { // ci : 0 ~ env.BATCH_OUT
        C_buf[((co*16) + ci)] = 0
        for (ko: int32, 0, 16) {  // ko : 0 ~ n
          attr [IterVar(i0: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
          /// 设置A_buf的compute地点；
          for (i3: int32, 0, 16) {  
            A_buf[i3] = (int8*)A_2[((ko*16) + i3)]
          }
          attr [IterVar(i0_1: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
          /// 设置B_buf的comptue地点；
          for (i3_1: int32, 0, 16) {
            B_buf[i3_1] = (int8*)B_2[((((co*4096) + (ko*256)) + (ci*16)) + i3_1)]
          }
          for (ki: int32, 0, 16) {  // ki : 0 ~ env.BATCH_IN
            C_buf[((co*16) + ci)] = ((int32*)C_buf[((co*16) + ci)] + (cast(int32, (int8*)A_buf[ki])*cast(int32, (int8*)B_buf[ki])))
          }
        }
      }
    }
    attr [IterVar(i0_2: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
    for (i1: int32, 0, 16) {
      for (i3_2: int32, 0, 16) {
        C_2[((i1*16) + i3_2)] = cast(int8, (int32*)C_buf[((i1*16) + i3_2)])
      }
    }
```

- reorder, 为tensorization做准备： co, ci, ko, ki ==> ko, co, ci, ki

```c++
# ko, bo, co, bi, ci, ki
s[C_buf].reorder(
    ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1], s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki
)
print("after reorder==>\n", tvm.lower(s, [A, B, C], simple_mode=True))
// ----------------------------------------------------
// 生成的代码如下：
    for (co.init: int32, 0, 1024) {
      for (ci.init: int32, 0, 16) {
        C_buf[((co.init*16) + ci.init)] = 0
      }
    }
    for (ko: int32, 0, 1024) {  // ko : 0 ~ n
      attr [IterVar(i0: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
      for (i3: int32, 0, 16) {
        A_buf[i3] = (int8*)A_2[((ko*16) + i3)]
      }
      attr [IterVar(i0_1: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
      for (i0_1, 0, 1024) {
        for (i2: int32, 0, 16) {
          for (i3_1: int32, 0, 16) {
            B_buf[(((i0_1*256) + (i2*16)) + i3_1)] = (int8*)B_2[((((i0_1*262144) + (ko*256)) + (i2*16)) + i3_1)]
          }
        }
      }
      for (co: int32, 0, 1024) { // co : 0 ~ m
        for (ci: int32, 0, 16) {  // ci : 0 ~ env.BATCH_OUT
          for (ki: int32, 0, 16) { // Ki : 0 ~ env.BATCH_IN
            C_buf[((co*16) + ci)] = ((int32*)C_buf[((co*16) + ci)] + (cast(int32, (int8*)A_buf[ki])*cast(int32, (int8*)B_buf[(((co*256) + (ci*16)) + ki)])))
          }
        }
      }
    }
    attr [IterVar(i0_2: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
    for (i1: int32, 0, 1024) {
      for (i3_2: int32, 0, 16) {
        C_2[((i1*16) + i3_2)] = cast(int8, (int32*)C_buf[((i1*16) + i3_2)])
      }
    }
```

- Tensorization

```c++
// bo, co, bi, ci
// 从ci开始？
s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)
// ----------------------------------------------------
// 生成的代码如下：
 primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(int8), int8, [1, 1024, 1, 16], []),
             A: Buffer(A_2: Pointer(int8), int8, [1, 1024, 1, 16], []),
             B: Buffer(B_2: Pointer(int8), int8, [1024, 1024, 16, 16], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  attr [C_buf: Pointer(int32)] "storage_scope" = "local.acc_buffer";
  attr [A_buf: Pointer(int8)] "storage_scope" = "local.inp_buffer";
  attr [B_buf: Pointer(int8)] "storage_scope" = "local.wgt_buffer" {
    attr [IterVar(vta: int32, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 2 {
      attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushGEMMOp" {
        @tir.call_extern("VTAUopLoopBegin", 1024, 1, 0, 0, dtype=int32)
        @tir.vta.uop_push(0, 1, 0, 0, 0, 0, 0, 0, dtype=int32)
        @tir.call_extern("VTAUopLoopEnd", dtype=int32)
      }
      @tir.vta.coproc_dep_push(2, 1, dtype=int32)
    }
    for (ko: int32, 0, 1024) {
      attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 1 {
        @tir.vta.coproc_dep_pop(2, 1, dtype=int32)
        @tir.call_extern("VTALoadBuffer2D", @tir.tvm_thread_context(@tir.vta.command_handle(, dtype=handle), dtype=handle), A_2, ko, 1, 1, 1, 0, 0, 0, 0, 0, 2, dtype=int32)
        @tir.call_extern("VTALoadBuffer2D", @tir.tvm_thread_context(@tir.vta.command_handle(, dtype=handle), dtype=handle), B_2, ko, 1, 1024, 1024, 0, 0, 0, 0, 0, 1, dtype=int32)
        @tir.vta.coproc_dep_push(1, 2, dtype=int32)
      }
      attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 2 {
        @tir.vta.coproc_dep_pop(1, 2, dtype=int32)
        attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushGEMMOp" {
          /// 这里的1024表示c0？
          @tir.call_extern("VTAUopLoopBegin", 1024, 1, 0, 1, dtype=int32)
          @tir.vta.uop_push(0, 0, 0, 0, 0, 0, 0, 0, dtype=int32)
          @tir.call_extern("VTAUopLoopEnd", dtype=int32)
        }
        @tir.vta.coproc_dep_push(2, 1, dtype=int32)
      }
    }
    @tir.vta.coproc_dep_push(2, 3, dtype=int32)
    @tir.vta.coproc_dep_pop(2, 1, dtype=int32)
    attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 3 {
      @tir.vta.coproc_dep_pop(2, 3, dtype=int32)
      @tir.call_extern("VTAStoreBuffer2D", @tir.tvm_thread_context(@tir.vta.command_handle(, dtype=handle), dtype=handle), 0, 4, C_2, 0, 1024, 1, 1024, dtype=int32)
    }
    @tir.vta.coproc_sync(, dtype=int32)
  }
}
```

### 改进

如上所述，即使A_buf和B_buf不会导致on-chip buffer不够的问题，但是当m和n过大时C_buf就会导致on-chip buffer不够的问题。为了解决这个问题，就需要用到更高层级的blocking技术：[on-chip buffer blocking](#blocking分类), 具体实现细节可以参见[3.3 Matrix Multiply Blocking](#3.3-matrix-multiply-blocking)。

### 遗留issue

- 为啥在tensorization的时候是从C0以下的维度开始的，但是最后生成的tir的code却只看到k0这一层循环？
  - 我怀疑Runtime会需要和处理这一层，待确认。

## 3.3 Matrix Multiply Blocking

- Link: https://tvm.apache.org/docs/vta/tutorials/optimize/matrix_multiply_opt.html
- In this tutorial, the computation is intentionally too large to fit onto VTA’s on-chip buffers all at once. Therefore in the scheduling phase we will rely on computation blocking strategies to break the computation down into manageable chunks.

### Blocking分类

- VTA其实有2层buffer，由不同的硬件参数决定：

  - on-chip buffer，由下面的参数决定：

  | `LOG_INP_BUFF_SIZE` | int (log2) | Input on-chip buffer in Bytes.       |
  | ------------------- | ---------- | ------------------------------------ |
  | `LOG_WGT_BUFF_SIZE` | int (log2) | Weight on-chip buffer in Bytes.      |
  | `LOG_ACC_BUFF_SIZE` | int (log2) | Accumulator on-chip buffer in Bytes. |

  - tensor buffer，和计算能力强相关（见VTA的架构图），由LOG_BATCH和LOG_BLOCK两个参数决定；

- 上述的2层buffer决定了我们可以进行2层blocking：

  - **on-chip buffer level blocking**： 这种blocking可以视为一种优化手段，这种优化可以减少DMA反复的数据搬运，从而提高性能；这种blocking我理解是非强制的，如果未加这个level的blocking，schedule过程中必须限制on-chip buffer的计算地点，否则可能会超过on-chip buffer的限制，具体可以参考2.3.2中的做法：

  ```c++
  # Set the intermediate tensor's scope to VTA's on-chip buffers
  s[A_buf].set_scope(env.inp_scope)
  s[B_buf].set_scope(env.wgt_scope)
  s[C_buf].set_scope(env.acc_scope)
  
  # Move buffer copy into matrix multiply loop
  # 这里把buffer的范围限制了，所以无需尽心进一步blocking
  s[A_buf].compute_at(s[C_buf], ko)
  s[B_buf].compute_at(s[C_buf], ko)
  
  // ----------------------------------------------------
  // 生成的代码如下：
  for (co: int32, 0, 64) {
        for (ci: int32, 0, 16) {
          C_buf[((co*16) + ci)] = 0
          for (ko: int32, 0, 64) {
            attr [IterVar(i0: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
            /// 限制了A_buffer的计算地点；
            for (i3: int32, 0, 16) {
              A_buf[i3] = (int8*)A_2[((ko*16) + i3)]
            }
            attr [IterVar(i0_1: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
            /// 限制了B_buffer的计算地点；
            for (i3_1: int32, 0, 16) {
              B_buf[i3_1] = (int8*)B_2[((((co*16384) + (ko*256)) + (ci*16)) + i3_1)]
            }
            for (ki: int32, 0, 16) {
              C_buf[((co*16) + ci)] = ((int32*)C_buf[((co*16) + ci)] + (cast(int32, (int8*)A_buf[ki])*cast(int32, (int8*)B_buf[ki])))
            }
          }
        }
      }
  
  ```

  - **tensor buffer level blocking**：强制性的，这个blocking是由VTA计算引擎的计算能力决定的。

### On-Chip Buffer Blocking Scheduling步骤

- 原始loop（注意，bo和co都被优化掉了，因为它们的值为1）：

```c++
# Reduction axes
# two reduce axis, outer and inner;
ic = te.reduce_axis((0, in_channels // env.BLOCK_IN), name="ic")
ic_tns = te.reduce_axis((0, env.BLOCK_IN), name="ic_tns")

# Declare matrix multiply computation
# Note(ltp): bo and bi are optimized out;
# co, ci, ic, ic_tns
res_gemm = te.compute(
    output_shape,
    lambda bo, co, bi, ci: te.sum(
        data_buf[bo, ic, bi, ic_tns].astype(env.acc_dtype)
        * weight_buf[co, ic, ci, ic_tns].astype(env.acc_dtype),
        axis=[ic, ic_tns],
    ),
    name="res_gem",
)

# Add shift stage for fix-point normalization
res_shr = te.compute(output_shape, lambda *i: res_gemm(*i) >> env.INP_WIDTH, name="res_shr")

# Apply clipping between (0, input max value)
inp_max = (1 << (env.INP_WIDTH - 1)) - 1
res_max = te.compute(output_shape, lambda *i: tvm.te.max(res_shr(*i), 0), "res_max")
res_min = te.compute(output_shape, lambda *i: tvm.te.min(res_max(*i), inp_max), "res_min")

# Apply typecast to input data type before sending results back
res = te.compute(output_shape, lambda *i: res_min(*i).astype(env.inp_dtype), name="res")

// ----------------------------------------------------
// 生成的代码：
   for (co: int32, 0, 64) {  // ==> co
      for (ci: int32, 0, 16) { // ==> ci
        res_gem[((co*16) + ci)] = 0 
        // ic and ic_tns are reduce axis
        for (ic: int32, 0, 64) {  // ==> ic
          for (ic_tns: int32, 0, 16) {  // ==> ic_tns
            res_gem[((co*16) + ci)] = ((int32*)res_gem[((co*16) + ci)] + (cast(int32, (int8*)data_buf[((ic*16) + ic_tns)])*cast(int32, (int8*)weight_buf[((((co*16384) + (ic*256)) + (ci*16)) + ic_tns)])))
          }
        }
      }
    }
    for (i1_2: int32, 0, 64) {
      for (i3_2: int32, 0, 16) {
        res_gem[((i1_2*16) + i3_2)] = @tir.shift_right((int32*)res_gem[((i1_2*16) + i3_2)], 8, dtype=int32)
      }
    }
    for (i1_3: int32, 0, 64) {
      for (i3_3: int32, 0, 16) {
        res_gem[((i1_3*16) + i3_3)] = max((int32*)res_gem[((i1_3*16) + i3_3)], 0)
      }
    }
    for (i1_4: int32, 0, 64) {
      for (i3_4: int32, 0, 16) {
        res_gem[((i1_4*16) + i3_4)] = min((int32*)res_gem[((i1_4*16) + i3_4)], 127)
      }
    }
    for (i1_5: int32, 0, 64) {
      for (i3_5: int32, 0, 16) {
        res_2[((i1_5*16) + i3_5)] = cast(int8, (int32*)res_gem[((i1_5*16) + i3_5)])
      }
    }
```

- 把batch维度和oc维度按照256来分块并安排res_gemm的计算地点：

```c++
# Let's define tiling sizes (expressed in multiples of VTA tensor shape size)
b_block = 1 // env.BATCH
i_block = 256 // env.BLOCK_IN
o_block = 256 // env.BLOCK_OUT

# Tile the output tensor along the batch and output channel dimensions
# (since by default we are doing single batch inference, the split along
#  the batch dimension has no effect)
# b维度没啥影响，因为都是1
b, oc, b_tns, oc_tns = s[res].op.axis
b_out, b_inn = s[res].split(b, b_block)
oc_out, oc_inn = s[res].split(oc, o_block)
s[res].reorder(b_out, oc_out, b_inn, oc_inn)

// ----------------------------------------------------
// 生成的代码：
   for (co: int32, 0, 64) { 
      for (ci: int32, 0, 16) {
        res_gem[((co*16) + ci)] = 0
        // ic and ic_tns are reduce axis
        for (ic: int32, 0, 64) {
          for (ic_tns: int32, 0, 16) {
            res_gem[((co*16) + ci)] = ((int32*)res_gem[((co*16) + ci)] + (cast(int32, (int8*)data_buf[((ic*16) + ic_tns)])*cast(int32, (int8*)weight_buf[((((co*16384) + (ic*256)) + (ci*16)) + ic_tns)])))
          }
        }
      }
    }
    for (i1_2: int32, 0, 64) {
      for (i3_2: int32, 0, 16) {
        res_gem[((i1_2*16) + i3_2)] = @tir.shift_right((int32*)res_gem[((i1_2*16) + i3_2)], 8, dtype=int32)
      }
    }
    for (i1_3: int32, 0, 64) {
      for (i3_3: int32, 0, 16) {
        res_gem[((i1_3*16) + i3_3)] = max((int32*)res_gem[((i1_3*16) + i3_3)], 0)
      }
    }
    for (i1_4: int32, 0, 64) {
      for (i3_4: int32, 0, 16) {
        res_gem[((i1_4*16) + i3_4)] = min((int32*)res_gem[((i1_4*16) + i3_4)], 127)
      }
    }
    /// Compute for res
    for (i1.outer: int32, 0, 4) {  // ==> oc_out
      for (i1.inner: int32, 0, 16) { // ==> oc_in
        for (i3_5: int32, 0, 16) {
          res_2[(((i1.outer*256) + (i1.inner*16)) + i3_5)] = cast(int8, (int32*)res_gem[(((i1.outer*256) + (i1.inner*16)) + i3_5)])
        }
      }
    }

```

- 安排res_gemm的计算地点：

```c++
# Move intermediate computation into each output compute tile
s[res_gemm].compute_at(s[res], oc_out)
s[res_shr].compute_at(s[res], oc_out)
s[res_max].compute_at(s[res], oc_out)
s[res_min].compute_at(s[res], oc_out)

// ----------------------------------------------------
// 生成的代码：
for (i1.outer: int32, 0, 4) {  // ==> oc_out
      for (co: int32, 0, 16) { // ==> co
        for (ci: int32, 0, 16) { // ==> ci
          res_gem[((co*16) + ci)] = 0
          /// ic and ic_tns are reduce axis
          for (ic: int32, 0, 64) { // ==> ic
            for (ic_tns: int32, 0, 16) { // ==> ic_tns
              res_gem[((co*16) + ci)] = ((int32*)res_gem[((co*16) + ci)] + (cast(int32, (int8*)data_buf[((ic*16) + ic_tns)])*cast(int32, (int8*)weight_buf[(((((i1.outer*262144) + (co*16384)) + (ic*256)) + (ci*16)) + ic_tns)])))
            }
          }
        }
      }
      // s[res_shr].compute_at(s[res], oc_out)
      for (i1_2: int32, 0, 16) {
        for (i3_2: int32, 0, 16) {
          res_gem[((i1_2*16) + i3_2)] = @tir.shift_right((int32*)res_gem[((i1_2*16) + i3_2)], 8, dtype=int32)
        }
      }
      // s[res_max].compute_at(s[res], oc_out)
      for (i1_3: int32, 0, 16) {
        for (i3_3: int32, 0, 16) {
          res_gem[((i1_3*16) + i3_3)] = max((int32*)res_gem[((i1_3*16) + i3_3)], 0)
        }
      }
      // s[res_min].compute_at(s[res], oc_out)
      for (i1_4: int32, 0, 16) {
        for (i3_4: int32, 0, 16) {
          res_gem[((i1_4*16) + i3_4)] = min((int32*)res_gem[((i1_4*16) + i3_4)], 127)
        }
      }
      for (i1.inner: int32, 0, 16) { // oc_in
        for (i3_5: int32, 0, 16) {
          res_2[(((i1.outer*256) + (i1.inner*16)) + i3_5)] = cast(int8, (int32*)res_gem[((i1.inner*16) + i3_5)])
        }
      }
    }
}
```

- ic也要blocking：

```c++
# Apply additional loop split along reduction axis (input channel)
b_inn, oc_inn, b_tns, oc_tns = s[res_gemm].op.axis
ic_out, ic_inn = s[res_gemm].split(ic, i_block)

// ----------------------------------------------------
// 生成的代码：
   for (i1.outer: int32, 0, 4) { // ==> oc_out
      for (co: int32, 0, 16) {   // ==> oc_inn
        for (ci: int32, 0, 16) { // ==> oc_tns
          res_gem[((co*16) + ci)] = 0
          // ic and ic_tns are reduce axis
          for (ic.outer: int32, 0, 4) {     // ==> ic_out
            for (ic.inner: int32, 0, 16) {  // ==> ic_inn
              for (ic_tns: int32, 0, 16) {  // ==> ic_tns
                res_gem[((co*16) + ci)] = ((int32*)res_gem[((co*16) + ci)] + (cast(int32, (int8*)data_buf[(((ic.outer*256) + (ic.inner*16)) + ic_tns)])*cast(int32, (int8*)weight_buf[((((((i1.outer*262144) + (co*16384)) + (ic.outer*4096)) + (ic.inner*256)) + (ci*16)) + ic_tns)])))
              }
            }
          }
        }
      }
      /// shift
      for (i1_2: int32, 0, 16) {
        for (i3_2: int32, 0, 16) {
          res_gem[((i1_2*16) + i3_2)] = @tir.shift_right((int32*)res_gem[((i1_2*16) + i3_2)], 8, dtype=int32)
        }
      }
      /// max
      for (i1_3: int32, 0, 16) {
        for (i3_3: int32, 0, 16) {
          res_gem[((i1_3*16) + i3_3)] = max((int32*)res_gem[((i1_3*16) + i3_3)], 0)
        }
      }
      /// min
      for (i1_4: int32, 0, 16) {
        for (i3_4: int32, 0, 16) {
          res_gem[((i1_4*16) + i3_4)] = min((int32*)res_gem[((i1_4*16) + i3_4)], 127)
        }
      }
      for (i1.inner: int32, 0, 16) {
        for (i3_5: int32, 0, 16) {
          res_2[(((i1.outer*256) + (i1.inner*16)) + i3_5)] = cast(int8, (int32*)res_gem[((i1.inner*16) + i3_5)])
        }
      }
    }
  }

```

-  move the ic_out axis all the way out of the GEMM loop to block along the reduction axis：

```c++
# Reorder axes. We move the ic_out axis all the way out of the GEMM
# loop to block along the reduction axis
# original: (oc_inn, oc_tns, ic_out, ic_inn, ic_tns)
# current:  (ic_out, oc_inn, ic_inn, oc_tns, ic_tns)
s[res_gemm].reorder(ic_out, b_inn, oc_inn, ic_inn, b_tns, oc_tns, ic_tns)

// ----------------------------------------------------
// 生成的代码：
   for (i1.outer: int32, 0, 4) {  /// oc_out
      for (co.init: int32, 0, 16) {  
        for (ci.init: int32, 0, 16) {  
          res_gem[((co.init*16) + ci.init)] = 0
        }
      }
      for (ic.outer: int32, 0, 4) { // ==> ic_out
        for (co: int32, 0, 16) {    // ==> oc_inn
          for (ic.inner: int32, 0, 16) { // ==> ic_inn
            for (ci: int32, 0, 16) {  // ==> oc_tns
              for (ic_tns: int32, 0, 16) { // ==> tc_tns
                res_gem[((co*16) + ci)] = ((int32*)res_gem[((co*16) + ci)] + (cast(int32, (int8*)data_buf[(((ic.outer*256) + (ic.inner*16)) + ic_tns)])*cast(int32, (int8*)weight_buf[((((((i1.outer*262144) + (co*16384)) + (ic.outer*4096)) + (ic.inner*256)) + (ci*16)) + ic_tns)])))
              }
            }
          }
        }
      }
      for (i1_2: int32, 0, 16) {
        for (i3_2: int32, 0, 16) {
          res_gem[((i1_2*16) + i3_2)] = @tir.shift_right((int32*)res_gem[((i1_2*16) + i3_2)], 8, dtype=int32)
        }
      }
      for (i1_3: int32, 0, 16) {
        for (i3_3: int32, 0, 16) {
          res_gem[((i1_3*16) + i3_3)] = max((int32*)res_gem[((i1_3*16) + i3_3)], 0)
        }
      }
      for (i1_4: int32, 0, 16) {
        for (i3_4: int32, 0, 16) {
          res_gem[((i1_4*16) + i3_4)] = min((int32*)res_gem[((i1_4*16) + i3_4)], 127)
        }
      }
      for (i1.inner: int32, 0, 16) {
        for (i3_5: int32, 0, 16) {
          res_2[(((i1.outer*256) + (i1.inner*16)) + i3_5)] = cast(int8, (int32*)res_gem[((i1.inner*16) + i3_5)])
        }
      }
    }
```

- 标识不同buffer的范围和计算地点：

```c++
# Set scope of SRAM buffers
s[data_buf].set_scope(env.inp_scope)
s[weight_buf].set_scope(env.wgt_scope)
s[res_gemm].set_scope(env.acc_scope)
s[res_shr].set_scope(env.acc_scope)
s[res_min].set_scope(env.acc_scope)
s[res_max].set_scope(env.acc_scope)

# Block data and weight cache reads
s[data_buf].compute_at(s[res_gemm], ic_out)
s[weight_buf].compute_at(s[res_gemm], ic_out)

# Use DMA copy pragma on DRAM->SRAM operations
s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)
s[weight_buf].pragma(s[weight_buf].op.axis[0], env.dma_copy)

# Use DMA copy pragma on SRAM->DRAM operation
# (this implies that these copies should be performed along b_inn,
# or result axis 2)
s[res].pragma(s[res].op.axis[2], env.dma_copy)

// ----------------------------------------------------
// 生成的代码：
for (i1.outer: int32, 0, 4) {
    for (co.init: int32, 0, 16) {
      for (ci.init: int32, 0, 16) {
        res_gem[((co.init*16) + ci.init)] = 0
      }
    }
    for (ic.outer: int32, 0, 4) {
      attr [IterVar(i0: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
      /// copy data_buf
      for (i1: int32, 0, 16) {
        for (i3: int32, 0, 16) {
          data_buf[((i1*16) + i3)] = (int8*)data_2[(((ic.outer*256) + (i1*16)) + i3)]
        }
      }
      /// copy weights_buff
      attr [IterVar(i0_1: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
      for (i0_1, 0, 16) {
        for (i1_1: int32, 0, 16) {
          for (i2: int32, 0, 16) {
            for (i3_1: int32, 0, 16) {
              weight_buf[((((i0_1*4096) + (i1_1*256)) + (i2*16)) + i3_1)] = (int8*)weight_2[((((((i1.outer*262144) + (i0_1*16384)) + (ic.outer*4096)) + (i1_1*256)) + (i2*16)) + i3_1)]
            }
          }
        }
      }
      /// compute res_gem
      for (co: int32, 0, 16) {
        for (ic.inner: int32, 0, 16) {
          for (ci: int32, 0, 16) {
            for (ic_tns: int32, 0, 16) {
              res_gem[((co*16) + ci)] = ((int32*)res_gem[((co*16) + ci)] + (cast(int32, (int8*)data_buf[((ic.inner*16) + ic_tns)])*cast(int32, (int8*)weight_buf[((((co*4096) + (ic.inner*256)) + (ci*16)) + ic_tns)])))
            }
          }
        }
      }
    }
    for (i1_2: int32, 0, 16) {
      for (i3_2: int32, 0, 16) {
        res_gem[((i1_2*16) + i3_2)] = @tir.shift_right((int32*)res_gem[((i1_2*16) + i3_2)], 8, dtype=int32)
      }
    }
    for (i1_3: int32, 0, 16) {
      for (i3_3: int32, 0, 16) {
        res_gem[((i1_3*16) + i3_3)] = max((int32*)res_gem[((i1_3*16) + i3_3)], 0)
      }
    }
    for (i1_4: int32, 0, 16) {
      for (i3_4: int32, 0, 16) {
        res_gem[((i1_4*16) + i3_4)] = min((int32*)res_gem[((i1_4*16) + i3_4)], 127)
      }
    }
    for (i1.inner: int32, 0, 16) {
      attr [IterVar(i2_1: int32, (nullptr), "DataPar", "")] "pragma_dma_copy" = 1;
      for (i3_5: int32, 0, 16) {
        res_2[(((i1.outer*256) + (i1.inner*16)) + i3_5)] = cast(int8, (int32*)res_gem[((i1.inner*16) + i3_5)])
      }
    }
  }

```

- Tensorization

```c++
# Apply tensorization over the batch tensor tile axis
s[res_gemm].tensorize(b_tns, env.gemm)

# Add an ALU pragma over the shift and clipping operations
s[res_shr].pragma(s[res_shr].op.axis[0], env.alu)
s[res_min].pragma(s[res_min].op.axis[0], env.alu)
s[res_max].pragma(s[res_max].op.axis[0], env.alu)

// ----------------------------------------------------
// 生成的代码：
@tir.vta.coproc_dep_push(3, 2, dtype=int32)
    for (i1.outer: int32, 0, 4) {
      attr [IterVar(vta: int32, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 2 {
        @tir.vta.coproc_dep_pop(3, 2, dtype=int32)
        attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushGEMMOp" {
          @tir.call_extern("VTAUopLoopBegin", 16, 1, 0, 0, dtype=int32)
          @tir.vta.uop_push(0, 1, 0, 0, 0, 0, 0, 0, dtype=int32)
          @tir.call_extern("VTAUopLoopEnd", dtype=int32)
        }
        @tir.vta.coproc_dep_push(2, 1, dtype=int32)
      }
      for (ic.outer: int32, 0, 4) {
        attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 1 {
          @tir.vta.coproc_dep_pop(2, 1, dtype=int32)
          @tir.call_extern("VTALoadBuffer2D", @tir.tvm_thread_context(@tir.vta.command_handle(, dtype=handle), dtype=handle), data_2, (ic.outer*16), 16, 1, 16, 0, 0, 0, 0, 0, 2, dtype=int32)
          @tir.call_extern("VTALoadBuffer2D", @tir.tvm_thread_context(@tir.vta.command_handle(, dtype=handle), dtype=handle), weight_2, ((i1.outer*1024) + (ic.outer*16)), 16, 16, 64, 0, 0, 0, 0, 0, 1, dtype=int32)
          @tir.vta.coproc_dep_push(1, 2, dtype=int32)
        }
        attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 2 {
          @tir.vta.coproc_dep_pop(1, 2, dtype=int32)
          attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushGEMMOp" {
            @tir.call_extern("VTAUopLoopBegin", 16, 1, 0, 16, dtype=int32)
            @tir.call_extern("VTAUopLoopBegin", 16, 0, 1, 1, dtype=int32)
            @tir.vta.uop_push(0, 0, 0, 0, 0, 0, 0, 0, dtype=int32)
            @tir.call_extern("VTAUopLoopEnd", dtype=int32)
            @tir.call_extern("VTAUopLoopEnd", dtype=int32)
          }
          @tir.vta.coproc_dep_push(2, 1, dtype=int32)
        }
      }
      @tir.vta.coproc_dep_pop(2, 1, dtype=int32)
      attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 2 {
        attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushALUOp" {
          @tir.call_extern("VTAUopLoopBegin", 16, 1, 1, 0, dtype=int32)
          @tir.vta.uop_push(1, 0, 0, 0, 0, 3, 1, 8, dtype=int32)
          @tir.call_extern("VTAUopLoopEnd", dtype=int32)
        }
        attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushALUOp" {
          @tir.call_extern("VTAUopLoopBegin", 16, 1, 1, 0, dtype=int32)
          @tir.vta.uop_push(1, 0, 0, 0, 0, 1, 1, 0, dtype=int32)
          @tir.call_extern("VTAUopLoopEnd", dtype=int32)
        }
        attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushALUOp" {
          @tir.call_extern("VTAUopLoopBegin", 16, 1, 1, 0, dtype=int32)
          @tir.vta.uop_push(1, 0, 0, 0, 0, 0, 1, 127, dtype=int32)
          @tir.call_extern("VTAUopLoopEnd", dtype=int32)
        }
        @tir.vta.coproc_dep_push(2, 3, dtype=int32)
      }
      attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 3 {
        @tir.vta.coproc_dep_pop(2, 3, dtype=int32)
        for (i1.inner: int32, 0, 16) {
          @tir.call_extern("VTAStoreBuffer2D", @tir.tvm_thread_context(@tir.vta.command_handle(, dtype=handle), dtype=handle), i1.inner, 4, res_2, ((i1.outer*16) + i1.inner), 1, 1, 1, dtype=int32)
        }
        @tir.vta.coproc_dep_push(3, 2, dtype=int32)
      }
    }
```



## 3.4 2D Convolution Optimization

- Link: https://tvm.apache.org/docs/vta/tutorials/optimize/convolution_opt.html#sphx-glr-vta-tutorials-optimize-convolution-opt-py

TODO

## 3.5 End2End

- Link: https://tvm.apache.org/docs/vta/tutorials/frontend/deploy_classification.html#sphx-glr-vta-tutorials-frontend-deploy-classification-py

- 适配性修改

  - 采用onnx模型而不是mxnet模型；
  - 因此前处理和后处理会有些许区别；

- 遗留Issues

  - 调用graph_pack接口会出现call.attr.channels属性不存在的情况，主要原因是：这个接口默认模型是从mxnet模型转换而来，mxnet模型转换的时候会往relay的op里面添加channels属性，而我们的demo是从onnx转换而来，缺少这个“channels”属性；同样的问题论坛里也有问到https://discuss.tvm.apache.org/t/vta-onnx-error-while-loading-resnet18-model/6601.

    - 解决办法：修改tvm/vta/python/vta/top/graphpack.py文件，从op的otype中提取得到channels的值（我估计不是最终的解决办法）

  - 最终运行的结果不对，采用了同样的前处理和后处理，cpu的结果是ok的，但是vta的结果不对；需要进一步定位原因。

  - 几个核心抽象需要搞明白：

    - target
    - ext_dev
    - ctx

  - 对于整个flow里面我有几个关键问题没搞明白：

    - 为何env.TARGET是sim的时候，需要重新封装target对象

    ```python
     if env.TARGET == "intelfocl" or env.TARGET == "sim":
                # multiple targets to run both on cpu and vta
                print("before target:", target)
                target = {"cpu": env.target_vta_cpu, "ext_dev": target}
                print("after target:", target)
            with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
                graph, lib, params = relay.build(
                    relay_prog, target=target, params=params, target_host=env.target_host
                )
    ```

    - 为何env.TARGET是sim的时候，需要多个ctxes？

    ```python
     if env.TARGET == "intelfocl" or env.TARGET == "sim":
            ctxes = [remote.ext_dev(0), remote.cpu(0)]
            m = graph_runtime.create(graph, lib, ctxes)
    ```



# 4. TVM Support

## 4.1 VTA Low-Level Build Flow

### 4.1.1 函数入口和build_config

VTA的build入口函数在tvm/vta/python/vta/build_module.py：

```python
def build(*args, **kwargs):
    """Thin wrapper of tvm.build

    This wrapper automatically applies VTA's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.build : The original TVM's build function
    """
    pass_ctx = tvm.transform.PassContext.current()
    if not pass_ctx.config.get("tir.add_lower_pass"):
        with build_config():
            return tvm.build(*args, **kwargs)
    return tvm.build(*args, **kwargs)
```

这个函数很简单，其主要的工作是：

1. 判断当前的pass ctx的配置中是否有“tir.add_lower_pass”，如果没有则添加VTA相应的build_config到context中去；
2. 调用标准的tvm.build接口进行编译；

在vta的build_config里面会添加一系列的lower pass专门用于VTA的lowering：

```python
pass_list = [
        (0, transform.InjectConv2DTransposeSkip()),
        (1, transform.InjectDMAIntrin()),
        (1, transform.InjectSkipCopy()),
        (1, transform.AnnotateALUCoProcScope()),
        (1, tvm.tir.transform.LiftAttrScope("coproc_uop_scope")),
        (1, transform.LiftAllocToScopeBegin()),
        (1, tvm.tir.transform.LiftAttrScope("coproc_scope")),
        (1, transform.InjectCoProcSync()),
        (1, EarlyRewrite()),
    ]
    if debug_flag:
        pass_list.append((1, add_debug))
    pass_list.append((2, transform.InjectALUIntrin()))
    pass_list.append((3, tvm.tir.transform.LowerDeviceStorageAccessInfo()))
    pass_list.append((3, transform.FoldUopLoop()))
    pass_list.append((3, transform.CPUAccessRewrite()))
```

### 4.1.2 VTA Low-Level Build Flow

VTA low level的build flow遵循标准的tvm的build flow，如下所示：

![lower build flow](.\VTA\lower build flow.png)

> 注：在VTA的当前case下，mdev会返回None；

总流程比较简单，先lower把schedule lower成PrimFunc；然后再编译（支持异构编译）成不同的运行时Module，最后打包一个Module；

### 4.1.3 codegen.build_module实现

codegen.build_module调用的是_ffi_api中的Build接口：

```python
def build_module(mod, target):
    """Build IRModule into Module.

    Parameters
    ----------
    mod : tvm.IRModule
        The ir module.

    target : str
        The target module type.

    Returns
    -------
    module : runtime.Module
        The corressponding module.
    """
    target = Target(target) if isinstance(target, str) else target
    return _ffi_api.Build(mod, target)
```

而_ffi_api.Build(mod, target) 会调用C++里面的实现：

```c++
runtime::Module Build(IRModule mod, Target target) {
  if (transform::PassContext::Current()
          ->GetConfig<Bool>("tir.disable_assert", Bool(false))
          .value()) {
    mod = tir::transform::SkipAssert()(mod);
  }

  // the build function.
  std::string build_f_name = "target.build." + target->kind->name;
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  ICHECK(bf != nullptr) << build_f_name << " is not enabled";
  return (*bf)(mod, target);
}
```

在这个接口里或获取对应target的build函数(PackedFunc类型)，然后利用该函数进行实际的build过程；在VTA这个例子下，“target->kind->name”为“llvm”，所以“build_f_name”为：target.build.llvm，然后再调用这函数去生成具体的运行时Module；

但是这时候其实有个问题：VTA是JIT编译，其runtime会提供相应的API给tvm，tvm生成的tir里面会有一些对应的tir.call_extern intrisic来调用VTA runtime的接口。但是从上面整个流程下来上面生成的host rt module里面肯定是不包含VTA的runtime的（gemm.o里面涉及到VTA runtime的接口都是外部符号），那么最终生成的f Module，里面调用计算函数的时候为啥能找到VTA runtime的外部API符号呢？

```python
# Build GEMM VTA kernel
my_gemm = vta.build(s, [A, B, C], "ext_dev", env.target_host, name="my_gemm")

# Write the compiled module into an object file.
temp = utils.tempdir()
my_gemm.save(temp.relpath("gemm.o")) # 注意：gemm.o里面涉及到的VTA runtime API都是外部符号

# Send the executable over RPC
remote.upload(temp.relpath("gemm.o"))

# Load the compiled module
f = remote.load_module("gemm.o")

...

# Invoke the module to perform the computation
f(A_nd, B_nd, C_nd)
```

> 用nm命令查询gemm.o，信息如下：
>
> ```bash
> (tvm) michael@xbjtianping30:~/tvm.end2end$ nm gemm.o
> 0000000000000000 r .LCPI0_0
> 0000000000000010 r .LCPI0_1
> 0000000000000020 b .tvm_func.__tvm_set_device
>               U VTADepPop              ==> U代表的这些都是外部符号；  
>               U VTADepPush
>               U VTALoadBuffer2D
>               U VTAPushGEMMOp
>               U VTAStoreBuffer2D
>               U VTASynchronize
>               U VTATLSCommandHandle
>               U VTAUopLoopBegin
>               U VTAUopLoopEnd
>               U VTAUopPush
> 0000000000000018 V __TVMAPISetLastError
> 0000000000000010 V __TVMBackendGetFuncFromEnv
> 0000000000000008 V __TVMFuncCall
> 0000000000001239 V __tvm_main__
> 0000000000000000 V __tvm_module_ctx
> 0000000000000000 T my_gemm
> 0000000000000560 t my_gemm_compute_
> ```

VTA里面用了2个不同的方式来解决这个问题：

- 当时用simulator时，当执行from vta.testing import simulator时会自动（tvm/vta/python/vta/testing/simulator.py）：

```
LIBS = _load_sw()
```

_load_sw会执行 libs = [ctypes.CDLL(lib_driver[0], ctypes.RTLD_GLOBAL)]把runtime库import到当前环境中；

- 当使用真正的rcp方式时：VTA覆写了一部分rpc的接口，当rpc server启动的时候就会完成VTA runtime的加载（实现文件位于tvm/vta/python/vta/exec/rpc_server.py)，从而当rpc server导入gemm.o的时候runtime API这些runtime的符号都是可见的；

```python
@tvm.register_func("tvm.rpc.server.start", override=True)
def server_start():
    """VTA RPC server extension."""
    # pylint: disable=unused-variable
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../../../../"))
    dll_path = find_libvta("libvta")[0]
    cfg_path = os.path.abspath(os.path.join(proj_root, "3rdparty/vta-hw/config/vta_config.json"))
    runtime_dll = []
    _load_module = tvm.get_global_func("tvm.rpc.server.load_module")

    def load_vta_dll():
        """Try to load vta dll"""
        if not runtime_dll:
            runtime_dll.append(ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL))
        logging.info("Loading VTA library: %s", dll_path)
        return runtime_dll[0]

    @tvm.register_func("tvm.rpc.server.load_module", override=True)
    def load_module(file_name):
        load_vta_dll()
        return _load_module(file_name)

    @tvm.register_func("device_api.ext_dev")
    def ext_dev_callback():
        load_vta_dll()
        return tvm.get_global_func("device_api.ext_dev")()

   ...

    @tvm.register_func("tvm.rpc.server.shutdown", override=True)
    def server_shutdown():
        if runtime_dll:
            runtime_dll[0].VTARuntimeShutdown()
            runtime_dll.pop()
            
    ...
```

### 4.1.4 相关问题

**如果传入的是schedule，lower的过程和最终生成的东西是什么?**

- 我理解是把一个schedule变成一个PrimFunc(由tir组成的low-level函数)；



**lower和_build_for_device的联系和区别是什么？**

- lower是把schedule变成PrimFunc；
- _build_for_device负责：
  - 把IR module partition成dev ir mod和host ir mod，并且分别要经过一些特定的pass；（partition之后dev_mod可能不包含任何PrimFunc）;
  - 为dev ir mod生成device-specific code和打包成rt dev mod(Module)，即运行时module；



**CallingConv.DEVICE_KERNEL_LAUNCH和CallingConv.C_PACKED_FUNC的具体区别是什么? **

> 引申问题：
>
> - 为何单单只在DEVICE_KERNEL_LAUNCH的情况下会去生成rt_dev Module，为何C_PACKED_FUNC情况下不需要?
>
> - 如果我增加新的device支持，什么情况下选择DEVICE_KERNEL_LAUNCH，什么情况下选择C_PACKED_FUNC？

- 具体callingConv在C++中的定义，总结起来其实就是三种：
  - 当定义为kDefault时，实现的时候以默认的方式暴露出去；
  - 当定义为kCPackedFunc时，实现的时候以CPackedFunc的方式暴露出去；
  - 当定义为kDeviceKernelLaunch时，实现的时候由device本身决定；

```c++
/*!
 * \brief Possible Calling conventions.
 *
 *  NOTE: The calling convention also implies
 *  the way we implement the function during lowering.
 */
enum class CallingConv : int {
  /*!
   * \brief Default calling convetion.
   *
   * - Uses the native calling convention of the target.
   * - Implementation: specified by the native target.
   */
  kDefault = 0,
  /*!
   * \brief PackedFunc that exposes a CPackedFunc signature.
   *
   * - Calling by PackedFunc calling convention.
   * - Implementation: Expose a function with the CPackedFunc signature.
   */
  kCPackedFunc = 1,
  /*!
   * \brief Device kernel launch
   *
   * - Call by PackedFunc calling convention.
   * - Implementation: defined by device runtime(e.g. runtime/cuda)
   */
  kDeviceKernelLaunch = 2,
};

```

- 在TVM当前支持的target当中：

  - 以kCPackedFunc实现的有:hexagon
  - 以kDeviceKernelLaunch实现的有：Cuda，AOCL, Metal，OpenCL, VHLS, SPIRV, 

  

**在_build_for_device接口中完成partition的pass是哪个（标识calling_conv属性）？具体是怎么实现的？**

- tvm.tir.transform.SplitHostDevice() pass会做整个操作：

```python
    opt_mixed += [
        tvm.tir.transform.ThreadSync("shared"),
        tvm.tir.transform.ThreadSync("warp"),
        tvm.tir.transform.InferFragment(),
        tvm.tir.transform.LowerThreadAllreduce(),
        tvm.tir.transform.MakePackedAPI(),
        # 整个pass会去标识calling_conv属性值；
        tvm.tir.transform.SplitHostDevice(),
    ]
```

- 这个pass的具体实现在（tvm/src/tir/transforms/split_host_device.cc)；初步来看，感觉是PrimFunc中的stmtNode存在属性（attr::thread_extend || attr::pipeline_exec_scope || attr::device_scope）即会被把PrimcFunc split成2个func,并且存入dev_mod中：

  - 一个device function，需要在device上实现或者编译；
  - 一个负责调用上述device 函数的function；

  > 引申问题：这些属性是在什么时候设置的呢? 由谁负责来设置？

```c++
class HostDeviceSplitter : public StmtMutator {
 public:
  explicit HostDeviceSplitter(IRModule* device_mod, Target device_target, std::string name_prefix)
      : device_mod_(device_mod), device_target_(device_target), name_prefix_(name_prefix) {}
...

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    /// 具有如下属性的会split device function？
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::pipeline_exec_scope ||
        op->attr_key == attr::device_scope) {
      return SplitDeviceFunc(GetRef<Stmt>(op));
    }
    return StmtMutator::VisitStmt_(op);
  }
...

Stmt SplitDeviceFunc(Stmt body) {
    ...
    PrimFunc device_func(params, Substitute(body, remap_vars));
    device_func = WithAttr(std::move(device_func), tir::attr::kDeviceThreadAxis, m.thread_axis_);
    /// 注：会被标注CallingConv::kDeviceKernelLaunch属性
    device_func = WithAttr(std::move(device_func), tvm::attr::kCallingConv,
                           Integer(CallingConv::kDeviceKernelLaunch));
    device_func =
        WithAttr(std::move(device_func), tvm::attr::kGlobalSymbol, runtime::String(kernel_symbol));
    device_func = WithAttr(std::move(device_func), tir::attr::kNoAlias, Integer(1));
    device_func = WithAttr(std::move(device_func), tvm::attr::kTarget, device_target_);
    (*device_mod_)->Add(GlobalVar(kernel_symbol), device_func);

    // generate calls to the device function
    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(kernel_symbol));
    for (PrimExpr arg : arguments) {
      call_args.push_back(arg);
    }
    for (PrimExpr ext : m.thread_extent_) {
      call_args.push_back(ext);
    }
    return Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), call_args));
  }
```



**在_build_for_device中，具体什么样的PrimFunc会被partition成dev mod然后调用codegen.build进行编译？**

- 从代码上看，具有calling_conv属性，并且属性值为“CallingConv.DEVICE_KERNEL_LAUNCH”的PrimFunc会别放入mod_dev(IRModule)，然后进一步被编译生成rt_mod_dev(Module)；
- 针对VTA当前的case来说，mod_dev中的function数量为0，所有的primfunc被划分到了host mod中；

```python
opt_device = tvm.transform.Sequential(
        [
            tvm.tir.transform.Filter(
                lambda f: "calling_conv" in f.attrs
                and f.attrs["calling_conv"].value == CallingConv.DEVICE_KERNEL_LAUNCH
            ),
            tvm.tir.transform.LowerWarpMemory(),
            tvm.tir.transform.Simplify(),
            tvm.tir.transform.LowerDeviceStorageAccessInfo(),
            tvm.tir.transform.LowerCustomDatatypes(),
            tvm.tir.transform.LowerIntrin(),
        ]
    )
mod_dev = opt_device(mod_mixed)

...
# 如果mod_dev中的function数量不为空，则打包dev的runtime module；    
rt_mod_dev = codegen.build_module(mod_dev, target) if len(mod_dev.functions) != 0 else None
```



**codegen.build_module的功能和具体实现？**

- 功能：为dev mod生成指令和打包相应运行时mod，为host mod生成指令和打包相应的运行时mod；
- 见[codegen.build_module实现](#codegen.build_module实现).



**这种把外部lib导入的方式是标准的吗？有没有其他的方式？**



**为何要用remote Module load_module之后才能使用f(A_nd, B_nd, C_nd)？为啥不能直接使用my_gemm Module拿到f然后直接调用?**

- 难道和ext_dev有关？



**如果PrimFunc的calling_conv属性为CallingConv.DEVICE_KERNEL_LAUNCH，有没有全流程的例子？**



### 4.1.5 总结

- 为了给VTA生成指令，需要在build_config里面添加针对VTA的lower pass，其主要目标是在lower的过程中生成针对VTA的tir（主要是tir.call_extern）；
- 在_build_for_device里面会对ir module进行过滤，把target_host侧和device侧的PrimFunc分开，然后分开编译成不同的运行时Module后再重新打包在一起；注：当前VTA case下没有device侧的PrimFunc；
- 涉及到具体target的编译都是由codegen.build_module来完成的，其会委托到各自不同target自身的target.build.xxx来完成具体运行时Module的生成；
- 当前VTA的场景下没有device运行时Module，所有的编译都在host侧，但是host侧里面会使用到tir.extern_call进行外部函数调用（也就是VTA runtime的API），因此执行host侧生成的Module时需要导入VTA runtime lib，根据使用场景不同，vta采用了不同的导入方式，见[codegen.build_module实现](#codegen.build_module实现)里面的介绍。

## 4.2 VTA Runtime

### 4.2.1 runtime

Runtime主要负责JIT的编译，里面定义了一些标准API；

### 4.2.2 device

需要实现标准的DeviceAPI，并且注册到runtime系统：

```c++
class VTADeviceAPI final : public DeviceAPI {
...
    
// Register device api with override.
static TVM_ATTRIBUTE_UNUSED auto& __register_dev__ =
    ::tvm::runtime::Registry::Register("device_api.ext_dev", true)
        .set_body([](TVMArgs args, TVMRetValue* rv) {
          DeviceAPI* ptr = VTADeviceAPI::Global();
          *rv = static_cast<void*>(ptr);
        });
```

## 4.3 Relay + VTA build Flow



## 4.4 Tensorization





# 5 Appenix

## Lower实现



## Reference

[1] How to Use TVM Pass Infra: https://tvm.apache.org/docs/tutorials/dev/use_pass_infra.html

[2] Pass Infrastructure: https://tvm.apache.org/docs/dev/pass_infra.html#pass-infra

[3] TVM/VTA代码生成流程: https://krantz-xrf.github.io/2019/10/24/tvm-workflow.html



