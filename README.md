
# Contents
- [Comprehensive Survey](#comprehensive-survey)
- [Halide](#halide)
- [TVM](#tvm)
- [MLIR](#mlir)
- [RISC-V](#risc-v)
- [DSA](#dsa)
- [Talks](#talks)
- [Courses](#courses)

# Comprehensive Survey

- Deep Learning Compilers [[slide](https://ucbrise.github.io/cs294-ai-sys-sp19/assets/lectures/lec12/dl-compilers.pdf)]
  - 说明了DL Compiler领域需要解决的问题；
  - 强调了Halide的短板和TVM，TC要解决的问题；
- The Evolution of Domain-Specific Computing for Deep Learning [[paper](https://ieeexplore.ieee.org/abstract/document/9439420/)]
  - Xilinx Lab所撰写的survey paper；
  - 里面对AIE Engine做了一些介绍；
  - 对MLIR生成AIE code做了简单介绍；
- Compute Substrate for Software 2.0 [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9373921)]



# Halide

- Decoupling Algorithms from Schedules for Easy Optimization of Image Processing Pipelines [[paper](https://people.csail.mit.edu/jrk/halide12/halide12.pdf)] 



# TVM

- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning [[paper](https://homes.cs.washington.edu/~arvind/papers/tvm.pdf)]

  

- Enabling TVM on RISC-V Architectures with SIMD Instructions [[slide](https://riscv.org/wp-content/uploads/2019/03/16.45-Enabling-TVM-on-RISC-V-Architectures-with-SIMD-Instructions-v2.pdf)]



- TVM and XLA区别: https://zhuanlan.zhihu.com/p/87664838

  

- TVM schedule

  - An Operational Model of Schedules in  Tensor Expression [[doc](https://docs.google.com/document/d/1nmz00_n4Ju-SpYN0QFl3abTHTlR_P0dRyo5zsWC0Q1k/edit)]



- TVM Conf 2020 - Tutorials - Bring Your Own Codegen to TVM [[video](https://www.youtube.com/watch?v=DD8GdZ_OKco)]
  - dev guide: https://tvm.apache.org/docs/dev/relay_bring_your_own_codegen.html
  - blog: https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm



- Scalable and Intelligent Learning Systems [[thesis](https://digital.lib.washington.edu/researchworks/handle/1773/44766)]

  - 陈天奇的phd thesis

    

- Ansor: Generating High-Performance Tensor Programs for Deep Learning [[pdf](https://arxiv.org/pdf/2006.06762.pdf)]



- Bridging PyTorch and TVM [[web](https://tvm.apache.org/2020/07/14/bert-pytorch-tvm)]

  

- Compilation of Quantized Models in TVM [[pdf](http://lenlrx.cn/wp-content/uploads/2019/11/Nov8_TVM_meetup_Quantization.pdf)]



- The quantization story for TVM [[web](https://discuss.tvm.apache.org/t/quantization-story/3920)]

  

- Graph partitioning a -nd Heterogeneous Execution [[RFC](https://discuss.tvm.apache.org/t/graph-partitioning-and-heterogeneous-execution/504)]

  

- TVM Object System Multi language Support for just 19 99 [[video](https://www.youtube.com/watch?v=-TM_EPih4Co)]

  - Cross-Language interaction in Deep Learning
  - What is in an ABI?
  - TVM Object System
  - Packed Function
  - Extending to Rust

  

- TVM PackedFunc实现机制 [[link](https://hjchen2.github.io/2020/01/10/TVM-PackedFunc%E5%AE%9E%E7%8E%B0%E6%9C%BA%E5%88%B6/)]

  

- TVM Relay Build Flow [[link](https://zhuanlan.zhihu.com/p/257150960)]



# MLIR

- MLIR: A Compiler Infrastructure for the End of Moore's Law [[paper](https://arxiv.org/abs/2002.11054)]

- Multi-Level Intermediate Representation Compiler Infrastructure [[slide](https://docs.google.com/presentation/d/11-VjSNNNJoRhPlLxFgvtb909it1WNdxTnQFipryfAPU/edit#slide=id.g7d334b12e5_0_4)]

- MLIR: Scaling Compiler Infrastructure for Domain Specific Computation [[paper](https://research.google/pubs/pub49988/)]

  

- Thoughts on Tensor Code Generation in MLIR [[video](https://drive.google.com/file/d/1PKY5yVEL0Dl5UHaok4NgpxnbwXbi5pxS/view)] [[slide](https://docs.google.com/presentation/d/1M44If0Lw2lnrlyE_xNU1WOmXWxLo9FibMwdUbrAhOhU/edit#slide=id.g5fd22bdf8c_0_0)]



- HIGH PERFORMANCE CODE GENERATION IN MLIR: AN EARLY CASE STUDY WITH GEMM [[paper](https://arxiv.org/pdf/2003.00532.pdf)]

- Polyhedral Compilation Opportunities in MLIR [[slide](http://impact.gforge.inria.fr/impact2020/slides/IMPACT_2020_keynote.pdf)]



- Compiling ONNX Neural Network Models Using MLIR [[paper](https://arxiv.org/pdf/2008.08272.pdf)]



# RISC-V

- Design of the RISC-V Instruction Set Architecture [[thesis](https://digitalassets.lib.berkeley.edu/etd/ucb/text/Waterman_berkeley_0028E_15908.pdf)]



# DSA

- Custom Hardware Architectures for Deep Learning on Portable Devices: A Review [[paper](https://ieeexplore.ieee.org/abstract/document/9447019)]
- 从GPU谈异构（4）[[zhihu](https://zhuanlan.zhihu.com/p/376409878)]



# Talks

- The Golden Age of Compilers, in an era of Hardware/Software co-design [[slide](https://docs.google.com/presentation/d/1ZMtzT6nmfvNOlIaHRzdaXpFeaAklcT7DvfGjhgpzcxk/edit#slide=id.p)] [[video](https://drive.google.com/file/d/1eIxFZZLOM7a3LYL1QaKhflKl0jRLPp-V/view)]
  - A discussion about accelerator design, benefits of reducing fragmentation by standardizing non-differentiated parts of large scale HW/SW systems.



# Courses

- CS448h - Domain-specific Languages for Graphics, Imaging, and Beyond: http://cs448h.stanford.edu/
  - Designing intermediate representations;
  - IR design, transformations, and code generation;

