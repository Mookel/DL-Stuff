
# Contents
- [Comprehensive Survey](#comprehensive-survey)
- [Halide](#halide)
- [TVM](#tvm)
  * [Basics](#basics)
  * [Schedule](#schedule)
  * [Quantization](#quantization)
  * [Build Flow](#build-flow)
  * [Runtime](#runtime)
- [MLIR](#mlir)
- [Hardware](#hardware)
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
- MIT 6.815/6.865 [[link](https://stellar.mit.edu/S/course/6/sp15/6.815/materials.html)]
  - 对Halide的核心思想，调度原语做了很好的介绍，非常多的例子，具有很好的参考价值；



# TVM

- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning [[paper](https://homes.cs.washington.edu/~arvind/papers/tvm.pdf)]
- Scalable and Intelligent Learning Systems [[thesis](https://digital.lib.washington.edu/researchworks/handle/1773/44766)]

  - 陈天奇的phd thesis
- TVM Conf 2020 - Tutorials - Bring Your Own Codegen to TVM [[video](https://www.youtube.com/watch?v=DD8GdZ_OKco)]
  - dev guide: https://tvm.apache.org/docs/dev/relay_bring_your_own_codegen.html
  - blog: https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm
- Bridging PyTorch and TVM [[web](https://tvm.apache.org/2020/07/14/bert-pytorch-tvm)]
- TVM and XLA区别 [[link](https://zhuanlan.zhihu.com/p/87664838)] 
- Enabling TVM on RISC-V Architectures with SIMD Instructions [[slide](https://riscv.org/wp-content/uploads/2019/03/16.45-Enabling-TVM-on-RISC-V-Architectures-with-SIMD-Instructions-v2.pdf)]

## Basics

- TVM: Design and Architecture [[web](https://tvm.apache.org/docs/dev/index.html)]

- TVM Object System Multi language Support for just 19.99 [[video](https://www.youtube.com/watch?v=-TM_EPih4Co)]
  - What is in an ABI?
  - TVM Object System
  - Packed Function

- TVM PackedFunc实现机制 [[link](https://hjchen2.github.io/2020/01/10/TVM-PackedFunc%E5%AE%9E%E7%8E%B0%E6%9C%BA%E5%88%B6/)]
- TVM之Tensor数据结构解读 [[zhihu](https://zhuanlan.zhihu.com/p/341257418)]
  - 对于Tensor，Operation之间的关系做了比较好的介绍；
- TVM中的IR设计与技术实现 [[blog](https://www.cnblogs.com/CocoML/p/14643355.html)]
  - 对IRModule和Module做了非常专业的介绍

## Schedule

- TVM schedule: An Operational Model of Schedules in Tensor Expression [[doc](https://docs.google.com/document/d/1nmz00_n4Ju-SpYN0QFl3abTHTlR_P0dRyo5zsWC0Q1k/edit)]
- Ansor: Generating High-Performance Tensor Programs for Deep Learning [[pdf](https://arxiv.org/pdf/2006.06762.pdf)]

## Quantization

- The quantization story for TVM [[web](https://discuss.tvm.apache.org/t/quantization-story/3920)]
- Compilation of Quantized Models in TVM [[pdf](http://lenlrx.cn/wp-content/uploads/2019/11/Nov8_TVM_meetup_Quantization.pdf)]

## Build Flow

- TVM Relay Build Flow [[link](https://zhuanlan.zhihu.com/p/257150960)]
- TVM的“hello world“基础流程（上）[[link](https://blog.csdn.net/jinzhuojun/article/details/117135551)]
  - 写的非常专业，对一些TE/TIE中的python/C++类做了介绍；
- TVM/VTA代码生成流程 [[blog](https://krantz-xrf.github.io/2019/10/24/tvm-workflow.html)]
  - 流程图画的还挺清晰
- TVM的编译流程详解 [[LINK](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247494801&idx=1&sn=b893c43133eea1343034bb0aca356e24&scene=21#wechat_redirect)]
- TVM的CodeGen流程[[LINK](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247495638&idx=1&sn=ca711f6f33e6f83ed9ec27afddb416a0&chksm=9f835540a8f4dc56f3fe77b2d4aefd211220bd61bc3f2bbe03ae425478309d831f339ed6b39a&scene=178&cur_album_id=1788791560017346569#rd)]

## Runtime

- Graph partitioning and Heterogeneous Execution [[RFC](https://discuss.tvm.apache.org/t/graph-partitioning-and-heterogeneous-execution/504)]

# MLIR

- MLIR: A Compiler Infrastructure for the End of Moore's Law [[paper](https://arxiv.org/abs/2002.11054)]
- Multi-Level Intermediate Representation Compiler Infrastructure [[slide](https://docs.google.com/presentation/d/11-VjSNNNJoRhPlLxFgvtb909it1WNdxTnQFipryfAPU/edit#slide=id.g7d334b12e5_0_4)]
- MLIR: Scaling Compiler Infrastructure for Domain Specific Computation [[paper](https://research.google/pubs/pub49988/)]

- Thoughts on Tensor Code Generation in MLIR [[video](https://drive.google.com/file/d/1PKY5yVEL0Dl5UHaok4NgpxnbwXbi5pxS/view)] [[slide](https://docs.google.com/presentation/d/1M44If0Lw2lnrlyE_xNU1WOmXWxLo9FibMwdUbrAhOhU/edit#slide=id.g5fd22bdf8c_0_0)]

- HIGH PERFORMANCE CODE GENERATION IN MLIR: AN EARLY CASE STUDY WITH GEMM [[paper](https://arxiv.org/pdf/2003.00532.pdf)]

- Polyhedral Compilation Opportunities in MLIR [[slide](http://impact.gforge.inria.fr/impact2020/slides/IMPACT_2020_keynote.pdf)]

- Compiling ONNX Neural Network Models Using MLIR [[paper](https://arxiv.org/pdf/2008.08272.pdf)] [[github](https://github.com/onnx/onnx-mlir)]



# Hardware

- Design of the RISC-V Instruction Set Architecture [[thesis](https://digitalassets.lib.berkeley.edu/etd/ucb/text/Waterman_berkeley_0028E_15908.pdf)]

- Custom Hardware Architectures for Deep Learning on Portable Devices: A Review [[paper](https://ieeexplore.ieee.org/abstract/document/9447019)]
- 从GPU谈异构（4）[[zhihu](https://zhuanlan.zhihu.com/p/376409878)]



# Talks

- The Golden Age of Compilers, in an era of Hardware/Software co-design [[slide](https://docs.google.com/presentation/d/1ZMtzT6nmfvNOlIaHRzdaXpFeaAklcT7DvfGjhgpzcxk/edit#slide=id.p)] [[video](https://drive.google.com/file/d/1eIxFZZLOM7a3LYL1QaKhflKl0jRLPp-V/view)]
  - A discussion about accelerator design, benefits of reducing fragmentation by standardizing non-differentiated parts of large scale HW/SW systems.



# Courses

- CS448h - Domain-specific Languages for Graphics, Imaging, and Beyond [[link]( http://cs448h.stanford.edu/)]
  - Designing intermediate representations;
  - IR design, transformations, and code generation;
- CSE 599W: System for ML [[link](https://dlsys.cs.washington.edu/)]
