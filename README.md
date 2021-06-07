# Comprehensive Survey

- Deep Learning Compilers [[slide](https://ucbrise.github.io/cs294-ai-sys-sp19/assets/lectures/lec12/dl-compilers.pdf)]
  - 说明了DL Compiler领域需要解决的问题；
  - 强调了Halide的短板和TVM，TC要解决的问题；



# Open Courses

- CS448h Domain-specific Languages for Graphics, Imaging, and Beyond: http://cs448h.stanford.edu/
  - Designing intermediate representations;
  - IR design, transformations, and code generation;



# Halide

- Decoupling Algorithms from Schedules for Easy Optimization of Image Processing Pipelines [[paper](https://people.csail.mit.edu/jrk/halide12/halide12.pdf)] 



# TVM

- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning [[paper](https://homes.cs.washington.edu/~arvind/papers/tvm.pdf)]

  

- Enabling TVM on RISC-V Architectures with SIMD Instructions [[slide](https://riscv.org/wp-content/uploads/2019/03/16.45-Enabling-TVM-on-RISC-V-Architectures-with-SIMD-Instructions-v2.pdf)]



- TVM and XLA区别: https://zhuanlan.zhihu.com/p/87664838



# MLIR

- The Evolution of Domain-Specific Computing for Deep Learning [[paper](https://ieeexplore.ieee.org/abstract/document/9439420/)]
  - Xilinx Lab所撰写的survey paper；
  - 里面对AIE Engine做了一些介绍；
  - 对MLIR生成AIE code做了介绍，但是我觉得没有啥价值：主要没有讲清楚客户侧MLIR编程的方式；



- MLIR: A Compiler Infrastructure for the End of Moore's Law [[paper](https://arxiv.org/abs/2002.11054)]



- Multi-Level Intermediate Representation Compiler Infrastructure [[slide](https://docs.google.com/presentation/d/11-VjSNNNJoRhPlLxFgvtb909it1WNdxTnQFipryfAPU/edit#slide=id.g7d334b12e5_0_4)]



- Thoughts on Tensor Code Generation in MLIR [[video](https://drive.google.com/file/d/1PKY5yVEL0Dl5UHaok4NgpxnbwXbi5pxS/view)] [[slide](https://docs.google.com/presentation/d/1M44If0Lw2lnrlyE_xNU1WOmXWxLo9FibMwdUbrAhOhU/edit#slide=id.g5fd22bdf8c_0_0)]



# Tech Talk

- The Golden Age of Compilers, in an era of Hardware/Software co-design [[slide](https://docs.google.com/presentation/d/1ZMtzT6nmfvNOlIaHRzdaXpFeaAklcT7DvfGjhgpzcxk/edit#slide=id.p)] [[video](https://drive.google.com/file/d/1eIxFZZLOM7a3LYL1QaKhflKl0jRLPp-V/view)]
  - A discussion about accelerator design, benefits of reducing fragmentation by standardizing non-differentiated parts of large scale HW/SW systems.

