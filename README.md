# Table of contents

- [Comprehensive Survey](#comprehensive-survey)
- [Halide](#halide)
- [TVM](#tvm)
  - [Basics](#basics)
  - [Tensor IR and Schedule](#tensor-ir-and-schedule)
  - [Quantization](#quantization)
  - [Build Flow](#build-flow)
  - [Runtime](#runtime)
  - [Custom Accelarator Support](#custom-accelarator-support)
- [MLIR](#mlir)
- [Hardware](#hardware)
- [Talks](#talks)
- [Courses](#courses)
- [TEAM](#team)

# Comprehensive Survey

- Deep Learning Compilers [[slide](https://ucbrise.github.io/cs294-ai-sys-sp19/assets/lectures/lec12/dl-compilers.pdf)]
  - 说明了DL Compiler领域需要解决的问题；
  - 强调了Halide的短板和TVM，TC要解决的问题；
- The Evolution of Domain-Specific Computing for Deep Learning [[paper](https://ieeexplore.ieee.org/abstract/document/9439420/)]
  - Xilinx Lab所撰写的survey paper；
  - 里面对AIE Engine做了一些介绍；
  - 对MLIR生成AIE code做了简单介绍；
- Compute Substrate for Software 2.0 [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9373921)]

- A New Golden Age for Computer Architecture [[link](https://cacm.acm.org/magazines/2019/2/234352-a-new-golden-age-for-computer-architecture/fulltext)]

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
- TVM Codebase Walkthrough by Example [[doc](https://tvm.apache.org/docs/dev/codebase_walkthrough.html)]
  - 官方文档

## Tensor IR and Schedule

- TVM之Tensor数据结构解读 [[zhihu](https://zhuanlan.zhihu.com/p/341257418)]
  - 对于Tensor，Operation之间的关系做了比较好的介绍；
- TVM中的IR设计与技术实现 [[blog](https://www.cnblogs.com/CocoML/p/14643355.html)]
  - 对IRModule和Module做了非常专业的介绍
- [IR] Unified TVM IR Infra [[RFC]( https://discuss.tvm.apache.org/t/ir-unified-tvm-ir-infra/4801)]
  - 陈天奇提出的RFC, 里面提到了几个重要的改进和设计目标：
    - A unified module, pass and type system for all IR function variants.
    - Two major variants of IR expressions and functions: the high-level functional IR(relay)and the tensor-level IR for loop optimizations.
    - First-class Python and hybrid script support, and a cross-language in-memory IR structure.
    - A unified runtime::Module to enable extensive combination of traditional devices, microcontrollers and NPUs.
  - 他提出的这个RFC是目前TVM的基石，很厉害！
- TVM schedule: An Operational Model of Schedules in Tensor Expression [[doc](https://docs.google.com/document/d/1nmz00_n4Ju-SpYN0QFl3abTHTlR_P0dRyo5zsWC0Q1k/edit)]
- Ansor: Generating High-Performance Tensor Programs for Deep Learning [[pdf](https://arxiv.org/pdf/2006.06762.pdf)]
- Contributing new docs for InferBound [[DISCUSS](https://discuss.tvm.apache.org/t/discuss-contributing-new-docs-for-inferbound/2151)]
  - 对Infer Bound pass的内核做了很好的介绍，对Schedule抽象也做了部分介绍；
  - 对应的官方文档: https://tvm.apache.org/docs/dev/inferbound.html?highlight=inferbound
- 也谈TVM和深度学习编译器 [[知乎](https://zhuanlan.zhihu.com/p/87664838)]
  - 里面有Tensor IR的语法定义；

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
- Bring Your Own Codegen to Deep Learning Compiler [[Paper](https://arxiv.org/pdf/2105.03215.pdf)]
- TVM Runtime System [[doc](https://tvm.apache.org/docs/dev/runtime.html)]
- Device/Target Interactions [[doc](https://tvm.apache.org/docs/dev/device_target_interactions.html)]

## Custom Accelarator Support

- Feedback on TVM port to custom accelerator [[discuss](https://discuss.tvm.apache.org/t/feedback-on-tvm-port-to-custom-accelerator/9548)]
  - 介绍了为啥不使用BYOC框架及不参考VTA的详细原因；

> ...
>
> - We have looked into using BYOC, but we felt like this was a very direct mapping of Relay to instructions, which bypasses a lot of scheduling/optimization magic (Tensor Expressions, AutoTVM) from the rest of the TVM stack. It also did not seem like a very scalable solution to us, since it seems like we would have to map a lot of Relay instructions directly to a HWLib function call, which we also have to develop ourselves.
> - We have looked into VTA, but VTA is quite different from our platform. We don’t have a fully fledged workstation host device at hand, apart from the bare metal microcontroller. Also we would like to compile as much as possible statically and AoT, and not in a JIT-fashion. Maybe there are some accelerator specific parts we can reuse though. If someone can share their experience on reusing some of this work that would be very insightful!
>
> ...
>
> > Is tensorization an option here, or do you need to do more with the TIR after schedule generation?
>
> Yes, i’m currently trying to use tensorization to map entire convolutions and data preparation steps (data layout, padding) to a HWLib function call, but the process hasn’t been particularly smooth for such coarse computations i’m afraid. [Getting data to be transformed from TVM seems suboptimal.](https://discuss.tvm.apache.org/t/te-using-reshape-without-copy/9480) Also creating large tensorization intrinsics is tricky; Right now for example it looks like I would have to generate a separate TIR pass, because I can not merge e.g.`Relu(Conv(Pad(ChgDataLayout(input)),filter))` into one intrinsic; [tensorize/tir does not allow for creating an intrinsic with nested computations 3](https://discuss.tvm.apache.org/t/tensorize-how-to-use-tensorize-for-composition-op-eg-conv-relu/2336) The TIR pass i’m envisioning could detect those sequential operations and maybe merge them into one as a workaround for this problem.

- Which is the best way to port tvm to a new AI accelerator? [[discuss](https://discuss.tvm.apache.org/t/which-is-the-best-way-to-port-tvm-to-a-new-ai-accelerator/6905)]

  > 1 ： BYOC, BYOC can offload the ops to your new device which your new device support. BYOC is simple and graceful, But we can’t use AutoTVM in BYOC. I think AutoTVM is the very import feature of TVM.
  >
  > 2 : Tensorize, By using TVM’s schedule primitive Tensorize, we can replace a unit of computation with the corresponding intrinsic, such as GEMM instruction. We can use AutoTVM in this way, but we may need to use tensorize to modify very ops’s schedule.
  >
  > 3 : like cuDNN, we can use tvm to call new device like use cuDNN to call GPU. this way is not better than BYOC
  >
  > 4 : like GPU/CPU, we can add a new target in tvm like GPU/CPU, we need develop compute and schedule for every op, we also need to develop graph optimize for this new device. we can use AutoTVM in this way. But this way is the most time-consuming and the most difficult
  >
  > I think if we only have the op level’s api of new device, BYOC is the best way.
  >
  > If we have ISA level’s interface of new device, which way is the best?

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



# TEAM

- Zhang Research Group - Accelerating Design of Future Computing Systems : https://zhang.ece.cornell.edu/index.html
