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
- The Deep Learning Compiler: A Comprehensive Survey [[link](https://arxiv.org/abs/2002.03794)]
- Efficient Execution of Quantized Deep Learning Models: A Compiler Approach [[link]( https://arxiv.org/abs/2006.10226)]
- Comparison of Deep Learning Frameworks and Compilers [[link]( https://www.researchgate.net/profile/Hadjer_Benmeziane/publication/343320122_Master_Thesis_-_Comparison_of_Deep_Learning_Frameworks_and_Compilers/links/5f22f86b299bf1340492ff51/Master-Thesis-Comparison-of-Deep-Learning-Frameworks-and-Compilers.pdf)]
- Why are ML Compilers so Hard?  [[blog](https://petewarden.com/2021/12/24/why-are-ml-compilers-so-hard/)]
- Albert Cohen: Herding Tensor Compilers [[video](https://www.youtube.com/watch?v=3YKogLrk6Rg)]
  - Topics
    - Compiling graphs of tensor operations
      - Jax + XLA

    - Meta-programming and autotuning
      - tvm, FLAME/BLIS

    - Tensor compiler construction
      - MLIR

  - recall algebraic principles supporting the compilation of tensor algebra, and illustrate these principles on three optimization strategies with different degrees of human/expert intervention. 
  - also discuss MLIR, a large-scale compiler construction effort to rationalize the landscape of machine learning systems.
  - 都是对现有的tensor compiler一些简单的介绍，没有insights。




# IR Design

- Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning Workloads [[paper](https://arxiv.org/pdf/2104.05755.pdf)]
  - slides: https://drive.google.com/file/d/1b8gXD0cdAwl1mxtzb4V1joIOP-exquhv/view
- IR Design for Heterogeneity: Challenges and Opportunities [[link](https://conf.researchr.org/getImage/CC-2020/orig/IR+Design+for+Heterogeneity+-+Challenges+and+Opportunities.pdf)]
- Compilers and IRs: LLVM IR, SPIR-V, and MLIR [[blog](https://www.lei.chat/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/)]

> 换言之，LLVM IR 天然中心化并且偏好统一的编译流程，MLIR 的基础设施和 dialect 生态则天然是去中心化并且偏好离散的编译流程。
>
> ...
>
> 技术栈的底层一般相对稳定。少数几种硬件架构、编译器和操作系统统治很多年。 但半导体进展的变慢和计算需求的爆炸式增长也在驱动着底层技术的变革。 现在依然依靠通用架构和普适优化很难再满足各种需求，开发领域专用的整体的解决方案是一条出路。 RISC-V 在芯片指令集层次探索模块化和定制化，MLIR 则是在编译器以及中间表示层面做类似探索。 两者联手会给底层技术栈带来何种革新是一个值得拭目以待的事情。

- Fireiron: A Scheduling Language for High-Performance Linear Algebra on GPUs [[link](https://arxiv.org/abs/2003.06324)]
  - High-Performance Domain-Specific Compilation without Domain-Specific Compilers [[link](https://bastianhagedorn.github.io/files/publications/2020/thesis.pdf)]
- SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning [[paper](https://arxiv.org/pdf/2207.04606.pdf)]




# Halide

- Decoupling Algorithms from Schedules for Easy Optimization of Image Processing Pipelines [[paper](https://people.csail.mit.edu/jrk/halide12/halide12.pdf)] 
- MIT 6.815/6.865 [[link](https://stellar.mit.edu/S/course/6/sp15/6.815/materials.html)]
  - 对Halide的核心思想，调度原语做了很好的介绍，非常多的例子，具有很好的参考价值；
- Vyasa: A High-Performance Vectorizing Compiler for Tensor Convolutions on the Xilinx AI Engine [[link](https://arxiv.org/abs/2006.01331)] 



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
- Unified TVM IR Infra [[RFC]( https://discuss.tvm.apache.org/t/ir-unified-tvm-ir-infra/4801)]
  - 陈天奇提出的RFC, 里面提到了几个重要的改进和设计目标：
    - A unified module, pass and type system for all IR function variants.
    - Two major variants of IR expressions and functions: the high-level functional IR(relay)and the tensor-level IR for loop optimizations.
    - First-class Python and hybrid script support, and a cross-language in-memory IR structure.
    - A unified runtime::Module to enable extensive combination of traditional devices, microcontrollers and NPUs.
  - 他提出的这个RFC是目前TVM的基石，很厉害！
- Introduction to TOPI [[doc](https://tvm.apache.org/docs/tutorials/topi/intro_topi.html#generic-schedules-and-fusing-operations)]

## Relay

- Jared Roesch's PHD thesis [[phd thesis](https://digital.lib.washington.edu/researchworks/handle/1773/46765)]
  - 融合实现及怎么schedule fuse op的思路
  - 定点的实现；
  - relay op的runtime实现（待细看）
- Relay: A New IR for Machine Learning Frameworks [[paper](https://arxiv.org/pdf/1810.00952.pdf)]
  - 最早期的论文，不够详细；
- Relay: A High-Level Compiler for Deep Learning [[paper](https://arxiv.org/abs/1904.08368)]
  - 相比于早期的论文，写的很详细，与PHD thesis里面的很多是一致的；
- TVM之Relay IR实践 [[zhihu](https://zhuanlan.zhihu.com/p/339348734)]

## Optimizations

### Fusion

- TVM代码走读（六） 图优化4-- Fuse ops [[zhihu](https://zhuanlan.zhihu.com/p/153098112)]
- TVM系列 - 图优化 - 算子融合 [[blog](https://blog.csdn.net/Artyze/article/details/108796250)]
- Jared Roesch's PHD thesis [[phd thesis](https://digital.lib.washington.edu/researchworks/handle/1773/46765)]
  - 里面介绍了怎么给fused_op做schedule的思路
  - 具体tvm的实现代码为：能inline的话就inline（traverse_inline），否则就是root；
- Operator fusing with AutoTVM on GPU [[disscuss](https://discuss.tvm.apache.org/t/operator-fusing-with-autotvm-on-gpu/3725)]
- Regarding Operator Fusion [[discuss](https://discuss.tvm.apache.org/t/regarding-operator-fusion/8657)]

> “Op fusion happens later in the pipeline. AutoTVM extracts tuning tasks from a graph before fusion, so it only looks at individual op (conv, dense etc).
>
> Our fusion rule is not hardware dependent (for now). Both CPU and GPU backend get the same fused operator. We only fuse cheap ops into convolution, dense etc, with the assumption that a tuned convolution schedule is also optimal if it is fused with other cheap ops. That allows autotvm tuning and fusion be done independently.”

### ConvertLayout

- Convert Layout Pass [[doc](https://tvm.apache.org/docs/dev/convert_layout.html)]

### AutoTvm

- Search-space and Learning-based Transformations [[doc](https://tvm.apache.org/docs/dev/index.html#transformations)]



## Tensor IR and Schedule

- TVM之Tensor数据结构解读 [[zhihu](https://zhuanlan.zhihu.com/p/341257418)]
  - 对于Tensor，Operation之间的关系做了比较好的介绍；
- TVM中的IR设计与技术实现 [[blog](https://www.cnblogs.com/CocoML/p/14643355.html)]
  - 对IRModule和Module做了非常专业的介绍
- TVM schedule: An Operational Model of Schedules in Tensor Expression [[doc](https://docs.google.com/document/d/1nmz00_n4Ju-SpYN0QFl3abTHTlR_P0dRyo5zsWC0Q1k/edit)]
- Ansor: Generating High-Performance Tensor Programs for Deep Learning [[pdf](https://arxiv.org/pdf/2006.06762.pdf)]
- Contributing new docs for InferBound [[DISCUSS](https://discuss.tvm.apache.org/t/discuss-contributing-new-docs-for-inferbound/2151)]
  - 对Infer Bound pass的内核做了很好的介绍，对Schedule抽象也做了部分介绍；
  - 对应的官方文档: https://tvm.apache.org/docs/dev/inferbound.html?highlight=inferbound
- 也谈TVM和深度学习编译器 [[知乎](https://zhuanlan.zhihu.com/p/87664838)]
  - 里面有Tensor IR的语法定义；



## Tensorization

- How to use tensorize for composition op(eg. conv+relu) [[discuss](https://discuss.tvm.apache.org/t/tensorize-how-to-use-tensorize-for-composition-op-eg-conv-relu/2336)]



## Quantization

- The quantization story for TVM [[web](https://discuss.tvm.apache.org/t/quantization-story/3920)]
- Compilation of Quantized Models in TVM [[pdf](http://lenlrx.cn/wp-content/uploads/2019/11/Nov8_TVM_meetup_Quantization.pdf)]



## Compilation Flow

- TVM Relay Build Flow [[link](https://zhuanlan.zhihu.com/p/257150960)]
- TVM的“hello world“基础流程（上）[[link](https://blog.csdn.net/jinzhuojun/article/details/117135551)]
  - 写的非常专业，对一些TE/TIE中的python/C++类做了介绍；
- TVM/VTA代码生成流程 [[blog](https://krantz-xrf.github.io/2019/10/24/tvm-workflow.html)]
  - 流程图画的还挺清晰
- TVM的编译流程详解 [[LINK](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247494801&idx=1&sn=b893c43133eea1343034bb0aca356e24&scene=21#wechat_redirect)]
- TVM的CodeGen流程[[LINK](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247495638&idx=1&sn=ca711f6f33e6f83ed9ec27afddb416a0&chksm=9f835540a8f4dc56f3fe77b2d4aefd211220bd61bc3f2bbe03ae425478309d831f339ed6b39a&scene=178&cur_album_id=1788791560017346569#rd)]
- The lifecycle of opt_gemm in tvm [[blog](https://wenxiaoming.github.io/2020/01/12/The%20lifecycle%20of%20opt_gemm%20in%20tvm/)]
  - 非常详细介绍了PackFunc实现机制；
  - 介绍了opt_gemm的执行流程；



## Runtime

### Basics

- TVM Runtime System [[doc](https://tvm.apache.org/docs/dev/runtime.html)]
- Device/Target Interactions [[doc](https://tvm.apache.org/docs/dev/device_target_interactions.html)]

### Heterogeneous

- Heterogeneous execution in Relay VM [[RFC](https://github.com/apache/tvm/issues/4178)]

> ## Current Design in Relay Graph Runtime
>
> ### Compilation
>
> Reference: [#2361](https://github.com/apache/tvm/pull/2361)
>
> Summary: If users want to specify a device for an operator to run on, they can use an annotation operator named `on_device(expr, dev_id)` to wrap an expression. At a step `RunDeviceAnnotationPass` during `relay.build`, we will replace `on_device` node with `device_copy` node. At the step of `PasGraphPlanMemory` , we compute the device assignment(`device_type` see next section) of each memory block. This is possible because graph runtime only support static graph, so we can capture all the information statically. Then during native code generation, `device_copy` node is mapped to special packed function named `__copy`.
>
> ### Runtime
>
> Reference: [#1695](https://github.com/apache/tvm/pull/1695)
>
> Summary: In the graph json file, a new field named `device_type` specifies which device a static memory node should be scheduled to, the runtime allocates the memory in on the device accordingly. When graph runtime sees special operator named `__copy`, it calls `TVMArrayCopyFromTo` to move memory across devices correctly.

- Graph partitioning and Heterogeneous Execution [[RFC](https://discuss.tvm.apache.org/t/graph-partitioning-and-heterogeneous-execution/504)]

### Dynamic Support

- Extending TVM with Dynamic Execution [[slides](https://tvmconf.org/slides/2019/Jared-Roesch-Haichen-Shen-RelayVM.pdf)]
- Jared Roesch's PHD thesis [[phd thesis](https://digital.lib.washington.edu/researchworks/handle/1773/46765)]
- Relay Dynamic Runtime [[RFC](https://github.com/apache/tvm/issues/2810)]

### BYOC

- Bring Your Own Codegen to Deep Learning Compiler [[Paper](https://arxiv.org/pdf/2105.03215.pdf)]
  - BYOC框架的原始论文；
  - 值得注意的是，里面提到了MetaData runtime module的作用，代码看不懂可以参考这个；
- How to Bring Your Own Codegen to TVM [[blog](https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm)]



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
  > 2 : Tensorize, By using TVM’s schedule primitive Tensorize, we can replace a unit of computation with the corresponding intrinsic, such as GEMM instruction. We can use AutoTVM in this way, but we may need to use tensorize to modify every ops’s schedule.
  >
  > 3 : like cuDNN, we can use tvm to call new device like use cuDNN to call GPU. this way is not better than BYOC
  >
  > 4 : like GPU/CPU, we can add a new target in tvm like GPU/CPU, we need develop compute and schedule for every op, we also need to develop graph optimize for this new device. we can use AutoTVM in this way. But this way is the most time-consuming and the most difficult
  >
  > I think if we only have the op level’s api of new device, BYOC is the best way.
  >
  > If we have ISA level’s interface of new device, which way is the best?



## VTA

- VTA Technical Report [[link]( https://tvm.apache.org/2018/07/12/vta-release-announcement.html)]
- VTA Install Guide [[link]( https://tvm.apache.org/docs/vta/install.html)]
- VTA Hardware Guide [[link]( https://tvm.apache.org/docs/vta/dev/hardware.html)]
- [VTA&TVM\] Questions after investigating resnet.py tutorial [[disscuss](https://discuss.tvm.apache.org/t/vta-tvm-questions-after-investigating-resnet-py-tutorial/1649)]



# XLA

- Accelerated linear algebra compiler for computationally efficient numerical models: Success and potential area of improvement [[paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282265)]

- Deep Dive into XLA (Draft)  [[blog](https://minjae.blog/2020/02/29/xla-compiler.html)]

- OpenXLA

  - repo: https://github.com/openxla
  - RFC: Establish SIG OpenXLA [[rfc](https://github.com/tensorflow/community/pull/419)]
  - Announcing OpenXLA (also moving tensorflow/compiler/mlir/hlo) [[link](https://groups.google.com/a/tensorflow.org/g/mlir/c/ovMNwqloJQc)]

  

# MLIR

## Basics

- MLIR: A Compiler Infrastructure for the End of Moore's Law [[paper](https://arxiv.org/abs/2002.11054)]
- Multi-Level Intermediate Representation Compiler Infrastructure [[slide](https://docs.google.com/presentation/d/11-VjSNNNJoRhPlLxFgvtb909it1WNdxTnQFipryfAPU/edit#slide=id.g7d334b12e5_0_4)]
- MLIR: Scaling Compiler Infrastructure for Domain Specific Computation [[paper](https://research.google/pubs/pub49988/)]
- Thoughts on Tensor Code Generation in MLIR [[video](https://drive.google.com/file/d/1PKY5yVEL0Dl5UHaok4NgpxnbwXbi5pxS/view)] [[slide](https://docs.google.com/presentation/d/1M44If0Lw2lnrlyE_xNU1WOmXWxLo9FibMwdUbrAhOhU/edit#slide=id.g5fd22bdf8c_0_0)]
- Polyhedral Compilation Opportunities in MLIR [[slide](http://impact.gforge.inria.fr/impact2020/slides/IMPACT_2020_keynote.pdf)]
- Abstraction raising in MLIR [[master thesis](https://pure.tue.nl/ws/portalfiles/portal/175414453/Komisarczyk_K..pdf)]
- An overview of loop nest optimization, parallelization and acceleration in MLIR [[link]( http://www.cs.utah.edu/~mhall/mlir4hpc/cohen-MLIR-loop-overview.pdf)]
- MLIR: an Agile Infrastructure for Building a Compiler Ecosystem [[slide](https://llvm-hpc-2020-workshop.github.io/presentations/llvmhpc2020-amini.pdf)]
- 2020 LLVM Developers’ Meeting: M. Amini & R. Riddle “MLIR Tutorial” [[video](https://www.youtube.com/watch?v=Y4SvqTtOIDk)]



## Dialects and Design

- Codegen Dialect Overview [[link](https://discourse.llvm.org/t/codegen-dialect-overview/2723)]
  - MLIR体系中各个dialect的关系

- Linalg Dialect Rationale: The Case For Compiler-Friendly Custom Operations [[doc](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/)]

  - 总结了linalg的设计动机

  - 总结了从很多其他ir设计中学到的经验，包括onnx，lift，xla，halide and tvm，等等，非常有参考价值

- MLIR Open Meeting 2022-01-27: Introduction to Linalg.generic [[video](https://www.youtube.com/watch?v=A805W2KSCxQ)]
  - a nice introduction to Linalg and Linalg.generic concepts

- Development of high-level Tensor Compute Primitives dialect(s) and transformations [[link](https://discourse.llvm.org/t/development-of-high-level-tensor-compute-primitives-dialect-s-and-transformations/388)]

> Jumping a bit into the technical discussion, I think that one of the issues with HLO specifically is that it combines opinions on a few different axes that don’t necessarily need to be combined in new work.
>
> 1. Static shapes (versus a full story for dynamic)
> 2. Implicit vs explicit broadcast semantics
> 3. Functional control flow (vs CFG)
> 4. Preference for “primitive” vs high-level math ops
> 5. Preference for explicit reductions vs high-level aggregate ops

- “TCP” Is More Than TCP [[link](https://drive.google.com/file/d/1iljcpTQ5NPaMfGpoPDFml1XkYxjK_6A4/view)]
- Compiler Support for Sparse Tensor Computations in MLIR [[link](https://arxiv.org/pdf/2202.04305.pdf)]
- Structured Ops in MLIR- Compiling Loops, Libraries and DSLs [[slides](https://docs.google.com/presentation/d/1P-j1GrH6Q5gLBjao0afQ-GfvcAeF-QU4GXXeSy0eJ9I/edit#slide=id.p)]
  - 介绍了structure ops的背后的设计思想，暂时还没有领会～

- An overview of loop nest optimization, parallelization and acceleration in MLIR [[slides](http://www.cs.utah.edu/~mhall/mlir4hpc/cohen-MLIR-loop-overview.pdf)]
  - 介绍了Affine dialect和linalg dialect的example
  - 介绍了Linalg Rationale




## GPU & GEMM

- Performance Analysis of Tiling for Automatic Code Generation for GPU Tensor Cores using MLIR [[link](http://mcl.csa.iisc.ac.in/theses/Vivek_MTech_Thesis.pdf)]
  - slides: https://mlir.llvm.org/OpenMeetings/2021-08-26-High-Performance-GPU-Tensor-CoreCode-Generation-for-Matmul-Using-MLIR.pdf
  - High Performance Code Generation in MLIR: An Early Case Study with GEMM [[paper](https://arxiv.org/pdf/2003.00532.pdf)]
- High Performance GPU Code Generation for Matrix-Matrix Multiplication using MLIR: Some Early Results [[pdf](https://arxiv.org/pdf/2108.13191.pdf)]
- Polyhedral Compilation Opportunities in MLIR [[slide](http://impact.gforge.inria.fr/impact2020/slides/IMPACT_2020_keynote.pdf)]



## End2End Flow

- Compiling ONNX Neural Network Models Using MLIR [[paper](https://arxiv.org/pdf/2008.08272.pdf)] [[github](https://github.com/onnx/onnx-mlir)]
- Does IREE support custom ML accelerators? [[issue](https://github.com/google/iree/issues/6720)]
- Composable and Modular Code Generation in MLIR [[link](https://arxiv.org/pdf/2202.03293.pdf)]
  - iree的实现基本体现了该论文中的design；
  - 这个论文很精辟的总结了不同AI编译器的优缺点
  - 更新一下我对深度学习编译器和框架的认识 [[zhihu](https://zhuanlan.zhihu.com/p/474324656)]



## IREE

- MLIR CodeGen Dialects for Machine Learning Compilers [[link](https://www.lei.chat/posts/mlir-codegen-dialects-for-machine-learning-compilers/)]

> - 说明了MLIR dialects的特点和目前MLIR dialect的体系
> - 非常好的一片blog，深入浅出的阐明了mlir中不同层级dialect的必要性和各个不同dialect承担的职能；

- CodeGen Performant Convolution Kernels for Mobile GPUs [[link](https://www.lei.chat/posts/codegen-performant-convolution-kernels-for-mobile-gpus/)]

> - This blog post talks about how to generate performant code for convolution ops using MLIR’s multiple levels of abstractions and transformations.
> - 以IREE为例阐述了convolution的code gen过程。

- Composable and Modular Code Generation in MLIR [[link](https://arxiv.org/pdf/2202.03293.pdf)]
  - iree的实现基本体现了该论文中的design；
  - 这个论文很精辟的总结了不同AI编译器的优缺点
  - 更新一下我对深度学习编译器和框架的认识 [[zhihu](https://zhuanlan.zhihu.com/p/474324656)]
- IREE CodeGen [[slides](https://docs.google.com/presentation/d/1NetHjKAOYg49KixY5tELqFp6Zr2v8_ujGzWZ_3xvqC8/edit#slide=id.g91bae7fd94_1_0)]
- MLIR-based End-to-End ML Tooling [[slides](https://docs.google.com/presentation/d/1RCQ4ZPQFK9cVgu3IH1e5xbrBcqy7d_cEZ578j84OvYI/edit#slide=id.g6e31674683_0_0)]



# Dynamic Shape Support

- DISC: https://github.com/alibaba/BladeDISC
  - BladeDISC：动态Shape深度学 习编译器实践 [[pdf](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/BladeDISC%EF%BC%9A%E5%8A%A8%E6%80%81Shape%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8%E5%AE%9E%E8%B7%B5%E7%9A%84.pdf)]



# ONNXRuntime

- ONNXRuntime源码阅读 [[blog](https://www.cnblogs.com/xxxxxxxxx/category/2124139.html)]



# Accelerator Design

- Union: A Unified HW-SW Co-Design Ecosystem in MLIR for Evaluating Tensor Operations on Spatial Accelerators [[pdf](https://arxiv.org/pdf/2109.07419.pdf)]



# Other AI Tensor Compilers

- LoopStack: a Lightweight Tensor Algebra Compiler Stack [[link](https://arxiv.org/pdf/2205.00618.pdf)]
- HPVM: Heterogeneous Parallel Virtual Machine [[link](https://publish.illinois.edu/hpvm-project/)]
- Introducing Triton: Open-Source GPU Programming for Neural Networks [[link](https://openai.com/blog/triton/)]
  - PHD Thesis: https://dash.harvard.edu/bitstream/handle/1/37368966/ptillet-dissertation-final.pdf?sequence=1&isAllowed=y 

- Roller: Fast and Efficient Tensor Compilation for Deep Learning [[link](https://www.microsoft.com/en-us/research/publication/roller-fast-and-efficient-tensor-compilation-for-deep-learning/)]
- Fireiron: A Scheduling Language for High-Performance Linear Algebra on GPUs [[link](https://arxiv.org/abs/2003.06324)]
  - Author page: [[link]( https://bastianhagedorn.github.io/)]
  - Talk: High-Performance Domain-Specific Compilation without Domain-Specific Compilers [[link](https://bastianhagedorn.github.io/talks/2020-08-viva)]
  - Phd thesis: [[link](https://bastianhagedorn.github.io/files/publications/2020/thesis.pdf)]

> 我觉得fireiron有几个问题：
>
> - Fireiron和lift都是用scala写的，首先从上手来讲就比较麻烦，不利于工程界使用。
> - Fireiron和lift语言到底和halide/tvm这些语言有什么本质的不同，有什么明显的优势吗？从lift的ir来看，这个语言和tvm相比，本身不是为machinling learning时代而生的：
>
> Lift: IR 
>
> ▣ Data types : Int, Float8 / Float16 / Float32, Arrays
>
> ▣ Algorithmic patterns: Map, Slide, Reduce, Zip, Join, Split 
>
> ▣ Address space operators: toChip, toDram, toOutput 
>
> ▣ Arithmetic operators: ScalarAdd, VVAdd, MVAdd, MMAdd, ScalarMul, VVMul, MVMul, MMMul, VVRelu, VVTanh

- Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning [[slides](https://docs.google.com/presentation/d/1CQ4S1ff8yURk9XmL5lpQOoMMlsjw4m0zPS6zYDcyp7Y/edit#slide=id.p)]
  - automatic model-parallel training system, built on top of a hierarchical view of ML parallelization methods: inter-op and intra-op.
- Hummingbird: A Tensor Compiler for Unified Machine Learning Prediction Serving [[link](https://www.usenix.org/conference/osdi20/presentation/nakandala)]
  - *Hummingbird* is a library for compiling trained traditional ML models into tensor computations. *Hummingbird* allows users to seamlessly leverage neural network frameworks (such as [PyTorch](https://pytorch.org/)) to accelerate traditional ML models. 



# Enhanced Compilers for GPU

- Numba: https://numba.pydata.org/
  - Difference with respect to numba.cuda? https://github.com/openai/triton/issues/160

> Triton sits somewhere between `Taichi` and `numba.cuda`. Numba is more like a `one-to-one` mapping to CUDA, so you still have to explicitly optimize things like memory coalescing, shared memory bank conflicts, sram prefetching, etc. by yourself. Also note that Numba also doesn't support FP16 (yet) so it is better suited for scientific computing than DNNs



# Miscellaneous

- AI框架算子层级的思考 [[zhihu](https://zhuanlan.zhihu.com/p/388682140)]
- tvm or mlir ？[[zhihu](https://zhuanlan.zhihu.com/p/388452164)]
- WAIC 2021 深度学习编译框架前沿技术闭门论坛 [[link](https://gitee.com/MondayYuan/WAIC-DLCompiler-MeetingMinutes)]
- Why GEMM is at the heart of deep learning [[blog](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)]
- DLGR: A Rule-Based Approach to Graph Replacement for Deep Learning [[link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9763757)]



# Talks

- The Golden Age of Compilers, in an era of Hardware/Software co-design [[slide](https://docs.google.com/presentation/d/1ZMtzT6nmfvNOlIaHRzdaXpFeaAklcT7DvfGjhgpzcxk/edit#slide=id.p)] [[video](https://drive.google.com/file/d/1eIxFZZLOM7a3LYL1QaKhflKl0jRLPp-V/view)]
  - A discussion about accelerator design, benefits of reducing fragmentation by standardizing non-differentiated parts of large scale HW/SW systems.



# Courses

- CS448h - Domain-specific Languages for Graphics, Imaging, and Beyond [[link]( http://cs448h.stanford.edu/)]
  - Designing intermediate representations;
  - IR design, transformations, and code generation;
- CSE 599W: System for ML [[link](https://dlsys.cs.washington.edu/)]
- Machine Learning Compilation [[link](https://mlc.ai/summer22/schedule)]
  - courses by Chen Tianqing using TVM




# TEAM

- SAMPL: http://sampl.cs.washington.edu/publications.html
- PolyMage Labs: https://www.polymagelabs.com/



# Person/Developer Blog

- Lei.Chat(): developer of IREE, https://www.lei.chat/zh/
  - Compilers and IRs: LLVM IR, SPIR-V, and MLIR [[link](https://www.lei.chat/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/)]
  - CodeGen Performant Convolution Kernels for Mobile GPUs [[link](https://www.lei.chat/posts/codegen-performant-convolution-kernels-for-mobile-gpus/)]
- Bastian Hagedorn: https://bastianhagedorn.github.io/
  - High-Performance Domain-Specific Compilation without Domain-Specific Compilers [[link](https://bastianhagedorn.github.io/files/publications/2020/thesis.pdf)] [[talk](https://bastianhagedorn.github.io/talks/2020-08-viva)]
