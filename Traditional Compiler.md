
# Dominators and Static Single Assignment

- Compiler Design Lab 
  - SSA Construction [[link](https://compilers.cs.uni-saarland.de/projects/ssaconstr/)]
  - SSA-based Register Allocation [[link](https://compilers.cs.uni-saarland.de/projects/ssara/)]

- The Static Single Assignment Book. [[pdf](http://ssabook.gforge.inria.fr/latest/book.pdf)]

  > *A very detailed and practical book about many aspects of the SSA intermediate representation.*

- Simple and Efficient Construction of Static Single Assignment Form. Matthias Braun, Sebastian Buchwald, Sebastian Hack, Roland Leißa, Christoph Mallon, and Andreas Zwinkau. [[pdf](https://c9x.me/compile/bib/braun13cc.pdf)]

  > *Combining laziness and memoization, the authors make a naive construction algorithm both efficient and correct. The algorithm is especially useful for fixing SSA invariants locally broken.*

- An Efficient Method of Computing Static Single Assignment Form. Ron Cytron, Jeanne Ferrante, Barry K. Rosen, Mark N. Wegman, and F. Kenneth Zadeck. [[pdf](https://c9x.me/compile/bib/ssa.pdf)]

  > *The paper explaining the classic construction algorithm of SSA form using domination information.*

- A Simple, Fast Dominance Algorithm. Keith D. Cooper, Tymothy J. Harvey, and Ken Kennedy. [[pdf](https://c9x.me/compile/bib/quickdom.pdf)]

  > *An algorithm to compute the domination information very practical and simple to implement. It can be slower than the more complex Lengauer-Tarjan algorithm on huge input programs.*



# Parallelization and Vectorization

- EE663: Optimizing Compilers [[link](https://engineering.purdue.edu/~eigenman/ECE663/Handouts/ece663slides.pdf)]

# Optimizations

- Dependence Analysis [[link](https://engineering.purdue.edu/~milind/ece468/2015fall/lecture-13.pdf)]
- Loop optimizations [[link](https://engineering.purdue.edu/~milind/ece468/2015fall/lecture-12.pdf)]

# Register Allocation

- Register Allocation for Programs in SSA Form [[PHD thesis](https://publikationen.bibliothek.kit.edu/1000007166/6532)]

- Linear Scan Register Allocation. Massimiliano Poletto and Vivek Sarkar. [[pdf](https://c9x.me/compile/bib/linearscan.pdf)]

  > *The classic linear scan paper, this algorithm is fairly popular in JIT compilers because it compiles quickly and generates reasonable code.*

- Linear Scan Register Allocation on SSA Form. Christian Wimmer Michael Franz. [[pdf](https://c9x.me/compile/bib/Wimmer10a.pdf)]

  > *An adaptation of the classic and fast linear scan algorithm to work on SSA form. It presents the challenges posed by SSA form pretty well.*

- Iterated Register Coalescing. Lal George and Andrew W. Appel. [[pdf](https://c9x.me/compile/bib/irc.pdf)]

  > *A classic paper for register allocation using graph coloring. It is usually said to be slow and not used in JIT compilers. It is also not often used in big compilers because not flexible enough (see below for a fix).*

- A Generalized Algorithm for Graph-Coloring Register Allocation. Michael D. Smith, Norman Ramsey, and Glenn Holloway. [[pdf](https://c9x.me/compile/bib/pcc-rega.pdf)]

  > *Graph-coloring as presented in textbooks is not suited to real computers. This paper presents various extensions to handle register classes and aliases. It is the algorithm used in PCC.*

# Code Generation

- Engineering a Simple, Efficient Code Generator Generator. Christopher W. Fraser, David R. Hanson, and Todd A. Proebsting. [[pdf](https://c9x.me/compile/bib/iburg.pdf)]

  > *This paper presents iburg, a flexible code generator generator that generates compact and simple code. It is used in the retargetable C compiler lcc.*

# Intermediate Representation(IR)

- Register Transfer Language for CRuby [[blog](https://developers.redhat.com/blog/2019/02/19/register-transfer-language-for-cruby#)]

# LLVM

- 一步步掌握 LLVM [[知乎专栏](https://www.zhihu.com/column/c_1250484713606819840)]
-  Enna1’s study notes about LLVM and Clang [[link](https://llvm-clang-study-notes.readthedocs.io/en/latest/index.html)]

# Courses

## Compiler

- Advanced course on compilers (spring 2015) [[link](https://wiki.aalto.fi/display/t1065450/Advanced+compilers+2015)]
- EE663: Optimizing Compilers [[link](https://engineering.purdue.edu/~eigenman/ECE663/Handouts/ece663slides.pdf)]
  - Parallelizing and Dependency Testing
- ***Static Program Analysis - DCC888*** [[link](https://homepages.dcc.ufmg.br/~fernando/classes/dcc888/)]
  - 包含视频讲解
  - SSA, loop optimizations, (SSA) Register allocations, automatic parallelization, Instruction Level Parallelism;
  - Other analysis;
  - 有一个LLVM介绍的[视频](https://youtube.com/playlist?list=PLDSTpI7ZVmVnvqtebWnnI8YeB8bJoGOyv)；
- CS745: Optimizing Compilers, Spring 2015 [[link](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15745-s15/www/)]
- Static Program Analysis [[link](https://cs.au.dk/~amoeller/spa/)]
  - There is a also a book involved in this class;

## Hardware

- Digital Design and Computer Architecture - Spring 2021 [[link](https://safari.ethz.ch/digitaltechnik/spring2021/doku.php?id=start)]
  - E85: Digital Electronics & Computer Architecture [[link](http://pages.hmc.edu/harris/class/e85/)]
- Digital Design and Computer Architecture Spring 2021 (252-0028-00L) [[link](https://safari.ethz.ch/digitaltechnik/spring2021/doku.php?id=start)]





# Notes 

## DCC888 Course

### Control Flow Graphs

- DAG + value numbers的概念描述的非常清晰；
- local register allocation有算法描述，也解释的很清晰；spill的算法采用的是Belady算法；

> ### The theoretically optimal page replacement algorithm
>
> The theoretically optimal page replacement algorithm (also known as OPT, [clairvoyant](https://en.wikipedia.org/wiki/Clairvoyance) replacement algorithm, or [Bélády's](https://en.wikipedia.org/wiki/László_Bélády) optimal page replacement policy)[[3\]](https://en.wikipedia.org/wiki/Page_replacement_algorithm#cite_note-3)[[4\]](https://en.wikipedia.org/wiki/Page_replacement_algorithm#cite_note-4)[[2\]](https://en.wikipedia.org/wiki/Page_replacement_algorithm#cite_note-lecture_notes_jones-2) is an algorithm that works as follows: ***when a page needs to be swapped in, the [operating system](https://en.wikipedia.org/wiki/Operating_system) swaps out the page whose next use will occur farthest in the future.*** For example, a page that is not going to be used for the next 6 seconds will be swapped out over a page that is going to be used within the next 0.4 seconds.
>
> This algorithm cannot be implemented in a general purpose operating system because it is impossible to compute reliably how long it will be before a page is going to be used, except when all software that will run on a system is either known beforehand and is amenable to static analysis of its memory reference patterns, or only a class of applications allowing run-time analysis. Despite this limitation, algorithms exist[*[citation needed](https://en.wikipedia.org/wiki/Wikipedia:Citation_needed)*] that can offer near-optimal performance — the operating system keeps track of all pages referenced by the program, and it uses those data to decide which pages to swap in and out on subsequent runs. This algorithm can offer near-optimal performance, but not on the first run of a program, and only if the program's memory reference pattern is relatively consistent each time it runs.
>
> Analysis of the paging problem has also been done in the field of [online algorithms](https://en.wikipedia.org/wiki/Online_algorithm). Efficiency of randomized online algorithms for the paging problem is measured using [amortized analysis](https://en.wikipedia.org/wiki/Amortized_analysis).



