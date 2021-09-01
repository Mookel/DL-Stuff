# DCC888 Course

## Control Flow Graphs

- DAG + value numbers的概念描述的非常清晰；
- local register allocation有算法描述，也解释的很清晰；spill的算法采用的是Belady算法；

> ### The theoretically optimal page replacement algorithm
>
> The theoretically optimal page replacement algorithm (also known as OPT, [clairvoyant](https://en.wikipedia.org/wiki/Clairvoyance) replacement algorithm, or [Bélády's](https://en.wikipedia.org/wiki/László_Bélády) optimal page replacement policy)[[3\]](https://en.wikipedia.org/wiki/Page_replacement_algorithm#cite_note-3)[[4\]](https://en.wikipedia.org/wiki/Page_replacement_algorithm#cite_note-4)[[2\]](https://en.wikipedia.org/wiki/Page_replacement_algorithm#cite_note-lecture_notes_jones-2) is an algorithm that works as follows: ***when a page needs to be swapped in, the [operating system](https://en.wikipedia.org/wiki/Operating_system) swaps out the page whose next use will occur farthest in the future.*** For example, a page that is not going to be used for the next 6 seconds will be swapped out over a page that is going to be used within the next 0.4 seconds.
>
> This algorithm cannot be implemented in a general purpose operating system because it is impossible to compute reliably how long it will be before a page is going to be used, except when all software that will run on a system is either known beforehand and is amenable to static analysis of its memory reference patterns, or only a class of applications allowing run-time analysis. Despite this limitation, algorithms exist[*[citation needed](https://en.wikipedia.org/wiki/Wikipedia:Citation_needed)*] that can offer near-optimal performance — the operating system keeps track of all pages referenced by the program, and it uses those data to decide which pages to swap in and out on subsequent runs. This algorithm can offer near-optimal performance, but not on the first run of a program, and only if the program's memory reference pattern is relatively consistent each time it runs.
>
> Analysis of the paging problem has also been done in the field of [online algorithms](https://en.wikipedia.org/wiki/Online_algorithm). Efficiency of randomized online algorithms for the paging problem is measured using [amortized analysis](https://en.wikipedia.org/wiki/Amortized_analysis).



## Data-Flow Analysis

### Liveness

- There is a path P from p to another program point p‘, where v is used.	 
- The path P does not go across any definition of v.	 

Conditions 1 and 2 determine when a variable v is alive at a program point p.

A variable is alive immediately before a program point p if and only if:	 

- It is alive immediately after p; and
- It is not redefined at p.

or

- It is used at p.	

### Available Expressions

An expression E is available immediately after a program point p if and only if:	 

- It is available immediately before p.	 
- No variable of E is redefined at p.	

or	 

- It is used at p.	 
- No variable of E is redefined at p.	

### Reaching Definitions

A definition of a variable v, at a program point p, reaches a program p', if there is a path from p to p', and this path does not cross any redefinition of v;

The propagation of information:

A definition of a variable v reaches the program point immediately after p if and only if:

- The definition reaches the point immediately before p
- variable v is not redefined at p.

or

- Variable v is defined at p;

### The Dataflow Framework

- A <u>may analysis</u> keeps tracks of facts that may happen during the execution of the program;
- A <u>must analysis</u> tracks facts that will - for sure - happen during the execution of the program.
- A <u>backward analysis</u> propagates information in the opposite direction in which the program flows;
- A <u>forward analysis</u> propagates information in the same direction in which the program flows;

The abstract semantics of a statement is given by a transfer function, Transfer functions differ if the analysis is forward or backward:

```python
OUT[S] = fs(IN[s]) --> Forward 

IN[s] = fs(OUT[s]) --> Backward
```

The transfer functions provides us with a new "interpretation" of the program. We can implement a machine that traverses the program, always fetching a given instruction, and applying the transfer function onto that instruction. The process goes on until the results produced by these transfer functions stop changing. This is abstract interpretation!



## Solving Data-Flow Analysis

### False negative and False Positive

- If, in an actual run of the program, a definition D reaches a block B, then the static analysis must say so, otherwise we will have a **false negative**. A false negative means that the analysis is wrong;
- If the static analysis says that a definition D reaches a block B, and it never does it in practice, then this is a **false positive**. False positive means that the analysis is imprecise, but not wrong.

### Chaotic Iterations: Constraints in a Bag

1. Put all of them in a bag;
2. Shake the bag breathlessly
3. Take one constraint C out of it.
4. Solve C.
5. If nothing has changed:
   - If there are constraints in the bag, go to step 3
   - else you are done;
6. else go to step 1.

### Speed-up

*Method - 1: Worklist + Reverse Post-order Ordering*

We can improve chaotic iterations with a worklist, once the worklist is empty, we are done;

- Simple version: last-in, first-out extraction/insertion
- orderings: 
  - Improve on the LIFO insertion/extraction
  - To find a better ordering, we can take a look into the constraint dependence graph
  - The dependence graph is not acyclic, hence, it does not have a topological ordering, but we may try to find a "quasi-ordering"
  - Reverse post-order
    - The node ordering that we obtain after a depth-first search in the graph

```python
# Reverse Postorder
i = numbers of nodes
mark all nodes as unvisited
while exist unvisited node h:
    DFS(h)
 
DFS(n):
    mark n as visited
    for each edge(n, m):
        if unvisited m: DFS(m)
    rPostOrder[n] = i
    i = i - 1
```

*Method-2: Worklist + Strong components + Reduced Graph*

- A strong component of a graph is a maximal subgraph with the property that there is  a path between any two nodes;

- The reduced graph Gr of a graph G is the graph that we obtain by replacing each strong component of G by a single node:

- We can order the reduced graph topologically

- We can solve constraints for each SC, following a topological ordering of the reduced graph;

### Representation of Sets

- Usually these sets can be implemented as **bit-vectors**
- Bit-vectors do well in dense analyses, in which case each program point might be associated with many variables or expressions.
