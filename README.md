CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Harris Kokkinakos
  * [LinkedIn](https://www.linkedin.com/in/haralambos-kokkinakos-5311a3210/), [personal website](https://harriskoko.github.io/Harris-Projects/)
* Tested on: Windows 24H2, i9-12900H @ 2.50GHz 16GB, RTX 3070TI Mobile

### Project Overview
This project implements and analyzes multiple approaches to two fundamental parallel computing primitives: prefix sum (scan) and stream compaction. The implementation compares CPU-based sequential algorithms against various GPU-accelerated approaches using CUDA, providing insights into parallel algorithm design and performance characteristics across different problem sizes.

### Algorithms Implemented
Four different implementations of exclusive prefix sum were developed and analyzed:
1. CPU Sequential Scan: Traditional single-threaded implementation serving as baseline
2. Naive GPU Scan: Direct parallelization using the inclusive-to-exclusive scan approach with ping-pong buffers
3. Work-Efficient GPU Scan: Implementation of the Blelloch scan algorithm using up-sweep and down-sweep phases, based on the approach described in GPU Gems 3
4. Thrust Library Scan: Leveraging NVIDIA's optimized Thrust library implementation

The Work-Efficient GPU Scan was used to implement Stream Compaction for removing zero's from an array of random integers. The algorithm follows the pipeline shown below:
1. Map Phase: Convert input elements to boolean mask (0→0, non-zero→1)
2. Scan Phase: Perform exclusive scan on boolean mask to determine output indices
3. Scatter Phase: Copy non-zero elements to their computed output positions

### Performance Analysis
Scan Algorithm Comparisons
![Alt text](img/all.png "Optional title")

The graph above compares the runtime of each of the implemented algorithms with increasing input array sizes. In almost all cases, Thrust performs the best, which is expected as it is a baseline for this project and is an optimized library developed by NVidia. In the case of 2²⁰ array elements, the Naive implementation does compute the scan result faster than thrust. This is likely due to memory optimizations which are done in my naive implementation. Their performance is extremely similar in this instance which is logical. Comparing to the three implementations written for this project, we see two major points of interest in this graph. The first is between 2¹⁶ and 2²⁰, where Naive begins to out-perform the CPU implementation. Initially, the CPU implementation performs better as a sequential based approach on the CPU will run quicker on lower array sizes. GPU implementations have the overhead of PCIE data transfer and global memory reads which greatly increases the runtime of the algorithms but speed up the arithmetic. Thus, in cases of smaller arrays, arithmetic is not the source of the most overhead, causing the GPU implementations to be slower. However, We see that between the first two array element sizes, the GPU implementation begins to perform better. This is the array input size threshold where the arithmetic begins to have more overhead. The second interesting point on this graph is between 2²⁰ and 2²⁴. This is where the Work Efficient algorithm begins to perform better than the Naive algorithm. The Work Efficient Implementation must spawn double the amount of kernels than the Naive Implementation. This overhead likely causes the increase in runtime compared to Naive until we see the arithmetic become the largest source of overhead as the number of elements increases. The testing for Naive was performed with kernel sizes of 128 and the Naive uses a block size of 256. These were found to be the best performing kernel size for my device as shown below. 

Scan Block Size Optimizations
![Alt text](img/n16.png "Optional title")
![Alt text](img/n24.png "Optional title")
![Alt text](img/we16.png "Optional title")
![Alt text](img/we24.png "Optional title")

As we see in the Naive case, runtime does not change greatly when the array input is large. However, at smaller array sizes, we see that the block size has a larger impact. Here, the block size of 1024 performs best. We can't use this as this is the only case which performs significantly worse in cases of larger arrays. Thus, we settled on 256 as it appears to have the lowest deviation within all cases. For the Work Efficient implementation, we see that block size has little to no impact on arrays with large size except in the case of 256 which is significantly worse. In smaller arrays, block size of 128 yields the shortest runtime.  

Stream Compaction Runtime


Stream Compation Block Size Optimization
