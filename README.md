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
