#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernNaiveScanStep(int* odata, int* idata, int n, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int* dev_odata;
            int* dev_idata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));

            cudaMemset(dev_idata, 0, sizeof(int));
            cudaMemcpy(dev_idata+1, idata, (n-1) * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int numSteps = ilog2ceil(n);
            for (int d = 1; d <= numSteps; d++) {
                int offset = 1 << (d - 1);
                int block_size = 512;
                dim3 num_blocks = (n + block_size - 1) / block_size;
                kernNaiveScanStep <<<num_blocks, block_size >> > (dev_odata, dev_idata, n, offset);
                cudaDeviceSynchronize();
                std::swap(dev_odata, dev_idata);  
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}
