#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int* idata, int n, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            int stride = 1 << (d + 1);  // 2^(d+1)

            int next_idx = stride * (index + 1) - 1;

            if (next_idx < n) {
                idata[next_idx] += idata[next_idx - (1 << d)];
            }
        }

        __global__ void kernDownSweep(int* idata, int n, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            int stride = 1 << (d + 1);  // 2^(d+1)
            int right_idx = stride * (index + 1) - 1;
            int left_idx = right_idx - (1 << d);

            if (right_idx < n) {
                int temp = idata[left_idx];                // Save left child
                idata[left_idx] = idata[right_idx];     // Set left = right
                idata[right_idx] += temp;                  // Set right = left + right
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int padded_n = 1;
            while (padded_n < n) {
                padded_n <<= 1; // keep increasing to next power of 2 until we are above n.
            }

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, sizeof(int) * padded_n);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(dev_idata + n, 0, sizeof(int) * (padded_n - n));

            timer().startGpuTimer();
            int numSteps = ilog2ceil(padded_n);

            // Perform Upsweep
            for (int d = 0; d < numSteps;d++) {
                int num_active_elements = padded_n / (1 << (d + 1));
                int threads_per_block = 128;  
                int num_blocks = (num_active_elements + threads_per_block - 1) / threads_per_block;

                kernUpSweep <<<num_blocks, threads_per_block >>> (dev_idata, padded_n, d);
                cudaDeviceSynchronize();
            }

            // Perform Downsweep
            cudaMemset(dev_idata + padded_n - 1, 0, sizeof(int));
            for (int d = numSteps - 1;d >= 0;d--) {
                int num_active_elements = padded_n / (1 << (d + 1));
                int threads_per_block = 128;
                int num_blocks = (num_active_elements + threads_per_block - 1) / threads_per_block;

                kernDownSweep << <num_blocks, threads_per_block >> > (dev_idata, padded_n, d);
                cudaDeviceSynchronize();
            }

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            int padded_n = 1;
            while (padded_n < n) {
                padded_n <<= 1;
            }
            int* dev_idata, * dev_bools, * dev_indices, * dev_odata;
            cudaMalloc((void**)&dev_idata, sizeof(int) * padded_n);
            cudaMalloc((void**)&dev_bools, sizeof(int) * padded_n);
            cudaMalloc((void**)&dev_indices, sizeof(int) * padded_n);
            cudaMalloc((void**)&dev_odata, sizeof(int) * padded_n);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(dev_idata + n, 0, sizeof(int) * (padded_n - n));
            timer().startGpuTimer();

            // Map to Booleans - process all padded elements
            int threads_per_block = 128;
            int num_blocks_padded = (padded_n + threads_per_block - 1) / threads_per_block;
            StreamCompaction::Common::kernMapToBoolean << <num_blocks_padded, threads_per_block >> > (padded_n, dev_bools, dev_idata);

            // Scan on bools
            cudaMemcpy(dev_indices, dev_bools, sizeof(int) * padded_n, cudaMemcpyDeviceToDevice);
            int numSteps = ilog2ceil(padded_n);

            // Perform Upsweep
            for (int d = 0; d < numSteps; d++) {
                int num_active_elements = padded_n / (1 << (d + 1));
                int threads_per_block = 128;
                int num_blocks = (num_active_elements + threads_per_block - 1) / threads_per_block;
                kernUpSweep << <num_blocks, threads_per_block >> > (dev_indices, padded_n, d);
                cudaDeviceSynchronize();
            }

            // Perform Downsweep
            cudaMemset(dev_indices + padded_n - 1, 0, sizeof(int));
            for (int d = numSteps - 1; d >= 0; d--) {
                int num_active_elements = padded_n / (1 << (d + 1));
                int threads_per_block = 128;
                int num_blocks = (num_active_elements + threads_per_block - 1) / threads_per_block;
                kernDownSweep << <num_blocks, threads_per_block >> > (dev_indices, padded_n, d);
                cudaDeviceSynchronize();
            }

            int compacted_size;
            int last_scan_value, last_bool_value;
            cudaMemcpy(&last_scan_value, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_bool_value, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            compacted_size = last_scan_value + last_bool_value;

            // Scatter
            int num_blocks_original = (n + threads_per_block - 1) / threads_per_block;
            StreamCompaction::Common::kernScatter << <num_blocks_original, threads_per_block >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, compacted_size * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            return compacted_size;
        }
    }
}
