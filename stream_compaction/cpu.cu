#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            odata[0] = 0;
            for (int j = 1; j < n;j++) {
                odata[j] = odata[j - 1] + idata[j-1];
            }
            timer().endCpuTimer();
            return;
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            int oIdx = 0;
            int count = 0;
            timer().startCpuTimer();
            for (int i = 0;i < n;i++) {
                if (idata[i] != 0) {
                    odata[oIdx] = idata[i];
                    oIdx++;
                    count++;
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int* odata, const int* idata) {
            int* scanResult = new int[n];
            int* maskedIdata = new int[n];

            timer().startCpuTimer();

            for (int k = 0; k < n;k++) {
                maskedIdata[k] = idata[k] != 0 ? 1 : 0;
            }

            scanResult[0] = 0;
            for (int j = 1; j < n;j++) {
                scanResult[j] = scanResult[j - 1] + maskedIdata[j - 1];
            }
           

            for (int i = 0; i < n; i++) {
                if (maskedIdata[i] == 1) {
                    odata[scanResult[i]] = idata[i];
                }
            }

            timer().endCpuTimer();

            int count = scanResult[n - 1];

            delete[] scanResult;
            delete[] maskedIdata;
            return count;
        }


    }
}
