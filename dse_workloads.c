#include <stdint.h>
#include <string.h>

#define L1_CAP_ELEMENTS (48 * 1024 / sizeof(uintptr_t))
#define L2_CAP_ELEMENTS (1536 * 1024 / sizeof(uintptr_t))
#define L1_ASSOC_STRIDE (32 * 1024 / sizeof(uintptr_t))
#define L2_ASSOC_STRIDE (256 * 1024 / sizeof(uintptr_t))

// Static array in BSS segment
static uintptr_t pool[2 * 1024 * 1024];

int main(int argc, char *argv[]) {
    const char *mode = "all";
    if (argc > 1) {
        mode = argv[1];
    }

    uintptr_t idx = 0;
    volatile uint64_t sum = 0;

    int run_l1 = (strcmp(mode, "all") == 0 || strcmp(mode, "l1") == 0 || strcmp(mode, "1") == 0);
    int run_assoc = (strcmp(mode, "all") == 0 || strcmp(mode, "assoc") == 0 || strcmp(mode, "3") == 0);
    int run_l2 = (strcmp(mode, "all") == 0 || strcmp(mode, "l2") == 0 || strcmp(mode, "2") == 0);

    // Isolated Workload A: L1 Capacity Thrashing (48 KB Ring, 10,000 iterations for isolated, 2,500 for all)
    if (run_l1) {
        int iters = (strcmp(mode, "l1") == 0 || strcmp(mode, "1") == 0) ? 10000 : 2500;
        for (int i = 0; i < L1_CAP_ELEMENTS; i++) {
            pool[i] = (i + 7919) % L1_CAP_ELEMENTS;
        }
        for (int i = 0; i < iters; i++) {
            idx = pool[idx];
        }
    }

    // Isolated Workload C: Associativity Thrashing (32 KB Stride, 1,000 iterations x 16 ways for isolated, 200 for all)
    if (run_assoc) {
        int iters = (strcmp(mode, "assoc") == 0 || strcmp(mode, "3") == 0) ? 1000 : 200;
        sum += idx;
        for (int iter = 0; iter < iters; iter++) {
            for (int i = 0; i < 16; i++) {
                sum += pool[i * L1_ASSOC_STRIDE];
            }
        }
        for (int iter = 0; iter < iters; iter++) {
            for (int i = 0; i < 16; i++) {
                sum += pool[i * L2_ASSOC_STRIDE];
            }
        }
    }

    // Isolated Workload B: L2 Capacity Thrashing (1.5 MB Ring, 10,000 iterations for isolated, 2,500 for all)
    if (run_l2) {
        int iters = (strcmp(mode, "l2") == 0 || strcmp(mode, "2") == 0) ? 10000 : 2500;
        for (int i = 0; i < L2_CAP_ELEMENTS; i++) {
            pool[i] = (i + 104729) % L2_CAP_ELEMENTS;
        }
        idx = 0;
        for (int i = 0; i < iters; i++) {
            idx = pool[idx];
        }
    }

    return (int)((idx ^ sum) & 0xFF);
}
