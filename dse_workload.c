#include <stdint.h>

#define L1_CAP_ELEMENTS (48 * 1024 / sizeof(uintptr_t))
#define L2_CAP_ELEMENTS (1536 * 1024 / sizeof(uintptr_t))
#define L1_ASSOC_STRIDE (32 * 1024 / sizeof(uintptr_t))
#define L2_ASSOC_STRIDE (256 * 1024 / sizeof(uintptr_t))

// Static array in BSS segment
static uintptr_t pool[2 * 1024 * 1024];

int main() {
    // Phase A: L1 Capacity Thrashing (48 KB Ring, 2,500 iterations)
    // 7919 is prime, establishing a pseudo-random permutation over 48 KB
    uintptr_t idx = 0;
    for (int i = 0; i < L1_CAP_ELEMENTS; i++) {
        pool[i] = (i + 7919) % L1_CAP_ELEMENTS;
    }
    for (int i = 0; i < 2500; i++) {
        idx = pool[idx];
    }

    // Phase B: L1 Associativity Thrashing (32 KB Stride, 200 iterations x 16 ways)
    volatile uint64_t sum = idx;
    for (int iter = 0; iter < 200; iter++) {
        for (int i = 0; i < 16; i++) {
            sum += pool[i * L1_ASSOC_STRIDE];
        }
    }

    // Phase C: L2 Associativity Thrashing (256 KB Stride, 200 iterations x 16 ways)
    for (int iter = 0; iter < 200; iter++) {
        for (int i = 0; i < 16; i++) {
            sum += pool[i * L2_ASSOC_STRIDE];
        }
    }

    // Phase D: L2 Capacity Thrashing (1.5 MB Ring, 2,500 iterations)
    // 104729 is prime, establishing a pseudo-random permutation over 1.5 MB
    for (int i = 0; i < L2_CAP_ELEMENTS; i++) {
        pool[i] = (i + 104729) % L2_CAP_ELEMENTS;
    }
    idx = 0;
    for (int i = 0; i < 2500; i++) {
        idx = pool[idx];
    }

    return (int)((idx ^ sum) & 0xFF);
}
