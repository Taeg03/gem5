#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define L1_CAP_SIZE (48 * 1024 / sizeof(void*))
#define L2_CAP_SIZE (1536 * 1024 / sizeof(void*))
#define L1_ASSOC_STRIDE (32 * 1024 / sizeof(void*))
#define L2_ASSOC_STRIDE (256 * 1024 / sizeof(void*))

void init_random_list(void** array, int num_elements) {
    for (int i = 0; i < num_elements; i++) array[i] = (void*)&array[i];
    // Fisher-Yates shuffle to create random pointer chain
    for (int i = num_elements - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        void* temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

int main() {
    printf("Starting DSE Workload...\n");
    void** memory = (void**)malloc(16 * 1024 * 1024); // 16MB pool

    // Phase A: L1 Capacity
    init_random_list(memory, L1_CAP_SIZE);
    void** p = memory;
    for(int i=0; i<500000; i++) p = (void**)*p;

    // Phase B: L1 Associativity
    volatile uint64_t sum = 0;
    for(int iter=0; iter<50000; iter++) {
        for(int i=0; i<16; i++) sum += (uint64_t)memory[i * L1_ASSOC_STRIDE];
    }

    // Phase C: L2 Associativity
    for(int iter=0; iter<50000; iter++) {
        for(int i=0; i<16; i++) sum += (uint64_t)memory[i * L2_ASSOC_STRIDE];
    }

    // Phase D: L2 Capacity
    init_random_list(memory, L2_CAP_SIZE);
    p = memory;
    for(int i=0; i<500000; i++) p = (void**)*p;

    printf("Done. Sum: %lu\n", sum);
    return 0;
}
