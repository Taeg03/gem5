#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define POOL_SIZE_BYTES (2 * 1024 * 1024)
#define NUM_ELEMENTS (POOL_SIZE_BYTES / sizeof(uintptr_t))

static uintptr_t pool[NUM_ELEMENTS];

int main(int argc, char *argv[]) {
    const char *mode = "compute";
    if (argc > 1) {
        mode = argv[1];
    }

    uintptr_t idx = 0;
    volatile uint64_t sum = 0;

    // -------------------------------------------------------------
    // Workload 1: Compute / ILP-Oriented (Tight math with unrolled ILP)
    // -------------------------------------------------------------
    if (strcmp(mode, "compute") == 0 || strcmp(mode, "1") == 0) {
        volatile uint64_t a = 1, b = 2, c = 3, d = 4;
        volatile uint64_t e = 5, f = 6, g = 7, h = 8;
        for (int iter = 0; iter < 30000; iter++) {
            a = (a * 1103515245 + 12345) ^ (b >> 3);
            b = (b * 1103515245 + 12345) ^ (c >> 3);
            c = (c * 1103515245 + 12345) ^ (d >> 3);
            d = (d * 1103515245 + 12345) ^ (e >> 3);
            e = (e * 1103515245 + 12345) ^ (f >> 3);
            f = (f * 1103515245 + 12345) ^ (g >> 3);
            g = (g * 1103515245 + 12345) ^ (h >> 3);
            h = (h * 1103515245 + 12345) ^ (a >> 3);
        }
        sum = a + b + c + d + e + f + g + h;
    }

    // -------------------------------------------------------------
    // Workload 2: Memory-Latency-Oriented (Serial pointer chase, 1.5 MB)
    // -------------------------------------------------------------
    else if (strcmp(mode, "latency") == 0 || strcmp(mode, "2") == 0) {
        int cap_elements = (1536 * 1024) / sizeof(uintptr_t);
        for (int i = 0; i < cap_elements; i++) {
            pool[i] = (i + 104729) % cap_elements;
        }
        idx = 0;
        for (int iter = 0; iter < 8000; iter++) {
            idx = pool[idx];
        }
        sum = idx;
    }

    // -------------------------------------------------------------
    // Workload 3: Memory-Concurrency / MLP-Oriented (8 Parallel streams)
    // -------------------------------------------------------------
    else if (strcmp(mode, "concurrency") == 0 || strcmp(mode, "3") == 0) {
        int stream_size = (192 * 1024) / sizeof(uintptr_t); // 8 streams x 192 KB = 1.5 MB
        for (int s = 0; s < 8; s++) {
            int offset = s * stream_size;
            for (int i = 0; i < stream_size; i++) {
                pool[offset + i] = offset + ((i + 7919) % stream_size);
            }
        }

        uintptr_t s0 = 0 * stream_size;
        uintptr_t s1 = 1 * stream_size;
        uintptr_t s2 = 2 * stream_size;
        uintptr_t s3 = 3 * stream_size;
        uintptr_t s4 = 4 * stream_size;
        uintptr_t s5 = 5 * stream_size;
        uintptr_t s6 = 6 * stream_size;
        uintptr_t s7 = 7 * stream_size;

        for (int iter = 0; iter < 1200; iter++) {
            s0 = pool[s0];
            s1 = pool[s1];
            s2 = pool[s2];
            s3 = pool[s3];
            s4 = pool[s4];
            s5 = pool[s5];
            s6 = pool[s6];
            s7 = pool[s7];
        }
        sum = s0 ^ s1 ^ s2 ^ s3 ^ s4 ^ s5 ^ s6 ^ s7;
    }

    return (int)(sum & 0xFF);
}
