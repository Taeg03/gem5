#!/bin/bash

# Define our 5 test configurations
# Format: L1_size L1_assoc L2_size L2_assoc
declare -a configs=(
    "16kB 2 512kB 2"      # The Bottleneck
    "128kB 16 512kB 2"    # L1 Heavy
    "16kB 2 4MB 16"       # L2 Heavy
    "64kB 8 2MB 8"        # The Balanced
    "128kB 16 4MB 16"     # The Maximizer
)

echo "Starting 5-point sanity check..."

for conf in "${configs[@]}"; do
    # Split the configuration string into variables
    read -r l1s l1a l2s l2a <<< "$conf"
    
    echo -e "\n========================================"
    echo "Evaluating -> L1: $l1s $l1a-way | L2: $l2s $l2a-way"
    echo "========================================"
    
    # Run gem5 (sending standard output to /dev/null to keep the console clean)
    build/X86/gem5.opt run_dse.py $l1s $l1a $l2s $l2a > /dev/null
    
    # Extract our specific metrics from the stats file
    grep "system.cpu.ipc" m5out/stats.txt
    grep "system.cpu.dcache.overallMisses::total" m5out/stats.txt
    grep "system.l2cache.overallMisses::total" m5out/stats.txt
    grep "system.mem_ctrl.dram.numReads::total" m5out/stats.txt
    grep "simSeconds" m5out/stats.txt
done

echo -e "\nSanity check complete!"
