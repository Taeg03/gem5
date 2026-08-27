import sys

import m5
from m5.objects import *

# Get arguments for the 5 runs
l1_size = sys.argv[1]
l1_assoc = int(sys.argv[2])
l2_size = sys.argv[3]
l2_assoc = int(sys.argv[4])

system = System()
system.clk_domain = SrcClockDomain(
    clock="2GHz", voltage_domain=VoltageDomain()
)
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]

system.cpu = DerivO3CPU()

# Fixed L1I
system.cpu.icache = Cache(
    size="32kB",
    assoc=4,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20,
    clusivity="mostly_incl",
)

# Variable L1D
system.cpu.dcache = Cache(
    size=l1_size,
    assoc=l1_assoc,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=8,
    tgts_per_mshr=20,
    clusivity="mostly_incl",
)

system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port

system.l2bus = L2XBar()
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

# Variable L2 (Forced Exclusive to avoid thrashing)
system.l2cache = Cache(
    size=l2_size,
    assoc=l2_assoc,
    tag_latency=10,
    data_latency=10,
    response_latency=10,
    mshrs=20,
    tgts_per_mshr=12,
    clusivity="mostly_excl",
)

system.l2cache.cpu_side = system.l2bus.mem_side_ports
system.membus = SystemXBar()
system.l2cache.mem_side = system.membus.cpu_side_ports

system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR4_2400_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.workload = SEWorkload.init_compatible("./dse_workload")
process = Process()
process.cmd = ["./dse_workload"]
system.cpu.workload = process
system.cpu.createThreads()
root = Root(full_system=False, system=system)
m5.instantiate()
print(f"Running L1:{l1_size}/{l1_assoc}-way, L2:{l2_size}/{l2_assoc}-way")
exit_event = m5.simulate()
print(f"Exited @ tick {m5.curTick()} because {exit_event.getCause()}")
