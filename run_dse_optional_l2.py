import sys
import m5
from m5.objects import *

# CLI Arguments:
# 1: issue_width (2, 4, 8)
# 2: rob_size (32, 64, 128)
# 3: l1d_mshrs (2, 4, 8)
# 4: l1d_size ("16kB", "32kB", "64kB")
# 5: l1d_assoc (2, 4, 8)
# 6: has_l2 ("True" or "False")
# 7: l2_size ("512kB", "1MB", "2MB" if has_l2 else "inactive")
# 8: workload_mode ("compute", "latency", "concurrency")

issue_width = int(sys.argv[1])
rob_size = int(sys.argv[2])
l1d_mshrs = int(sys.argv[3])
l1d_size = sys.argv[4]
l1d_assoc = int(sys.argv[5])
has_l2 = (sys.argv[6].lower() == "true")
l2_size = sys.argv[7]
workload_mode = sys.argv[8] if len(sys.argv) > 8 else "compute"

system = System()
system.clk_domain = SrcClockDomain(clock="2GHz", voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]

# 1. Parameterized DerivO3CPU Core
system.cpu = DerivO3CPU()
system.cpu.fetchWidth = issue_width
system.cpu.decodeWidth = issue_width
system.cpu.renameWidth = issue_width
system.cpu.dispatchWidth = issue_width
system.cpu.issueWidth = issue_width
system.cpu.wbWidth = issue_width
system.cpu.commitWidth = issue_width

system.cpu.numROBEntries = rob_size
system.cpu.LQEntries = max(16, rob_size // 4)
system.cpu.SQEntries = max(16, rob_size // 4)
system.cpu.numPhysIntRegs = max(64, rob_size + 32)
system.cpu.numPhysFloatRegs = max(64, rob_size + 32)

# 2. Fixed L1I Cache
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

# 3. Parameterized L1D Cache
system.cpu.dcache = Cache(
    size=l1d_size,
    assoc=l1d_assoc,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=l1d_mshrs,
    tgts_per_mshr=20,
    clusivity="mostly_incl",
)

system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port

system.membus = SystemXBar()

if has_l2:
    # Standard 2-Level Cache Hierarchy
    system.l2bus = L2XBar()
    system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
    system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

    system.l2cache = Cache(
        size=l2_size,
        assoc=8,
        tag_latency=10,
        data_latency=10,
        response_latency=10,
        mshrs=20,
        tgts_per_mshr=12,
        clusivity="mostly_excl",
    )
    system.l2cache.cpu_side = system.l2bus.mem_side_ports
    system.l2cache.mem_side = system.membus.cpu_side_ports
else:
    # Direct L1 to SystemXBar (No L2 Cache)
    system.cpu.icache.mem_side = system.membus.cpu_side_ports
    system.cpu.dcache.mem_side = system.membus.cpu_side_ports

system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR4_2400_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.workload = SEWorkload.init_compatible("./workload_729_bin")
process = Process()
process.cmd = ["./workload_729_bin", workload_mode]
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()
exit_event = m5.simulate()
