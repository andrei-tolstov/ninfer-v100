#include "core/arena.h"
#include "core/numa.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#    include <linux/mempolicy.h>
#    include <sched.h>
#    include <sys/mman.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

namespace {

using namespace ninfer;

void require(bool ok, const char* msg) {
    if (!ok) { throw std::runtime_error(msg); }
}

struct TempSysfs {
    std::filesystem::path root;

    TempSysfs() {
        const auto base = std::filesystem::temp_directory_path();
        for (int attempt = 0; attempt < 64; ++attempt) {
            auto candidate = base / ("ninfer_test_sysfs_" + std::to_string(attempt) + "_" +
                                     std::to_string(std::rand()));
            std::error_code ec;
            if (std::filesystem::create_directories(candidate, ec)) {
                root = std::move(candidate);
                return;
            }
        }
        throw std::runtime_error("failed to create temporary sysfs directory");
    }

    ~TempSysfs() {
        if (!root.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(root, ec);
        }
    }

    void write(const std::string& rel, const std::string& text) const {
        const std::filesystem::path p = root / rel;
        std::filesystem::create_directories(p.parent_path());
        std::ofstream f(p);
        require(bool(f), "failed to write synthetic sysfs file");
        f << text;
    }
};

void build_supermicro_2p_e5_2696v2_sysfs(TempSysfs& sys,
                                         std::uint64_t node0_avail_kib,
                                         std::uint64_t node1_avail_kib) {
    // Supermicro X9DAX-7F-HFT: 2 x Xeon E5-2696 v2 (12c/24t per socket = 24c/48t, 128 GiB/node)
    constexpr std::uint64_t kTotalKib = 128ULL * 1024ULL * 1024ULL;
    sys.write("devices/system/node/online", "0-1\n");
    sys.write("devices/system/node/node0/cpulist", "0-11,24-35\n");
    sys.write("devices/system/node/node1/cpulist", "12-23,36-47\n");

    const std::uint64_t n0_free     = node0_avail_kib / 4;
    const std::uint64_t n0_act_file = node0_avail_kib / 2;
    const std::uint64_t n0_ina_file = node0_avail_kib - n0_free - n0_act_file;
    sys.write("devices/system/node/node0/meminfo",
              "Node 0 MemTotal:       " + std::to_string(kTotalKib) + " kB\n"
              "Node 0 MemFree:        " + std::to_string(n0_free) + " kB\n"
              "Node 0 Active(file):   " + std::to_string(n0_act_file) + " kB\n"
              "Node 0 Inactive(file): " + std::to_string(n0_ina_file) + " kB\n"
              "Node 0 SReclaimable:   0 kB\n");

    const std::uint64_t n1_free  = node1_avail_kib / 3;
    const std::uint64_t n1_files = node1_avail_kib - n1_free;
    sys.write("devices/system/node/node1/meminfo",
              "Node 1 MemTotal:       " + std::to_string(kTotalKib) + " kB\n"
              "Node 1 MemFree:        " + std::to_string(n1_free) + " kB\n"
              "Node 1 FilePages:      " + std::to_string(n1_files) + " kB\n");

    for (int cpu = 0; cpu < 48; ++cpu) {
        const std::string base = "devices/system/cpu/cpu" + std::to_string(cpu);
        sys.write(base + "/online", "1\n");
        const int pkg  = (cpu < 12 || (cpu >= 24 && cpu < 36)) ? 0 : 1;
        const int core = cpu % 12;
        sys.write(base + "/topology/physical_package_id", std::to_string(pkg) + "\n");
        sys.write(base + "/topology/core_id", std::to_string(core) + "\n");
    }
}

void test_synthetic_supermicro_sysfs() {
    TempSysfs sys;
    constexpr std::uint64_t k72GiBKib = 72ULL * 1024ULL * 1024ULL;
    build_supermicro_2p_e5_2696v2_sysfs(sys, k72GiBKib, k72GiBKib);

    const NumaTopology topo = parse_numa_sysfs(sys.root);
    require(topo.multi_node(), "2P E5-2696 v2 sysfs should report multi_node=true");
    require(topo.node_count() == 2, "expected 2 NUMA nodes");
    require(topo.nodes[0].physical_cpus.size() == 12, "node 0 should deduplicate 12 phys cores");
    require(topo.nodes[0].logical_cpus.size() == 24, "node 0 should have 24 logical CPUs");
    require(topo.nodes[1].physical_cpus.size() == 12, "node 1 should deduplicate 12 phys cores");
    require(topo.nodes[1].logical_cpus.size() == 24, "node 1 should have 24 logical CPUs");
    require(topo.nodes[0].free_bytes == k72GiBKib * 1024ULL,
            "node 0 free_bytes must include reclaimable file cache");
    require(topo.nodes[1].free_bytes == k72GiBKib * 1024ULL,
            "node 1 free_bytes must parse FilePages");
    require(topo.node_for_cpu(25) == 0 && topo.node_for_cpu(37) == 1,
            "node_for_cpu mismatch for SMT siblings");

    // Quota cap test: cgroup cap of 8 CPUs across 2 nodes keeps 4 physical cores per node.
    const NumaTopology capped = parse_numa_sysfs(sys.root, {}, 8);
    require(capped.nodes[0].logical_cpus.size() == 4 && capped.nodes[0].physical_cpus.size() == 4,
            "quota cap must prefer distinct physical cores on node 0");
    require(capped.nodes[1].logical_cpus.size() == 4 && capped.nodes[1].physical_cpus.size() == 4,
            "quota cap must prefer distinct physical cores on node 1");

    // PCIe root-complex lookup (case-insensitive hex).
    sys.write("bus/pci/devices/0000:0a:00.0/numa_node", "0\n");
    sys.write("bus/pci/devices/0000:83:00.0/numa_node", "1\n");
    sys.write("bus/pci/devices/0000:ff:00.0/numa_node", "-1\n");
    require(parse_pci_device_numa_node(sys.root, "0000:0A:00.0") == 0,
            "case-insensitive PCIe node lookup failed for 0A:00.0");
    require(parse_pci_device_numa_node(sys.root, "0000:83:00.0") == 1,
            "PCIe node lookup failed for 83:00.0");
    require(parse_pci_device_numa_node(sys.root, "0000:FF:00.0") == kNoNumaNode,
            "-1 numa_node should map to kNoNumaNode");
}

void test_plan_host_numa_policies() {
    TempSysfs sys;
    constexpr std::uint64_t k72GiBKib = 72ULL * 1024ULL * 1024ULL;
    build_supermicro_2p_e5_2696v2_sysfs(sys, k72GiBKib, k72GiBKib);
    const NumaTopology topo = parse_numa_sysfs(sys.root);
    constexpr std::uint64_t kGiB = 1ULL << 30;

    // 1. Auto: weights (50 GiB) fit on GPU socket (72 GiB avail) & default threads -> Isolate.
    {
        const auto plan = plan_host_numa(topo, 0, 50ULL * kGiB, 0, NumaPolicy::Auto);
        require(plan.mode == HostNumaMode::Isolate, "Auto should Isolate when RAM+threads fit");
        require(plan.primary_node == 0, "primary_node should follow gpu_node");
        require(plan.node_plans.size() == 1, "Isolate uses single node_plan");
        require(plan.node_plans[0].worker_threads == 10, "12c socket auto threads should be 10");
        require(plan.node_plans[0].worker_cpus.size() == 9, "bg worker_cpus should be 9");
        require(plan.service_cpu >= 0 && topo.node_for_cpu(plan.service_cpu) == 0,
                "service_cpu must reside on primary node");
        require(plan.pinned_shards.size() == 1 && plan.pinned_shards[0].numa_node == 0,
                "Isolate should bind entire pinned region to node 0");
    }

    // 2. Auto: weights (80 GiB) exceed node 0 free (72 GiB) & total 144 GiB < 2.2*80 -> Sharded.
    {
        const auto plan = plan_host_numa(topo, 0, 80ULL * kGiB, 0, NumaPolicy::Auto);
        require(plan.mode == HostNumaMode::Sharded,
                "Auto should Shard when 1 socket RAM exceeded and total < 2.2x");
        require(plan.node_plans.size() == 2, "Sharded spans both nodes");
        require(plan.node_plans[0].node_id == 0 && plan.node_plans[0].worker_threads == 10,
                "Sharded node 0 gets 10 threads");
        require(plan.node_plans[1].node_id == 1 && plan.node_plans[1].worker_threads == 11,
                "Sharded node 1 gets 11 threads (12 - 1)");
        require(plan.pinned_shards.size() == 2, "Sharded creates 2 pinned shards");
        require(plan.pinned_shards[0].numa_node == 0 && plan.pinned_shards[1].numa_node == 1,
                "shards should target node 0 and node 1");
        require(plan.pinned_shards[0].bytes + plan.pinned_shards[1].bytes == 80ULL * kGiB,
                "shards must sum to full pinned size");
    }

    // 3. Auto: weights (30 GiB), requested_threads=16 (> 12 local phys), free 144 >= 2.2*30 -> Mirror.
    {
        const auto plan = plan_host_numa(topo, 1, 30ULL * kGiB, 16, NumaPolicy::Auto);
        require(plan.mode == HostNumaMode::Mirror,
                "Auto should Mirror when threads > local phys and free >= 2.2x");
        require(plan.primary_node == 1, "GPU on socket 1 must make node 1 primary");
        require(plan.node_plans.size() == 2, "Mirror uses both nodes");
        require(plan.node_plans[0].node_id == 1 && plan.node_plans[1].node_id == 0,
                "Mirror node_plans[0] must be primary GPU node 1");
        require(plan.node_plans[0].worker_threads + plan.node_plans[1].worker_threads == 16,
                "worker_threads total must match requested 16");
        require(plan.pinned_shards.size() == 1 && plan.pinned_shards[0].numa_node == 1,
                "Mirror primary pinned allocation stays on GPU node 1");
    }

    // 4. Auto: requested_threads=16, weights=60 GiB, total free 120 < 2.2*60=132 -> Sharded.
    {
        TempSysfs sys2;
        build_supermicro_2p_e5_2696v2_sysfs(sys2, 60ULL * 1024ULL * 1024ULL,
                                            60ULL * 1024ULL * 1024ULL);
        const NumaTopology topo2 = parse_numa_sysfs(sys2.root);
        const auto plan = plan_host_numa(topo2, 0, 50ULL * kGiB, 16, NumaPolicy::Auto);
        // free = 120 GiB >= 2.2*50 (110 GiB), so bump to 58 GiB -> 2.2*58 = 127.6 GiB > 120 GiB.
        const auto plan_shard = plan_host_numa(topo2, 0, 58ULL * kGiB, 16, NumaPolicy::Auto);
        require(plan.mode == HostNumaMode::Mirror, "50 GiB fits Mirror under 120 GiB");
        require(plan_shard.mode == HostNumaMode::Sharded,
                "58 GiB should Shard when free 120 GiB < 2.2*58 GiB");
    }

    // 5. Explicit Isolate with requested_threads=20 (> 12 phys, <= 24 log): distinct SMT CPUs.
    {
        const auto plan = plan_host_numa(topo, 0, 20ULL * kGiB, 20, NumaPolicy::Isolate);
        require(plan.mode == HostNumaMode::Isolate && plan.node_plans.size() == 1,
                "explicit Isolate ignores multi-node expansion");
        require(plan.node_plans[0].worker_threads == 20, "explicit thread count preserved");
        std::set<int> used;
        used.insert(plan.service_cpu);
        for (const int c : plan.node_plans[0].worker_cpus) {
            require(topo.node_for_cpu(c) == 0, "Isolate workers must stay on node 0");
            used.insert(c);
        }
        require(used.size() == 20, "20 Isolate threads should use 20 distinct logical CPUs");
    }

    // 6. Explicit Sharded even for small payload.
    {
        const auto plan = plan_host_numa(topo, 0, 8ULL * kGiB, 0, NumaPolicy::Sharded);
        require(plan.mode == HostNumaMode::Sharded && plan.node_plans.size() == 2 &&
                    plan.pinned_shards.size() == 2,
                "explicit Sharded forces 2-node sharding");
    }
}

// Exercises the review-driven fixes: Shmem subtraction, free-weighted Sharded split, explicit
// mirror/isolate RAM refusal, the CPU_SETSIZE guard, and policy save/restore.
void test_numa_review_fixes() {
#ifdef __linux__
    constexpr std::uint64_t kGiB = 1ULL << 30;
    constexpr std::uint64_t kKibPerGiB = 1024ULL * 1024ULL;

    // Shmem is subtracted from a node's reclaimable memory: a node reporting 10 GiB of shmem sees
    // 10 GiB less free than an otherwise identical one.
    {
        auto write_node = [](TempSysfs& s, int node, std::uint64_t avail_kib,
                             bool with_shmem) {
            const std::uint64_t free_kib  = avail_kib / 2;
            const std::uint64_t act_file  = avail_kib - free_kib;
            const std::uint64_t shmem_kib = with_shmem ? 10ULL * kKibPerGiB : 0ULL;
            s.write("devices/system/node/node" + std::to_string(node) + "/meminfo",
                    "Node " + std::to_string(node) + " MemTotal:       " +
                            std::to_string(128ULL * kKibPerGiB) + " kB\n"
                            "Node " + std::to_string(node) + " MemFree:        " +
                            std::to_string(free_kib) + " kB\n"
                            "Node " + std::to_string(node) + " Active(file):   " +
                            std::to_string(act_file) + " kB\n"
                            "Node " + std::to_string(node) + " Inactive(file): 0 kB\n"
                            "Node " + std::to_string(node) + " Shmem:          " +
                            std::to_string(shmem_kib) + " kB\n");
        };
        TempSysfs s0;
        s0.write("devices/system/node/online", "0-1\n");
        write_node(s0, 0, 80ULL * kKibPerGiB, /*with_shmem=*/false);
        write_node(s0, 1, 80ULL * kKibPerGiB, /*with_shmem=*/true);
        for (int cpu = 0; cpu < 8; ++cpu) {
            const std::string base = "devices/system/cpu/cpu" + std::to_string(cpu);
            s0.write(base + "/online", "1\n");
            s0.write(base + "/topology/physical_package_id",
                     std::string(cpu < 4 ? "0" : "1") + "\n");
            s0.write(base + "/topology/core_id", std::to_string(cpu % 4) + "\n");
        }
        for (int node = 0; node < 2; ++node) {
            s0.write("devices/system/node/node" + std::to_string(node) + "/cpulist",
                     node == 0 ? "0-3\n" : "4-7\n");
        }
        const NumaTopology t0 = parse_numa_sysfs(s0.root);
        const NumaNodeInfo* n0 = t0.find(0);
        const NumaNodeInfo* n1 = t0.find(1);
        require(n0 && n1, "2-node topology with shmem must parse");
        require(n1->free_bytes < n0->free_bytes,
                "node with 10 GiB shmem must report less free memory");
        const std::uint64_t delta = n0->free_bytes - n1->free_bytes;
        require(delta >= 9ULL * kGiB && delta <= 11ULL * kGiB,
                "shmem subtraction should remove ~10 GiB from free");
    }

    // Free-weighted Sharded: with node0 free 90 GiB and node1 free 54 GiB, the pinned range splits
    // ~5:3, not evenly 40:40.
    {
        TempSysfs s;
        s.write("devices/system/node/online", "0-1\n");
        const auto meminfo = [&](int node, std::uint64_t avail_kib) {
            s.write("devices/system/node/node" + std::to_string(node) + "/meminfo",
                    "Node " + std::to_string(node) + " MemTotal:       " +
                            std::to_string(128ULL * kKibPerGiB) + " kB\n"
                            "Node " + std::to_string(node) + " MemFree:        " +
                            std::to_string(avail_kib / 2) + " kB\n"
                            "Node " + std::to_string(node) + " Active(file):   " +
                            std::to_string(avail_kib - avail_kib / 2) + " kB\n"
                            "Node " + std::to_string(node) + " Inactive(file): 0 kB\n"
                            "Node " + std::to_string(node) + " SReclaimable:   0 kB\n");
        };
        meminfo(0, 90ULL * kKibPerGiB);
        meminfo(1, 54ULL * kKibPerGiB);
        for (int cpu = 0; cpu < 8; ++cpu) {
            const std::string base = "devices/system/cpu/cpu" + std::to_string(cpu);
            s.write(base + "/online", "1\n");
            s.write(base + "/topology/physical_package_id",
                    std::string(cpu < 4 ? "0" : "1") + "\n");
            s.write(base + "/topology/core_id", std::to_string(cpu % 4) + "\n");
        }
        s.write("devices/system/node/node0/cpulist", "0-3\n");
        s.write("devices/system/node/node1/cpulist", "4-7\n");
        const NumaTopology t = parse_numa_sysfs(s.root);
        const auto plan      = plan_host_numa(t, 0, 80ULL * kGiB, 0, NumaPolicy::Sharded);
        require(plan.pinned_shards.size() == 2, "free-weighted Sharded keeps 2 shards");
        require(plan.pinned_shards[0].bytes > plan.pinned_shards[1].bytes,
                "node with more free memory must carry the larger shard");
        const double ratio =
            static_cast<double>(plan.pinned_shards[0].bytes) /
            static_cast<double>(plan.pinned_shards[1].bytes);
        require(ratio > 1.5 && ratio < 3.5,
                "shard ratio should track the 90:54 free ratio, not 1:1");
        require(plan.pinned_shards[0].bytes + plan.pinned_shards[1].bytes == 80ULL * kGiB,
                "weighted shards must still sum to the pinned size");
    }

    // Explicit Mirror on a host that provably cannot hold the weights refuses up front.
    {
        TempSysfs s;
        s.write("devices/system/node/online", "0-1\n");
        const auto meminfo = [&](int node) {
            s.write("devices/system/node/node" + std::to_string(node) + "/meminfo",
                    "Node " + std::to_string(node) + " MemTotal:       " +
                            std::to_string(8ULL * kKibPerGiB) + " kB\n"
                    "Node " + std::to_string(node) + " MemFree:        " +
                            std::to_string(4ULL * kKibPerGiB) + " kB\n"
                    "Node " + std::to_string(node) + " SReclaimable:   0 kB\n");
        };
        meminfo(0);
        meminfo(1);
        for (int cpu = 0; cpu < 8; ++cpu) {
            const std::string base = "devices/system/cpu/cpu" + std::to_string(cpu);
            s.write(base + "/online", "1\n");
            s.write(base + "/topology/physical_package_id",
                    std::string(cpu < 4 ? "0" : "1") + "\n");
            s.write(base + "/topology/core_id", std::to_string(cpu % 4) + "\n");
        }
        s.write("devices/system/node/node0/cpulist", "0-3\n");
        s.write("devices/system/node/node1/cpulist", "4-7\n");
        const NumaTopology t = parse_numa_sysfs(s.root);
        bool refused         = false;
        try {
            (void)plan_host_numa(t, 0, 20ULL * kGiB, 0, NumaPolicy::Mirror);
        } catch (const std::runtime_error&) {
            refused = true;
        }
        require(refused, "explicit Mirror with insufficient RAM must throw");
    }

    // bind_thread_to_cpu rejects ids past the static cpu_set_t (CPU_SETSIZE) bound.
    require(!bind_thread_to_cpu(1 << 20), "bind_thread_to_cpu must reject cpu_id >= CPU_SETSIZE");
    require(!bind_thread_to_cpu(-1), "bind_thread_to_cpu must reject negative cpu_id");

    // ScopedThreadMemPolicy restores a non-default pre-existing policy (mode AND word-by-word
    // nodemask), and a rejected/failed scope preserves the caller's pre-existing policy.
    {
        auto read_policy = [](int& mode, std::array<unsigned long, 16>& mask) -> bool {
            mask.fill(0UL);
            return syscall(__NR_get_mempolicy, &mode, mask.data(), 1024UL + 1UL, nullptr, 0UL) == 0;
        };
        std::array<unsigned long, 16> want_mask{};
        want_mask[0] = 1UL;
        if (syscall(__NR_set_mempolicy, MPOL_INTERLEAVE, want_mask.data(), 1024UL + 1UL) == 0) {
            int before_mode = -1;
            std::array<unsigned long, 16> before_mask{};
            require(read_policy(before_mode, before_mask) && before_mode == MPOL_INTERLEAVE,
                    "failed to establish non-default MPOL_INTERLEAVE policy");
            { ScopedThreadMemPolicy scope(0, NumaPlacement::Bind); }
            int after_mode = -1;
            std::array<unsigned long, 16> after_mask{};
            require(read_policy(after_mode, after_mask), "get_mempolicy failed after scope");
            require(after_mode == before_mode && after_mask == before_mask,
                    "ScopedThreadMemPolicy failed to restore non-default mode and nodemask");

            // Rejected node id (2000 >= 1024) must not clobber the active policy.
            {
                const int bad_nodes[1] = {2000};
                ScopedThreadMemPolicy rejected(bad_nodes, /*interleave=*/false);
                require(!rejected.active(), "rejected node id 2000 must not activate policy");
            }
            require(read_policy(after_mode, after_mask) && after_mode == before_mode &&
                        after_mask == before_mask,
                    "rejected ScopedThreadMemPolicy clobbered pre-existing policy");
            (void)syscall(__NR_set_mempolicy, MPOL_DEFAULT, nullptr, 0UL);
        }
    }

    // Default CPU worker count must stay capped per socket (<= 16) and globally (<= 32) on
    // high-core-count 1P and 2P hosts rather than spawning unbounded spinning threads.
    {
        NumaTopology big1p;
        NumaNodeInfo n;
        n.node_id     = 0;
        n.total_bytes = 256ULL * kGiB;
        n.free_bytes  = 200ULL * kGiB;
        for (int c = 0; c < 32; ++c) {
            n.logical_cpus.push_back(c);
            n.physical_cpus.push_back(c);
        }
        big1p.nodes.push_back(n);
        const auto p1 = plan_host_numa(big1p, 0, 10ULL * kGiB, 0, NumaPolicy::Auto);
        require(p1.node_plans.size() == 1 && p1.node_plans[0].worker_threads <= 16,
                "1P 32-core default worker_threads must be capped at <= 16");

        NumaTopology big2p;
        for (int node = 0; node < 2; ++node) {
            NumaNodeInfo bn;
            bn.node_id     = node;
            bn.total_bytes = 256ULL * kGiB;
            bn.free_bytes  = 200ULL * kGiB;
            for (int c = 0; c < 24; ++c) {
                const int cpu = node * 24 + c;
                bn.logical_cpus.push_back(cpu);
                bn.physical_cpus.push_back(cpu);
            }
            big2p.nodes.push_back(std::move(bn));
        }
        const auto p2 = plan_host_numa(big2p, 0, 220ULL * kGiB, 0, NumaPolicy::Auto);
        std::uint32_t total_threads = 0;
        for (const auto& np : p2.node_plans) {
            require(np.worker_threads <= 16,
                    "per-socket default worker_threads must be capped at <= 16");
            total_threads += np.worker_threads;
        }
        require(total_threads <= 32, "total default worker_threads across sockets must be <= 32");
    }
#else
    (void)require;
#endif
}

void test_numa_pr1_regressions() {
    constexpr std::uint64_t kGiB       = 1ULL << 30;
    constexpr std::uint64_t kKibPerGiB = 1024ULL * 1024ULL;

    // T-13b: ShmemHugePages, ShmemPmdMapped, FileHugePages, and HugePages_Total/Free must not
    // alias Shmem or MemFree/FilePages.
    {
        TempSysfs s;
        s.write("devices/system/node/online", "0\n");
        s.write("devices/system/node/node0/cpulist", "0-3\n");
        for (int cpu = 0; cpu < 4; ++cpu) {
            const std::string base = "devices/system/cpu/cpu" + std::to_string(cpu);
            s.write(base + "/online", "1\n");
            s.write(base + "/topology/physical_package_id", "0\n");
            s.write(base + "/topology/core_id", std::to_string(cpu) + "\n");
        }
        s.write("devices/system/node/node0/meminfo",
                "Node 0 MemTotal:       " + std::to_string(128ULL * kKibPerGiB) + " kB\n"
                "Node 0 MemFree:        " + std::to_string(40ULL * kKibPerGiB) + " kB\n"
                "Node 0 Active(file):   " + std::to_string(30ULL * kKibPerGiB) + " kB\n"
                "Node 0 Inactive(file): " + std::to_string(10ULL * kKibPerGiB) + " kB\n"
                "Node 0 FileHugePages:  " + std::to_string(20ULL * kKibPerGiB) + " kB\n"
                "Node 0 Shmem:          " + std::to_string(4ULL * kKibPerGiB) + " kB\n"
                "Node 0 ShmemHugePages: " + std::to_string(16ULL * kKibPerGiB) + " kB\n"
                "Node 0 ShmemPmdMapped: " + std::to_string(16ULL * kKibPerGiB) + " kB\n"
                "Node 0 HugePages_Total: 16384\n"
                "Node 0 HugePages_Free:  16384\n");
        const NumaTopology t = parse_numa_sysfs(s.root);
        require(t.node_count() == 1, "T-13b must parse single node");
        // free_bytes = 40 + (30 + 10 - 4) = 76 GiB; none of the hugepage fields leak into free.
        require(t.nodes[0].free_bytes == 76ULL * kGiB,
                "T-13b parse_node_meminfo leaked hugepage/Shmem* fields into free_bytes");
    }

    auto make_2p = [&](std::uint64_t f0_gib, std::uint64_t f1_gib,
                       std::uint64_t t_gib = 128) -> NumaTopology {
        NumaTopology topo;
        for (int node = 0; node < 2; ++node) {
            NumaNodeInfo n;
            n.node_id     = node;
            n.total_bytes = t_gib * kGiB;
            n.free_bytes  = (node == 0 ? f0_gib : f1_gib) * kGiB;
            for (int c = 0; c < 8; ++c) {
                const int cpu = node * 8 + c;
                n.logical_cpus.push_back(cpu);
                n.physical_cpus.push_back(cpu);
            }
            topo.nodes.push_back(std::move(n));
        }
        return topo;
    };

    // T-15: Auto Sharded shard sizes must weight by usable free memory (free - headroom) and refuse
    // when total usable RAM across all nodes cannot fit pinned_bytes.
    {
        const NumaTopology asym = make_2p(4, 200, 256);
        // node0: free 4 GiB, headroom 2 GiB -> usable 2 GiB. Shard 0 must not exceed 2 GiB.
        const auto plan = plan_host_numa(asym, 0, 120ULL * kGiB, 0, NumaPolicy::Auto);
        require(plan.mode == HostNumaMode::Sharded && plan.pinned_shards.size() == 2,
                "T-15 Auto must shard asymmetric 2P topology");
        require(plan.pinned_shards[0].bytes <= 2ULL * kGiB,
                "T-15 node 0 shard exceeded free_0 - headroom_0");
        require(plan.pinned_shards[0].bytes + plan.pinned_shards[1].bytes == 120ULL * kGiB,
                "T-15 shards must sum to pinned_bytes");

        const NumaTopology tight = make_2p(4, 100, 128);
        // total usable = (4 - 2) + (100 - 2) = 100 GiB < 101 GiB -> must throw.
        bool refused = false;
        try {
            (void)plan_host_numa(tight, 0, 101ULL * kGiB, 0, NumaPolicy::Auto);
        } catch (const std::runtime_error&) { refused = true; }
        require(refused, "T-15 Auto Sharded must refuse when sum(free - headroom) < pinned_bytes");
    }

    // T-16: Explicit Isolate/Mirror/Sharded headroom gate must not be softer than Auto.
    {
        const NumaTopology t = make_2p(10, 10, 128); // headroom = 2 GiB/node -> usable 8 GiB/node
        auto throws_on = [&](std::uint64_t pinned, NumaPolicy pol) {
            try {
                (void)plan_host_numa(t, 0, pinned, 0, pol);
                return false;
            } catch (const std::runtime_error&) { return true; }
        };
        require(throws_on(9ULL * kGiB, NumaPolicy::Isolate),
                "T-16 explicit Isolate accepted pinned > free - headroom");
        require(throws_on(9ULL * kGiB, NumaPolicy::Mirror),
                "T-16 explicit Mirror accepted pinned > free - headroom");
        require(throws_on(17ULL * kGiB, NumaPolicy::Sharded),
                "T-16 explicit Sharded accepted pinned > sum(free - headroom)");
    }

    // T-17: 64 KiB page size and unaligned tail size must keep offsets page-aligned without
    // truncating trailing bytes. Also invalid non-power-of-two page_size must be rejected.
    {
        const NumaTopology t       = make_2p(64, 64, 128);
        constexpr std::uint64_t ps = 65536ULL;
        const std::uint64_t pinned = 40ULL * kGiB + 4096ULL;
        const auto plan = plan_host_numa(t, 0, pinned, 0, NumaPolicy::Sharded, ps);
        std::uint64_t sum = 0;
        for (const auto& s : plan.pinned_shards) {
            require(s.offset % ps == 0, "T-17 shard offset must align to page_size");
            sum += s.bytes;
        }
        require(sum == pinned, "T-17 shard bytes must equal pinned_bytes (tail truncated)");

        bool bad_ps_rejected = false;
        try {
            (void)plan_host_numa(t, 0, 8ULL * kGiB, 0, NumaPolicy::Sharded, 12345ULL);
        } catch (const std::invalid_argument&) { bad_ps_rejected = true; }
        require(bad_ps_rejected, "T-17 non-power-of-two page_size must be rejected");
    }

    // T-19: pinned=0 (disk residency) under Auto on 2P must span all sockets rather than Isolate.
    {
        const NumaTopology t = make_2p(64, 64, 128);
        const auto plan      = plan_host_numa(t, 0, 0, 0, NumaPolicy::Auto);
        require(plan.mode == HostNumaMode::Sharded && plan.node_plans.size() == 2 &&
                    plan.pinned_shards.empty(),
                "T-19 Auto with pinned=0 on 2P must keep both socket pools active");
    }

    // T-14b: Empty topology and node outside process CPU affinity (no logical_cpus).
    {
        const NumaTopology empty;
        const auto empty_plan = plan_host_numa(empty, 0, 0, 0, NumaPolicy::Auto);
        require(empty_plan.node_plans.empty(), "T-14b empty topology must return empty node_plans");

        NumaTopology no_aff = make_2p(64, 64, 128);
        no_aff.nodes[1].logical_cpus.clear();
        no_aff.nodes[1].physical_cpus.clear();
        const auto plan = plan_host_numa(no_aff, 0, 80ULL * kGiB, 0, NumaPolicy::Sharded);
        require(plan.node_plans.size() == 2 && plan.node_plans[1].worker_threads == 0,
                "T-14b node without affinity CPUs must get 0 worker_threads, not 1");
    }

#ifdef __linux__
    // T-17b: Multi-word nodemask (>64 nodes) sets exact bit positions and rejects node >= 1024.
    {
        std::vector<unsigned long> mask;
        unsigned long maxnode   = 0;
        const int high_nodes[2] = {65, 130};
        require(build_nodemask(high_nodes, mask, maxnode),
                "T-17b build_nodemask must accept node ids 65 and 130");
        constexpr std::size_t kBitsPerWord = 8U * sizeof(unsigned long);
        const std::size_t w65  = 65U / kBitsPerWord;
        const std::size_t w130 = 130U / kBitsPerWord;
        require(mask.size() > w130, "T-17b nodemask too small for node 130");
        require((mask[w65] & (1UL << (65U % kBitsPerWord))) != 0UL,
                "T-17b bit 65 missing in nodemask");
        require((mask[w130] & (1UL << (130U % kBitsPerWord))) != 0UL,
                "T-17b bit 130 missing in nodemask");
        require(maxnode == mask.size() * kBitsPerWord + 1UL, "T-17b maxnode off by one");

        const int too_high[1] = {1024};
        require(!build_nodemask(too_high, mask, maxnode),
                "T-17b build_nodemask must reject node id 1024");
    }

    // T-14: Host shard mbind diagnostics and /proc/self/numa_maps verification (no silent
    // degradation to Interleave).
    {
        const long psz = std::max(4096L, sysconf(_SC_PAGESIZE));
        const std::size_t half = static_cast<std::size_t>(psz);
        void* region = ::mmap(nullptr, 2 * half, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(region != MAP_FAILED, "T-14 mmap failed");
        const NumaTopology live = host_numa_topology();
        const int valid_node    = live.nodes.front().node_id;
        const std::array<HostNumaShard, 2> mixed_shards{
            HostNumaShard{0, half, valid_node},
            HostNumaShard{half, half, 999}, // nonexistent node -> mbind failure
        };
        std::vector<int> failed;
        const bool all_ok = bind_pinned_shards(region, 2 * half, mixed_shards, &failed);
        require(!all_ok && failed.size() == 1 && failed[0] == 999,
                "T-14 bind_pinned_shards must report failing node 999");
        static_cast<volatile char*>(region)[0] = 1;
        // Verify VMA 0 kept bind (not silently rewritten to interleave).
        std::ifstream maps("/proc/self/numa_maps");
        const auto addr_val = reinterpret_cast<std::uintptr_t>(region);
        char hexbuf[32];
        std::snprintf(hexbuf, sizeof(hexbuf), "%llx ", static_cast<unsigned long long>(addr_val));
        bool saw_non_interleave = false;
        std::string line;
        while (std::getline(maps, line)) {
            if (line.rfind(hexbuf, 0) == 0) {
                require(line.find("interleave") == std::string::npos,
                        "T-14 valid shard must not silently degrade to interleave");
                saw_non_interleave = true;
                break;
            }
        }
        require(saw_non_interleave, "T-14 could not inspect /proc/self/numa_maps for mmap shard");
        ::munmap(region, 2 * half);
    }
#endif
}

void test_live_host_and_cuda_numa() {
    const NumaTopology& topo = host_numa_topology();
    require(!topo.nodes.empty(), "host_numa_topology must yield >= 1 node");
    require(!topo.nodes[0].logical_cpus.empty(), "primary host node must have >= 1 CPU");
    const std::string summary = format_numa_summary(topo);
    require(!summary.empty(), "format_numa_summary must return readable text");

    const std::vector<int> cpus = select_worker_cpus(topo.nodes[0].node_id, 0);
    require(!cpus.empty(), "select_worker_cpus must select at least 1 CPU");
    std::set<int> uniq(cpus.begin(), cpus.end());
    require(uniq.size() == cpus.size(), "select_worker_cpus must return distinct CPUs");

#ifdef __linux__
    constexpr std::size_t kTestBytes = 16 * 1024;
    void* mem = ::mmap(nullptr, kTestBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(mem != MAP_FAILED, "mmap test allocation failed");
    const int target_node = topo.nodes.front().node_id;
    const bool bound      = bind_memory_to_node(mem, kTestBytes, target_node, /*move_pages=*/false);
    static_cast<volatile char*>(mem)[0] = 42;
    if (bound) {
        const int queried = query_page_numa_node(mem);
        require(queried == kNoNumaNode || queried == target_node,
                "query_page_numa_node disagreed with bind_memory_to_node");
    }
    ::munmap(mem, kTestBytes);
#endif

    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) { return; }

    const int dev_node = device_numa_node(0);
    {
        PinnedHostBuffer single(4096, dev_node,
                                dev_node >= 0 ? NumaPlacement::Bind : NumaPlacement::Default);
        require(single.data() != nullptr && single.size() == 4096,
                "NUMA-aware PinnedHostBuffer allocation failed");
        static_cast<std::uint8_t*>(single.data())[0] = 0x5A;
    }
    {
        const int second_node =
            topo.multi_node() ? topo.nodes.back().node_id : (dev_node >= 0 ? dev_node : 0);
        const std::array<PinnedShardRange, 2> shards{
            PinnedShardRange{0, 8192, dev_node >= 0 ? dev_node : 0},
            PinnedShardRange{8192, 8192, second_node},
        };
        PinnedHostBuffer sharded(16384, shards);
        require(sharded.data() != nullptr && sharded.size() == 16384,
                "sharded PinnedHostBuffer allocation failed");
        require(sharded.shard_mbind_failures() == 0,
                "valid PinnedHostBuffer shards reported mbind failure");
        static_cast<std::uint8_t*>(sharded.data())[0]     = 1;
        static_cast<std::uint8_t*>(sharded.data())[16383] = 2;

#ifdef __linux__
        const long psz         = std::max(4096L, sysconf(_SC_PAGESIZE));
        const std::size_t half = static_cast<std::size_t>(psz);
        const std::array<PinnedShardRange, 2> mixed{
            PinnedShardRange{0, half, dev_node >= 0 ? dev_node : 0},
            PinnedShardRange{half, half, 999},
        };
        PinnedHostBuffer diag(2 * half, mixed);
        require(diag.shard_mbind_failures() == 1,
                "T-14 GPU PinnedHostBuffer must count mbind failure on nonexistent node 999");
        require(diag.data() != nullptr && diag.size() == 2 * half,
                "T-14 GPU PinnedHostBuffer failed allocation");
#endif
    }
}

} // namespace

int main() {
    try {
        test_synthetic_supermicro_sysfs();
        test_plan_host_numa_policies();
        test_numa_review_fixes();
        test_numa_pr1_regressions();
        test_live_host_and_cuda_numa();
        std::cout << "test_numa: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
