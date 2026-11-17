#include "core/arena.h"
#include "core/numa.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#    include <sys/mman.h>
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
        static_cast<std::uint8_t*>(sharded.data())[0]     = 1;
        static_cast<std::uint8_t*>(sharded.data())[16383] = 2;
    }
}

} // namespace

int main() {
    try {
        test_synthetic_supermicro_sysfs();
        test_plan_host_numa_policies();
        test_live_host_and_cuda_numa();
        std::cout << "test_numa: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
