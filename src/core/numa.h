#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace ninfer {

inline constexpr int kNoNumaNode = -1;

// NUMA placement strategy for pinned buffers backing MoE experts and CPU worker pools.
enum class NumaPlacement : std::uint8_t {
    Default    = 0,
    Bind       = 1,
    Interleave = 2,
};

enum class HostNumaMode : std::uint8_t {
    Isolate = 0,
    Sharded = 1,
    Mirror  = 2,
};

struct HostNumaNodePlan {
    int node_id = 0;
    std::uint32_t worker_threads = 0;
    std::vector<int> worker_cpus;
};

struct HostNumaShard {
    std::size_t offset = 0;
    std::size_t bytes  = 0;
    int numa_node      = kNoNumaNode;
};

struct HostNumaPlan {
    HostNumaMode mode = HostNumaMode::Isolate;
    int primary_node  = 0;
    int service_cpu   = -1;
    std::vector<HostNumaNodePlan> node_plans;
    std::vector<HostNumaShard> pinned_shards;
};

struct NumaNodeInfo {
    int node_id              = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t free_bytes  = 0;
    // All online logical CPUs and deduplicated physical CPUs on this node allowed by affinity.
    std::vector<int> logical_cpus;
    std::vector<int> physical_cpus;
};

struct NumaTopology {
    std::vector<NumaNodeInfo> nodes;

    [[nodiscard]] bool multi_node() const noexcept { return nodes.size() > 1; }
    [[nodiscard]] std::size_t node_count() const noexcept { return nodes.size(); }
    [[nodiscard]] int index_of(int node_id) const noexcept;
    [[nodiscard]] const NumaNodeInfo* find(int node_id) const noexcept;
    [[nodiscard]] int node_for_cpu(int cpu_id) const noexcept;
};

// Parses Linux sysfs-style NUMA topology from `sys_base` (typically `/sys`), filtering CPUs
// against `allowed_cpus` when non-empty. Exposed for synthetic multi-socket unit testing.
[[nodiscard]] NumaTopology parse_numa_sysfs(const std::filesystem::path& sys_base,
                                            std::span<const int> allowed_cpus = {},
                                            std::uint32_t max_usable_cpus     = 0);

// Queries the live host NUMA topology (intersected with sched_getaffinity and cgroup CPU quota on
// Linux; returns a single logical node on platforms without sysfs NUMA).
[[nodiscard]] const NumaTopology& host_numa_topology();

// Returns the NUMA node attached to the CUDA device's PCIe root complex on Linux, or kNoNumaNode
// when unknown / single-node virtualized / non-Linux.
[[nodiscard]] int device_numa_node(int cuda_device);

// Reads `/sys/bus/pci/devices/<bus_id>/numa_node` relative to an arbitrary `sys_base`.
[[nodiscard]] int parse_pci_device_numa_node(const std::filesystem::path& sys_base,
                                             const std::string& pci_bus_id);

// Pins the calling thread to a specific logical CPU id. Returns true on success.
bool bind_thread_to_cpu(int cpu_id) noexcept;

// Pins the calling thread to all allowed CPUs on `node_id` (falling back to all logical CPUs on
// that node). Returns true on success.
bool bind_thread_to_node(int node_id) noexcept;

// Applies Linux `mbind` policy (`MPOL_BIND` or `MPOL_INTERLEAVE`) to the page-aligned range
// covering `[addr, addr + bytes)`. Fails gracefully (returns false) in restricted containers
// (`EPERM`/`ENOSYS`) or non-Linux builds.
bool bind_memory_nodes(void* addr, std::size_t bytes, std::span<const int> nodes,
                       bool interleave = false, bool move_pages = true) noexcept;

// Convenience overload for binding to a single node.
inline bool bind_memory_to_node(void* addr, std::size_t bytes, int node_id,
                                bool move_pages = true) noexcept {
    if (node_id < 0) { return false; }
    return bind_memory_nodes(addr, bytes, std::span<const int>{&node_id, 1}, false, move_pages);
}

// Queries the physical NUMA node backing the page containing `addr` via `get_mempolicy`.
// Returns kNoNumaNode if unavailable or unbacked.
[[nodiscard]] int query_page_numa_node(const void* addr) noexcept;

// RAII scope setting the calling thread's Linux memory allocation policy (`MPOL_BIND` or
// `MPOL_INTERLEAVE`) before allocating/faulting pages (e.g. `cudaHostAlloc`), restoring `MPOL_DEFAULT`
// on scope exit.
class ScopedThreadMemPolicy {
public:
    ScopedThreadMemPolicy(std::span<const int> nodes, bool interleave = false) noexcept;
    ScopedThreadMemPolicy(int node_id, NumaPlacement placement) noexcept;
    ~ScopedThreadMemPolicy() noexcept;

    ScopedThreadMemPolicy(const ScopedThreadMemPolicy&)            = delete;
    ScopedThreadMemPolicy& operator=(const ScopedThreadMemPolicy&) = delete;

    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    bool active_ = false;
};

// Resolves NUMA placement and per-node CPU thread assignment for host-resident MoE execution.
[[nodiscard]] HostNumaPlan plan_host_numa(const NumaTopology& topo, int gpu_node,
                                          std::uint64_t pinned_bytes,
                                          std::uint32_t requested_threads,
                                          NumaPolicy policy,
                                          std::uint64_t page_size = 4096);

// Returns worker CPU IDs pinned to `preferred_node` (physical cores first, then local SMT
// siblings) when `threads` fits on that node (0 picks physical cores of `preferred_node`), or
// spans physical then logical CPUs across all nodes when `threads` exceeds one node's capacity.
[[nodiscard]] std::vector<int> select_worker_cpus(int preferred_node,
                                                  std::uint32_t requested_threads = 0);

// Returns CPU ids formatted cleanly (e.g. "0-7,16-23") for startup diagnostics.
[[nodiscard]] std::string format_numa_summary(const NumaTopology& topo);

} // namespace ninfer
