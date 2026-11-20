#include "core/numa.h"

#include "core/wide_math.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#ifdef __linux__
#    include <linux/mempolicy.h>
#    include <pthread.h>
#    include <sched.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

namespace ninfer {
namespace {

std::vector<int> parse_cpu_range_list(const std::string& text) {
    std::vector<int> result;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) { continue; }
        const auto last  = item.find_last_not_of(" \t\r\n");
        const std::string token = item.substr(first, last - first + 1);
        const auto dash = token.find('-');
        if (dash == std::string::npos) {
            char* end = nullptr;
            const long v = std::strtol(token.c_str(), &end, 10);
            if (end != token.c_str() && *end == '\0' && v >= 0 && v < 65536) {
                result.push_back(static_cast<int>(v));
            }
            continue;
        }
        const std::string lo_str = token.substr(0, dash);
        const std::string hi_str = token.substr(dash + 1);
        char* end_a              = nullptr;
        char* end_b              = nullptr;
        const long lo            = std::strtol(lo_str.c_str(), &end_a, 10);
        const long hi            = std::strtol(hi_str.c_str(), &end_b, 10);
        if (end_a != lo_str.c_str() && *end_a == '\0' && end_b != hi_str.c_str() &&
            *end_b == '\0' && lo >= 0 && lo <= hi && hi < 65536) {
            for (long c = lo; c <= hi; ++c) { result.push_back(static_cast<int>(c)); }
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

bool read_first_int(const std::filesystem::path& path, int& out) {
    std::ifstream f(path);
    if (!f) { return false; }
    return static_cast<bool>(f >> out);
}

bool cpu_online_in_sysfs(const std::filesystem::path& cpu_root, int cpu_id) {
    const std::filesystem::path online_path =
        cpu_root / ("cpu" + std::to_string(cpu_id)) / "online";
    int online = 1;
    if (read_first_int(online_path, online)) { return online != 0; }
    return true;
}

void parse_node_meminfo(const std::filesystem::path& meminfo_path, std::uint64_t& total_bytes,
                        std::uint64_t& free_bytes) {
    std::ifstream f(meminfo_path);
    if (!f) { return; }
    std::uint64_t mem_free = 0, active_file = 0, inactive_file = 0, file_pages = 0,
                  sreclaimable = 0, shmem = 0;
    bool has_active_inactive = false;
    std::string line;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) { continue; }
        std::string_view key = std::string_view(line).substr(0, colon);
        while (!key.empty() && (key.front() == ' ' || key.front() == '\t')) {
            key.remove_prefix(1);
        }
        while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) {
            key.remove_suffix(1);
        }
        // Per-node meminfo prefixes keys with "Node <id> "; strip to the bare field name so
        // *HugePages* / ShmemPmdMapped / FileHugePages never alias MemFree, FilePages, or Shmem.
        if (key.rfind("Node ", 0) == 0) {
            const auto last_sp = key.rfind(' ');
            if (last_sp != std::string_view::npos) { key.remove_prefix(last_sp + 1); }
        }
        std::uint64_t kb = 0;
        {
            std::istringstream val(line.substr(colon + 1));
            if (!(val >> kb)) { continue; }
        }
        const std::uint64_t bytes = kb * 1024ULL;
        if (key == "MemTotal") {
            total_bytes = bytes;
        } else if (key == "MemFree") {
            mem_free = bytes;
        } else if (key == "Active(file)") {
            active_file         = bytes;
            has_active_inactive = true;
        } else if (key == "Inactive(file)") {
            inactive_file       = bytes;
            has_active_inactive = true;
        } else if (key == "FilePages") {
            file_pages = bytes;
        } else if (key == "SReclaimable") {
            sreclaimable = bytes;
        } else if (key == "Shmem") {
            shmem = bytes;
        }
    }
    std::uint64_t reclaimable = has_active_inactive ? (active_file + inactive_file) : file_pages;
    // Shmem (tmpfs / shared mappings) is resident and non-file-reclaimable; counting it as free
    // would overstate the RAM a pinned buffer can actually take on this node.
    if (reclaimable > shmem) { reclaimable -= shmem; } else { reclaimable = 0; }
    std::uint64_t avail = mem_free + reclaimable + sreclaimable;
    if (total_bytes > 0) { avail = std::min(avail, total_bytes); }
    free_bytes = avail;
}

std::string to_lower_ascii(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

#ifdef __linux__
std::vector<int> current_allowed_cpus(std::uint32_t& quota_cpus_out) {
    std::vector<int> cpus;
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) { cpus.push_back(cpu); }
        }
    }
    double quota = -1.0, period = 0.0;
    if (std::ifstream v2("/sys/fs/cgroup/cpu.max"); v2) {
        std::string limit;
        if (v2 >> limit >> period && limit != "max") { quota = std::strtod(limit.c_str(), nullptr); }
    } else if (std::ifstream q("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
               p("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
               q && p) {
        if (!(q >> quota && p >> period)) { quota = -1.0; }
    }
    if (quota > 0.0 && period > 0.0) {
        quota_cpus_out = std::max(1U, static_cast<unsigned>(quota / period));
    } else {
        quota_cpus_out = 0;
    }
    return cpus;
}
#endif

} // namespace

int NumaTopology::index_of(int node_id) const noexcept {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].node_id == node_id) { return static_cast<int>(i); }
    }
    return kNoNumaNode;
}

const NumaNodeInfo* NumaTopology::find(int node_id) const noexcept {
    const int idx = index_of(node_id);
    return idx >= 0 ? &nodes[static_cast<std::size_t>(idx)] : nullptr;
}

int NumaTopology::node_for_cpu(int cpu_id) const noexcept {
    for (const auto& node : nodes) {
        if (std::binary_search(node.logical_cpus.begin(), node.logical_cpus.end(), cpu_id)) {
            return node.node_id;
        }
    }
    return kNoNumaNode;
}

NumaTopology parse_numa_sysfs(const std::filesystem::path& sys_base,
                              std::span<const int> allowed_cpus,
                              std::uint32_t max_usable_cpus) {
    NumaTopology topo;
    const std::filesystem::path node_root = sys_base / "devices/system/node";
    const std::filesystem::path cpu_root  = sys_base / "devices/system/cpu";

    std::error_code ec;
    if (!std::filesystem::exists(node_root, ec)) { return topo; }

    std::set<int> allowed_set(allowed_cpus.begin(), allowed_cpus.end());
    std::set<int> online_nodes;
    bool have_online_list = false;
    if (std::ifstream f(node_root / "online"); f) {
        std::string line;
        if (std::getline(f, line)) {
            const auto parsed = parse_cpu_range_list(line);
            online_nodes.insert(parsed.begin(), parsed.end());
            have_online_list = !parsed.empty();
        }
    }

    std::vector<int> candidate_nodes;
    for (const auto& entry : std::filesystem::directory_iterator(node_root, ec)) {
        if (!entry.is_directory(ec)) { continue; }
        const std::string name = entry.path().filename().string();
        if (name.rfind("node", 0) != 0 || name.size() <= 4) { continue; }
        int id              = -1;
        const char* p_begin = name.data() + 4;
        const char* p_end   = name.data() + name.size();
        const auto [ptr, ec_parse] = std::from_chars(p_begin, p_end, id);
        if (ec_parse != std::errc{} || ptr != p_end || id < 0) { continue; }
        if (have_online_list && online_nodes.find(id) == online_nodes.end()) { continue; }
        candidate_nodes.push_back(id);
    }
    std::sort(candidate_nodes.begin(), candidate_nodes.end());

    for (const int id : candidate_nodes) {
        const std::filesystem::path dir = node_root / ("node" + std::to_string(id));
        NumaNodeInfo info;
        info.node_id = id;
        parse_node_meminfo(dir / "meminfo", info.total_bytes, info.free_bytes);

        std::string cpulist;
        if (std::ifstream f(dir / "cpulist"); f) { std::getline(f, cpulist); }
        std::vector<int> node_cpus = parse_cpu_range_list(cpulist);
        std::vector<std::pair<int, int>> seen_cores;
        for (const int cpu : node_cpus) {
            if (!allowed_set.empty() && allowed_set.find(cpu) == allowed_set.end()) { continue; }
            if (!cpu_online_in_sysfs(cpu_root, cpu)) { continue; }
            info.logical_cpus.push_back(cpu);
            int pkg  = id;
            int core = cpu;
            const std::filesystem::path topo_dir =
                cpu_root / ("cpu" + std::to_string(cpu)) / "topology";
            (void)read_first_int(topo_dir / "physical_package_id", pkg);
            (void)read_first_int(topo_dir / "core_id", core);
            const std::pair<int, int> key{pkg, core};
            if (std::find(seen_cores.begin(), seen_cores.end(), key) == seen_cores.end()) {
                seen_cores.push_back(key);
                info.physical_cpus.push_back(cpu);
            }
        }
        // Only expose nodes that either hold usable CPUs or report positive memory capacity.
        if (!info.logical_cpus.empty() || info.total_bytes > 0) {
            topo.nodes.push_back(std::move(info));
        }
    }

    if (max_usable_cpus > 0 && !topo.nodes.empty()) {
        std::size_t total_logical = 0;
        for (const auto& n : topo.nodes) { total_logical += n.logical_cpus.size(); }
        if (total_logical > max_usable_cpus) {
            const std::size_t kNodes = topo.nodes.size();
            const std::size_t base   = max_usable_cpus / kNodes;
            std::size_t rem          = max_usable_cpus % kNodes;
            for (auto& n : topo.nodes) {
                std::size_t cap = base + (rem > 0 ? 1U : 0U);
                if (rem > 0) { --rem; }
                cap = std::min(cap, n.logical_cpus.size());
                if (cap == 0 && !n.logical_cpus.empty() && max_usable_cpus >= kNodes) { cap = 1; }
                if (n.logical_cpus.size() > cap) {
                    std::set<int> kept_logical;
                    // Prefer keeping distinct physical cores first up to cap.
                    for (std::size_t i = 0; i < n.physical_cpus.size() && kept_logical.size() < cap;
                         ++i) {
                        kept_logical.insert(n.physical_cpus[i]);
                    }
                    for (std::size_t i = 0; i < n.logical_cpus.size() && kept_logical.size() < cap;
                         ++i) {
                        kept_logical.insert(n.logical_cpus[i]);
                    }
                    n.logical_cpus.assign(kept_logical.begin(), kept_logical.end());
                    n.physical_cpus.erase(
                        std::remove_if(n.physical_cpus.begin(), n.physical_cpus.end(),
                                       [&](int c) { return kept_logical.find(c) == kept_logical.end(); }),
                        n.physical_cpus.end());
                }
            }
        }
    }
    return topo;
}

NumaTopology host_numa_topology() {
    NumaTopology out;
#ifdef __linux__
    std::uint32_t quota_cap     = 0;
    const std::vector<int> mask = current_allowed_cpus(quota_cap);
    out                         = parse_numa_sysfs("/sys", mask, quota_cap);
    if (out.nodes.empty()) {
        NumaNodeInfo fallback;
        fallback.node_id = 0;
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long psize = sysconf(_SC_PAGESIZE);
        const long avail = sysconf(_SC_AVPHYS_PAGES);
        if (pages > 0 && psize > 0) {
            fallback.total_bytes = static_cast<std::uint64_t>(pages) * psize;
        }
        if (avail > 0 && psize > 0) {
            fallback.free_bytes = static_cast<std::uint64_t>(avail) * psize;
        }
        fallback.logical_cpus  = mask;
        fallback.physical_cpus = mask;
        if (!fallback.logical_cpus.empty() || fallback.total_bytes > 0) {
            out.nodes.push_back(std::move(fallback));
        }
    }
#else
    NumaNodeInfo single;
    single.node_id       = 0;
    const unsigned count = std::max(1U, std::thread::hardware_concurrency());
    for (unsigned i = 0; i < count; ++i) {
        single.logical_cpus.push_back(static_cast<int>(i));
        single.physical_cpus.push_back(static_cast<int>(i));
    }
    out.nodes.push_back(std::move(single));
#endif
    return out;
}

int parse_pci_device_numa_node(const std::filesystem::path& sys_base,
                               const std::string& pci_bus_id) {
    if (pci_bus_id.empty()) { return kNoNumaNode; }
    const std::filesystem::path numa_path =
        sys_base / "bus/pci/devices" / to_lower_ascii(pci_bus_id) / "numa_node";
    int node = kNoNumaNode;
    if (!read_first_int(numa_path, node) || node < 0) { return kNoNumaNode; }
    return node;
}

int device_numa_node(int cuda_device) {
    if (cuda_device < 0) { return kNoNumaNode; }
#ifdef __linux__
    static std::mutex cache_mutex;
    static std::unordered_map<int, int> cache;
    {
        std::lock_guard lock(cache_mutex);
        if (const auto it = cache.find(cuda_device); it != cache.end()) { return it->second; }
    }
    char bus_id[64]{};
    int node = kNoNumaNode;
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), cuda_device) == cudaSuccess) {
        node = parse_pci_device_numa_node("/sys", std::string(bus_id));
    }
    {
        std::lock_guard lock(cache_mutex);
        cache.emplace(cuda_device, node);
    }
    return node;
#else
    (void)cuda_device;
    return kNoNumaNode;
#endif
}

bool bind_thread_to_cpu(int cpu_id) noexcept {
#ifdef __linux__
    // CPU_SET writes past the static cpu_set_t (CPU_SETSIZE bits, 1024 on Linux) for ids at or
    // above the bound -- a stack overflow on machines with more logical CPUs than that (the 192-
    // core Supermicro X9DAX is one). bind_thread_to_node already guards the same bound.
    if (cpu_id < 0 || cpu_id >= CPU_SETSIZE) { return false; }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    return false;
#endif
}

bool bind_thread_to_node(int node_id) noexcept {
    if (node_id < 0) { return false; }
#ifdef __linux__
    const NumaTopology topo  = host_numa_topology();
    const NumaNodeInfo* info = topo.find(node_id);
    if (!info || info->logical_cpus.empty()) { return false; }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : info->logical_cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) { CPU_SET(cpu, &set); }
    }
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    return false;
#endif
}

bool build_nodemask(std::span<const int> nodes, std::vector<unsigned long>& mask,
                    unsigned long& maxnode_out) {
#ifdef __linux__
    if (nodes.empty()) { return false; }
    int max_id = -1;
    for (const int n : nodes) {
        if (n < 0 || n >= 1024) { return false; }
        max_id = std::max(max_id, n);
    }
    constexpr std::size_t kBitsPerWord = 8U * sizeof(unsigned long);
    const std::size_t words = static_cast<std::size_t>(max_id) / kBitsPerWord + 1U;
    mask.assign(words, 0UL);
    for (const int n : nodes) {
        const auto u = static_cast<std::size_t>(n);
        mask[u / kBitsPerWord] |= (1UL << (u % kBitsPerWord));
    }
    // Linux kernel `get_nodes()` subtracts 1 from maxnode internally (`--maxnode`).
    maxnode_out = static_cast<unsigned long>(words * kBitsPerWord + 1U);
    return true;
#else
    (void)nodes;
    (void)mask;
    (void)maxnode_out;
    return false;
#endif
}

bool bind_memory_nodes(void* addr, std::size_t bytes, std::span<const int> nodes,
                       bool interleave, bool move_pages) noexcept {
    if (!addr || bytes == 0 || nodes.empty()) { return false; }
#if defined(__linux__) && defined(__NR_mbind) && defined(__NR_set_mempolicy) &&                    \
    defined(__NR_get_mempolicy)
    std::vector<unsigned long> mask;
    unsigned long maxnode = 0;
    if (!build_nodemask(nodes, mask, maxnode)) { return false; }
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) { page_size = 4096; }
    const auto psz      = static_cast<std::uintptr_t>(page_size);
    const auto raw      = reinterpret_cast<std::uintptr_t>(addr);
    const auto aligned_begin = raw & ~(psz - 1U);
    const auto raw_end       = raw + bytes;
    const auto aligned_end   = (raw_end + psz - 1U) & ~(psz - 1U);
    if (aligned_end <= aligned_begin) { return false; }
    const unsigned long mode  = interleave ? MPOL_INTERLEAVE : MPOL_BIND;
    const unsigned flags      = move_pages ? MPOL_MF_MOVE : 0U;
    long rc = syscall(__NR_mbind, reinterpret_cast<void*>(aligned_begin),
                      static_cast<unsigned long>(aligned_end - aligned_begin), mode, mask.data(),
                      maxnode, flags);
    if (rc != 0 && move_pages && errno == EIO) {
        rc = syscall(__NR_mbind, reinterpret_cast<void*>(aligned_begin),
                     static_cast<unsigned long>(aligned_end - aligned_begin), mode, mask.data(),
                     maxnode, 0U);
    }
    return rc == 0;
#else
    (void)interleave;
    (void)move_pages;
    return false;
#endif
}

bool bind_pinned_shards(void* base, std::size_t size_bytes, std::span<const HostNumaShard> shards,
                        std::vector<int>* failed_nodes) noexcept {
    if (!base || size_bytes == 0) { return false; }
    bool all_ok = true;
    for (const auto& s : shards) {
        if (s.numa_node >= 0 && s.bytes > 0 && s.offset < size_bytes) {
            const std::size_t clamped = std::min(s.bytes, size_bytes - s.offset);
            if (!bind_memory_to_node(static_cast<std::byte*>(base) + s.offset, clamped,
                                     s.numa_node, /*move_pages=*/false)) {
                all_ok = false;
                if (failed_nodes) { failed_nodes->push_back(s.numa_node); }
            }
        }
    }
    return all_ok;
}

int query_page_numa_node(const void* addr) noexcept {
    if (!addr) { return kNoNumaNode; }
#if defined(__linux__) && defined(__NR_get_mempolicy)
    int node = kNoNumaNode;
    const long rc =
        syscall(__NR_get_mempolicy, &node, nullptr, 0UL, const_cast<void*>(addr),
                static_cast<unsigned long>(MPOL_F_NODE | MPOL_F_ADDR));
    return (rc == 0 && node >= 0) ? node : kNoNumaNode;
#else
    return kNoNumaNode;
#endif
}

#if defined(__linux__) && defined(__NR_set_mempolicy) && defined(__NR_get_mempolicy)
// Captures the calling thread's current memory policy so it can be restored when a
// ScopedThreadMemPolicy scope exits. `get_mempolicy` with a null address returns the thread-wide
// policy; MPOL_DEFAULT carries no nodemask, and the other modes fill `mask`/`maxnode`.
static bool capture_thread_mempolicy(int& mode_out, std::vector<unsigned long>& mask_out) noexcept {
    unsigned long maxnode = 0;
    constexpr std::size_t mask_words = 1024 / (8 * sizeof(unsigned long));
    mask_out.assign(mask_words, 0UL);
    const long rc = syscall(__NR_get_mempolicy, &mode_out, mask_out.data(), 1024UL + 1UL,
                            static_cast<void*>(nullptr), 0UL);
    if (rc != 0) { return false; }
    maxnode = 1024UL + 1UL;
    (void)maxnode;
    return true;
}

static void restore_thread_mempolicy(int mode, const std::vector<unsigned long>& mask) noexcept {
    if (mode == MPOL_DEFAULT || mode < 0) {
        (void)syscall(__NR_set_mempolicy, MPOL_DEFAULT, nullptr, 0UL);
        return;
    }
    (void)syscall(__NR_set_mempolicy, mode, mask.data(), 1024UL + 1UL);
}
#endif

ScopedThreadMemPolicy::ScopedThreadMemPolicy(std::span<const int> nodes,
                                             bool interleave) noexcept {
#if defined(__linux__) && defined(__NR_set_mempolicy) && defined(__NR_get_mempolicy)
    captured_ = capture_thread_mempolicy(prev_mode_, prev_mask_);
    if (!captured_) { return; }
    std::vector<unsigned long> mask;
    unsigned long maxnode = 0;
    if (build_nodemask(nodes, mask, maxnode)) {
        const int mode = interleave ? MPOL_INTERLEAVE : MPOL_BIND;
        if (syscall(__NR_set_mempolicy, mode, mask.data(), maxnode) == 0) { active_ = true; }
    }
#else
    (void)nodes;
    (void)interleave;
#endif
}

ScopedThreadMemPolicy::ScopedThreadMemPolicy(int node_id, NumaPlacement placement) noexcept {
#if defined(__linux__) && defined(__NR_set_mempolicy) && defined(__NR_get_mempolicy)
    captured_ = capture_thread_mempolicy(prev_mode_, prev_mask_);
    if (!captured_) { return; }
    if (placement == NumaPlacement::Bind && node_id >= 0) {
        std::vector<unsigned long> mask;
        unsigned long maxnode = 0;
        const int single[1]   = {node_id};
        if (build_nodemask(single, mask, maxnode) &&
            syscall(__NR_set_mempolicy, MPOL_BIND, mask.data(), maxnode) == 0) {
            active_ = true;
        }
    } else if (placement == NumaPlacement::Interleave) {
        const auto& topo = host_numa_topology();
        if (topo.nodes.size() > 1) {
            std::vector<int> ids;
            ids.reserve(topo.nodes.size());
            for (const auto& n : topo.nodes) { ids.push_back(n.node_id); }
            std::vector<unsigned long> mask;
            unsigned long maxnode = 0;
            if (build_nodemask(ids, mask, maxnode) &&
                syscall(__NR_set_mempolicy, MPOL_INTERLEAVE, mask.data(), maxnode) == 0) {
                active_ = true;
            }
        }
    }
#else
    (void)node_id;
    (void)placement;
#endif
}

ScopedThreadMemPolicy::~ScopedThreadMemPolicy() noexcept {
#if defined(__linux__) && defined(__NR_set_mempolicy) && defined(__NR_get_mempolicy)
    if (active_ && captured_) { restore_thread_mempolicy(prev_mode_, prev_mask_); }
#else
    (void)prev_mode_;
    (void)prev_mask_;
    (void)captured_;
#endif
}

HostNumaPlan plan_host_numa(const NumaTopology& topo, int gpu_node,
                              std::uint64_t pinned_bytes, std::uint32_t requested_threads,
                              NumaPolicy policy, std::uint64_t page_size) {
    HostNumaPlan plan;
    if (topo.nodes.empty()) { return plan; }
    std::uint64_t native_page = 4096ULL;
#ifdef __linux__
    const long native = sysconf(_SC_PAGESIZE);
    if (native > 0) { native_page = static_cast<std::uint64_t>(native); }
#endif
    // A zero page_size means "use the system's native page size" (the default argument).
    if (page_size == 0) {
        page_size = native_page;
    } else if ((page_size & (page_size - 1ULL)) != 0ULL || page_size < native_page) {
        throw std::invalid_argument("plan_host_numa: page_size must be a power of two >= native page size");
    }

    const NumaNodeInfo* gpu_info = (gpu_node >= 0) ? topo.find(gpu_node) : nullptr;
    if (!gpu_info) { gpu_info = &topo.nodes.front(); }
    plan.primary_node = gpu_info->node_id;

    const auto headroom_for = [](const NumaNodeInfo& n) -> std::uint64_t {
        return n.total_bytes > 0
                   ? std::min<std::uint64_t>(2ULL << 30, n.total_bytes / 16ULL)
                   : 0ULL;
    };
    const auto usable_free_for = [&](const NumaNodeInfo& n) -> std::uint64_t {
        const std::uint64_t h = headroom_for(n);
        return n.free_bytes > h ? (n.free_bytes - h) : 0ULL;
    };

    const bool unknown_pinned          = (pinned_bytes == kUnknownPinnedBytes);
    const std::uint64_t effective_pinned = unknown_pinned ? 0ULL : pinned_bytes;

    HostNumaMode mode = HostNumaMode::Isolate;
    if (topo.multi_node()) {
        if (policy == NumaPolicy::Isolate) {
            mode = HostNumaMode::Isolate;
        } else if (policy == NumaPolicy::Sharded) {
            mode = HostNumaMode::Sharded;
        } else if (policy == NumaPolicy::Mirror) {
            mode = HostNumaMode::Mirror;
        } else if (unknown_pinned || effective_pinned == 0) {
            // Disk/unpinned residency relies on OS page cache across nodes; keep workers spanning
            // all sockets instead of collapsing to the GPU's local socket.
            mode = HostNumaMode::Sharded;
        } else {
            // Auto: stay isolated to the GPU's local node when its available RAM holds the
            // weights (with OS headroom) and requested CPU threads fit within its physical cores.
            const std::uint64_t headroom = headroom_for(*gpu_info);
            const bool ram_known         = (gpu_info->total_bytes > 0);
            const bool ram_fits =
                !ram_known || (gpu_info->free_bytes >= effective_pinned + headroom);
            const std::size_t local_phys = gpu_info->physical_cpus.empty()
                                               ? gpu_info->logical_cpus.size()
                                               : gpu_info->physical_cpus.size();
            const bool threads_fit = (requested_threads == 0) || (requested_threads <= local_phys);

            if (ram_fits && threads_fit) {
                mode = HostNumaMode::Isolate;
            } else if (ram_fits) {
                std::uint64_t total_free = 0;
                bool every_node_fits     = true;
                const std::uint64_t per_node_need =
                    effective_pinned + std::min<std::uint64_t>(1ULL << 30, effective_pinned / 4);
                for (const auto& n : topo.nodes) {
                    total_free += n.free_bytes;
                    if (n.total_bytes > 0 && n.free_bytes < per_node_need) {
                        every_node_fits = false;
                    }
                }
                const auto mirror_threshold =
                    static_cast<std::uint64_t>(2.2 * static_cast<double>(effective_pinned));
                mode = (every_node_fits && total_free >= mirror_threshold) ? HostNumaMode::Mirror
                                                                           : HostNumaMode::Sharded;
            } else {
                mode = HostNumaMode::Sharded;
            }
        }
    }
    plan.mode = mode;

    // Validate headroom-adjusted free RAM before returning a plan that would OOM or degrade.
    if (effective_pinned > 0) {
        if (mode == HostNumaMode::Sharded) {
            std::uint64_t total_usable = 0;
            bool all_known             = true;
            for (const auto& n : topo.nodes) {
                total_usable += usable_free_for(n);
                if (n.total_bytes == 0) { all_known = false; }
            }
            if (all_known && total_usable < effective_pinned) {
                throw std::runtime_error(
                    "NUMA sharded: cannot pin " + std::to_string(effective_pinned) +
                    " B across " + std::to_string(topo.nodes.size()) +
                    " node(s) with only " + std::to_string(total_usable) + " B usable free");
            }
        } else if (policy != NumaPolicy::Auto) {
            const std::uint64_t headroom = headroom_for(*gpu_info);
            if (gpu_info->total_bytes > 0 && gpu_info->free_bytes < effective_pinned + headroom) {
                throw std::runtime_error(
                    std::string("NUMA ") + (mode == HostNumaMode::Mirror ? "mirror" : "isolate") +
                    ": node " + std::to_string(gpu_info->node_id) + " has only " +
                    std::to_string(gpu_info->free_bytes) + " B free (needs " +
                    std::to_string(effective_pinned + headroom) + " B including headroom)");
            }
        }
    }

    const auto node_phys = [](const NumaNodeInfo& n) -> const std::vector<int>& {
        return !n.physical_cpus.empty() ? n.physical_cpus : n.logical_cpus;
    };
    // Returns distinct physical cores first, followed by remaining SMT siblings so requested
    // worker counts above physical core count still spread across distinct logical CPUs.
    const auto node_ordered_cpus = [&](const NumaNodeInfo& n) -> std::vector<int> {
        std::vector<int> out = node_phys(n);
        for (const int cpu : n.logical_cpus) {
            if (std::find(out.begin(), out.end(), cpu) == out.end()) { out.push_back(cpu); }
        }
        return out;
    };

    constexpr std::uint32_t kMaxSocketAutoWorkers = 16U;
    constexpr std::uint32_t kMaxTotalAutoWorkers  = 32U;

    if (mode == HostNumaMode::Isolate || !topo.multi_node()) {
        const auto& phys      = node_phys(*gpu_info);
        const auto cpus       = node_ordered_cpus(*gpu_info);
        std::uint32_t workers = 0;
        if (phys.empty()) {
            workers = 0;
        } else if (requested_threads > 0) {
            workers = requested_threads;
        } else {
            const std::uint32_t count = static_cast<std::uint32_t>(phys.size());
            workers = std::min(kMaxSocketAutoWorkers,
                               count > 2 ? (count - 2U) : std::max(1U, count));
        }
        HostNumaNodePlan np;
        np.node_id        = gpu_info->node_id;
        np.worker_threads = workers;
        // On the primary node the service thread itself participates as compute worker 0, so
        // `worker_cpus` holds up to (workers - 1) background pool pins and `service_cpu` gets
        // the remaining distinct CPU.
        if (!cpus.empty() && workers > 0) {
            const std::uint32_t bg_needed = workers > 1 ? workers - 1U : 0U;
            if (cpus.size() >= workers) {
                for (std::uint32_t i = 0; i < bg_needed; ++i) { np.worker_cpus.push_back(cpus[i]); }
                plan.service_cpu = cpus[bg_needed];
            } else {
                // Fewer logical CPUs than requested threads: wrap cleanly across available slots.
                plan.service_cpu = cpus.back();
                for (std::uint32_t i = 0; i < bg_needed; ++i) {
                    np.worker_cpus.push_back(cpus[i % cpus.size()]);
                }
            }
        }
        plan.node_plans.push_back(std::move(np));
        if (effective_pinned > 0) {
            plan.pinned_shards.push_back(HostNumaShard{
                .offset    = 0,
                .bytes     = static_cast<std::size_t>(effective_pinned),
                .numa_node = plan.primary_node,
            });
        }
        return plan;
    }

    // Multi-socket Sharded or Mirror: place primary node first so index 0 is the GPU's socket.
    std::vector<const NumaNodeInfo*> ordered;
    ordered.push_back(gpu_info);
    for (const auto& n : topo.nodes) {
        if (n.node_id != gpu_info->node_id) { ordered.push_back(&n); }
    }

    if (requested_threads == 0) {
        std::uint32_t remaining_budget = kMaxTotalAutoWorkers;
        for (std::size_t idx = 0; idx < ordered.size(); ++idx) {
            const auto& n            = *ordered[idx];
            const auto& cpus         = node_phys(n);
            const std::uint32_t phys = static_cast<std::uint32_t>(cpus.size());
            std::uint32_t w          = 0;
            HostNumaNodePlan np;
            np.node_id = n.node_id;
            if (phys > 0 && remaining_budget > 0) {
                if (idx == 0) {
                    w = std::min({kMaxSocketAutoWorkers, remaining_budget,
                                  phys > 2 ? (phys - 2U) : std::max(1U, phys)});
                    const std::uint32_t bg = w > 1 ? w - 1U : 0U;
                    plan.service_cpu = cpus[std::min<std::size_t>(bg, cpus.size() - 1U)];
                    for (std::uint32_t j = 0; j < bg && j < cpus.size(); ++j) {
                        np.worker_cpus.push_back(cpus[j]);
                    }
                } else {
                    w = std::min({kMaxSocketAutoWorkers, remaining_budget,
                                  phys > 1 ? (phys - 1U) : std::max(1U, phys)});
                    for (std::uint32_t j = 0; j < w && j < cpus.size(); ++j) {
                        np.worker_cpus.push_back(cpus[j]);
                    }
                }
            }
            np.worker_threads = w;
            remaining_budget -= w;
            plan.node_plans.push_back(std::move(np));
        }
    } else {
        std::size_t total_phys = 0;
        for (const auto* n : ordered) { total_phys += node_phys(*n).size(); }
        std::vector<std::uint32_t> counts(ordered.size(), 0);
        if (total_phys > 0) {
            std::uint32_t remaining = requested_threads;
            for (std::size_t idx = 0; idx < ordered.size(); ++idx) {
                const std::size_t p = node_phys(*ordered[idx]).size();
                if (p == 0) { continue; }
                const auto w = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(requested_threads) * p) / total_phys);
                counts[idx] = w;
                remaining -= w;
            }
            for (std::size_t idx = 0; idx < ordered.size() && remaining > 0; ++idx) {
                if (node_phys(*ordered[idx]).empty()) { continue; }
                ++counts[idx];
                --remaining;
            }
        }
        for (std::size_t idx = 0; idx < ordered.size(); ++idx) {
            const auto& n    = *ordered[idx];
            const auto cpus  = node_ordered_cpus(n);
            HostNumaNodePlan np;
            np.node_id        = n.node_id;
            np.worker_threads = counts[idx];
            if (!cpus.empty() && counts[idx] > 0) {
                if (idx == 0) {
                    const std::uint32_t bg = counts[idx] > 1 ? counts[idx] - 1U : 0U;
                    plan.service_cpu = cpus[std::min<std::size_t>(bg, cpus.size() - 1U)];
                    for (std::uint32_t j = 0; j < bg; ++j) {
                        np.worker_cpus.push_back(cpus[j % cpus.size()]);
                    }
                } else {
                    for (std::uint32_t j = 0; j < counts[idx]; ++j) {
                        np.worker_cpus.push_back(cpus[j % cpus.size()]);
                    }
                }
            }
            plan.node_plans.push_back(std::move(np));
        }
    }

    if (effective_pinned > 0) {
        if (mode == HostNumaMode::Mirror) {
            plan.pinned_shards.push_back(HostNumaShard{
                .offset    = 0,
                .bytes     = static_cast<std::size_t>(effective_pinned),
                .numa_node = plan.primary_node,
            });
        } else {
            const std::uint64_t psize = page_size;
            const std::size_t kNodes  = ordered.size();
            // Weight shards by usable free memory (free - headroom) when node totals are known
            // so a tight socket never receives more bytes than its free - headroom budget; fall
            // back to raw free_bytes when totals are omitted, or even split if unknown.
            std::vector<std::uint64_t> weights(kNodes, 0);
            std::uint64_t weight_sum = 0;
            bool all_totals          = true;
            bool all_free            = true;
            for (const auto* n : ordered) {
                if (n->total_bytes == 0) { all_totals = false; }
                if (n->free_bytes == 0) { all_free = false; }
            }
            bool weighted = false;
            if (all_totals) {
                for (std::size_t idx = 0; idx < kNodes; ++idx) {
                    weights[idx] = usable_free_for(*ordered[idx]);
                    weight_sum  += weights[idx];
                }
                weighted = (weight_sum > 0);
            } else if (all_free) {
                for (std::size_t idx = 0; idx < kNodes; ++idx) {
                    weights[idx] = ordered[idx]->free_bytes;
                    weight_sum  += weights[idx];
                }
                weighted = (weight_sum > 0);
            }
            if (!weighted) {
                weights.assign(kNodes, 1ULL);
                weight_sum = kNodes;
            }
            std::size_t last_active = 0;
            for (std::size_t idx = 0; idx < kNodes; ++idx) {
                if (weights[idx] > 0) { last_active = idx; }
            }

            std::uint64_t prev       = 0;
            std::uint64_t cumulative = 0;
            for (std::size_t idx = 0; idx < kNodes; ++idx) {
                if (weights[idx] == 0) { continue; }
                std::uint64_t next;
                if (idx == last_active) {
                    // Last active shard keeps the exact unaligned tail so sum(bytes) == pinned_bytes
                    // while its starting offset remains page-aligned.
                    next = effective_pinned;
                } else {
                    cumulative += weights[idx];
                    // Portable 128-bit product/divide: pinned_bytes * cumulative overflows uint64
                    // at GiB scale, and __uint128_t is unavailable on MSVC.
                    const core::Uint128 prod = core::wide_multiply(effective_pinned, cumulative);
                    std::uint64_t rem        = prod.high % weight_sum;
                    std::uint64_t q          = 0;
                    for (int bit = 63; bit >= 0; --bit) {
                        const std::uint64_t bbit = (prod.low >> bit) & 1ULL;
                        const bool carry         = (rem >> 63) != 0ULL;
                        rem                      = (rem << 1) | bbit;
                        if (carry || rem >= weight_sum) {
                            rem -= weight_sum;
                            q |= (1ULL << bit);
                        }
                    }
                    next = (q / psize) * psize;
                }
                if (next < prev) { next = prev; }
                if (next > prev) {
                    plan.pinned_shards.push_back(HostNumaShard{
                        .offset    = static_cast<std::size_t>(prev),
                        .bytes     = static_cast<std::size_t>(next - prev),
                        .numa_node = ordered[idx]->node_id,
                    });
                    prev = next;
                }
            }
        }
    }
    return plan;
}

std::vector<int> select_worker_cpus(int preferred_node, std::uint32_t requested_threads) {
    const NumaTopology& topo = host_numa_topology();
    if (topo.nodes.empty()) { return {}; }
    const NumaNodeInfo* primary = (preferred_node >= 0) ? topo.find(preferred_node) : nullptr;
    if (!primary) { primary = &topo.nodes.front(); }

    const auto append_ordered = [](const NumaNodeInfo& n, std::vector<int>& dst) {
        const auto& phys = !n.physical_cpus.empty() ? n.physical_cpus : n.logical_cpus;
        for (const int cpu : phys) {
            if (std::find(dst.begin(), dst.end(), cpu) == dst.end()) { dst.push_back(cpu); }
        }
        for (const int cpu : n.logical_cpus) {
            if (std::find(dst.begin(), dst.end(), cpu) == dst.end()) { dst.push_back(cpu); }
        }
    };

    std::vector<int> cpus;
    if (requested_threads == 0) {
        const auto& phys =
            !primary->physical_cpus.empty() ? primary->physical_cpus : primary->logical_cpus;
        cpus.assign(phys.begin(), phys.end());
        return cpus;
    }
    append_ordered(*primary, cpus);
    if (requested_threads > cpus.size() && topo.multi_node()) {
        for (const auto& n : topo.nodes) {
            if (n.node_id == primary->node_id) { continue; }
            const auto& phys = !n.physical_cpus.empty() ? n.physical_cpus : n.logical_cpus;
            for (const int cpu : phys) {
                if (std::find(cpus.begin(), cpus.end(), cpu) == cpus.end()) { cpus.push_back(cpu); }
            }
        }
        for (const auto& n : topo.nodes) {
            if (n.node_id == primary->node_id) { continue; }
            for (const int cpu : n.logical_cpus) {
                if (std::find(cpus.begin(), cpus.end(), cpu) == cpus.end()) { cpus.push_back(cpu); }
            }
        }
    }
    if (cpus.size() > requested_threads) { cpus.resize(requested_threads); }
    return cpus;
}

std::string format_numa_summary(const NumaTopology& topo) {
    std::ostringstream os;
    os << topo.nodes.size() << " node(s)";
    for (std::size_t i = 0; i < topo.nodes.size(); ++i) {
        const auto& n = topo.nodes[i];
        os << (i == 0 ? ": " : "; ") << "node " << n.node_id << " ("
           << n.physical_cpus.size() << " phys/" << n.logical_cpus.size() << " log CPUs";
        if (n.total_bytes > 0) {
            const double free_gib  = double(n.free_bytes) / double(1ULL << 30);
            const double total_gib = double(n.total_bytes) / double(1ULL << 30);
            os << ", " << static_cast<std::uint64_t>(free_gib + 0.5) << "/"
               << static_cast<std::uint64_t>(total_gib + 0.5) << " GiB avail";
        }
        os << ")";
    }
    return os.str();
}

} // namespace ninfer
