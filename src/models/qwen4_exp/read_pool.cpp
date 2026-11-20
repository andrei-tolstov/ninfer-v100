#include "models/qwen4_exp/read_pool.h"

#include "core/numa.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen4_exp {

ReadPool::ReadPool(std::size_t threads, std::span<const int> cpu_ids) {
    workers_.reserve(threads);
    for (std::size_t i = 0; i < threads; ++i) {
        const int cpu_id = i < cpu_ids.size() ? cpu_ids[i] : -1;
        workers_.emplace_back([this, cpu_id] { work(cpu_id); });
    }
}

ReadPool::~ReadPool() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    ready_.notify_all();
    for (auto& worker : workers_) { worker.join(); }
}

void ReadPool::start(std::size_t count, std::function<void(std::size_t)> read) {
    {
        std::lock_guard lock(mutex_);
        if (remaining_ != 0 || job_) {
            throw std::logic_error("ReadPool: the previous batch has not finished");
        }
        if (count == 0) { return; }
        job_       = std::move(read);
        count_     = count;
        next_      = 0;
        remaining_ = count;
        ++batch_;
    }
    ready_.notify_all();
}

void ReadPool::finish() {
    drain();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [&] { return remaining_ == 0; });
    job_ = nullptr;
    if (failure_) { std::rethrow_exception(std::exchange(failure_, nullptr)); }
}

void ReadPool::run(std::size_t count, std::function<void(std::size_t)> read) {
    start(count, std::move(read));
    finish();
}

// Takes the batch's next index until none is left. The job stays in place until finish() has
// seen every index done.
void ReadPool::drain() {
    for (;;) {
        std::size_t index = 0;
        {
            std::lock_guard lock(mutex_);
            if (!job_ || next_ >= count_) { return; }
            index = next_++;
        }
        std::exception_ptr failure;
        try {
            job_(index);
        } catch (...) { failure = std::current_exception(); }
        std::lock_guard lock(mutex_);
        if (failure && !failure_) { failure_ = failure; }
        if (--remaining_ == 0) { done_.notify_all(); }
    }
}

void ReadPool::work(int cpu_id) {
    if (cpu_id >= 0) { bind_thread_to_cpu(cpu_id); }
    std::uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [&] { return stop_ || (batch_ != seen && job_ && next_ < count_); });
            if (stop_) { return; }
            seen = batch_;
        }
        drain();
    }
}

} // namespace ninfer::models::qwen4_exp
