#pragma once

#include <atomic>

namespace gotst {

class CancellationToken {
public:
    void cancel() {
        cancelled_.store(true, std::memory_order_release);
    }

    bool is_cancelled() const {
        return cancelled_.load(std::memory_order_acquire);
    }

    void reset() {
        cancelled_.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> cancelled_{false};
};

} // namespace gotst
