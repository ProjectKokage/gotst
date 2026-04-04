#include "gotst/core/cancellation_token.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <thread>

using namespace gotst;

TEST_CASE("CancellationToken starts as not cancelled", "[cancellation]") {
    CancellationToken token;
    CHECK_FALSE(token.is_cancelled());
}

TEST_CASE("CancellationToken is_cancelled returns true after cancel", "[cancellation]") {
    CancellationToken token;
    token.cancel();
    CHECK(token.is_cancelled());
}

TEST_CASE("CancellationToken reset restores to not cancelled", "[cancellation]") {
    CancellationToken token;
    token.cancel();
    CHECK(token.is_cancelled());
    token.reset();
    CHECK_FALSE(token.is_cancelled());
}

TEST_CASE("CancellationToken cancel is idempotent", "[cancellation]") {
    CancellationToken token;
    token.cancel();
    token.cancel();
    CHECK(token.is_cancelled());
}

TEST_CASE("CancellationToken reset on fresh token stays not cancelled", "[cancellation]") {
    CancellationToken token;
    token.reset();
    CHECK_FALSE(token.is_cancelled());
}

TEST_CASE("CancellationToken is visible across threads", "[cancellation][threads]") {
    CancellationToken token;

    std::atomic<bool> saw_cancelled{false};

    std::thread reader([&]() {
        while (!token.is_cancelled()) {
            std::this_thread::yield();
        }
        saw_cancelled.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK_FALSE(saw_cancelled.load(std::memory_order_acquire));

    token.cancel();

    reader.join();
    CHECK(saw_cancelled.load(std::memory_order_acquire));
}

TEST_CASE("CancellationToken can be reset and reused", "[cancellation]") {
    CancellationToken token;

    CHECK_FALSE(token.is_cancelled());
    token.cancel();
    CHECK(token.is_cancelled());

    token.reset();
    CHECK_FALSE(token.is_cancelled());

    token.cancel();
    CHECK(token.is_cancelled());
}

TEST_CASE("CancellationToken supports multiple reset cycles", "[cancellation]") {
    CancellationToken token;
    for (int i = 0; i < 10; ++i) {
        CHECK_FALSE(token.is_cancelled());
        token.cancel();
        CHECK(token.is_cancelled());
        token.reset();
    }
}
