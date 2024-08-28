#pragma once

#include <chrono>
#include <cstdio>

class Timer {
   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    bool running;
    double elapsed_time_second;

   public:
    Timer() = default;
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        // elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        elapsed_time_second = duration * 1e-9;
        running = false;
    }
    double getElapsedTimeSecond() {
        if (running) {
            stop();
        }
        return elapsed_time_second;
    }
};

inline double Tflops(size_t num, double time_second) { return num / (time_second * 1e12); }

// inline void timerEvent (const char* msg, ) {

// }

// clang-format off
// #define DEBUG
#ifdef DEBUG
#define TIMER_EVENT(cmd, num_op, msg)                                                    \
    do {                                                                                 \
        Timer timer;                                                                     \
        timer.start();                                                                   \
        cmd;                                                                             \
        timer.stop();                                                                    \
        printf("%s: %lf second\n", msg, timer.getElapsedTimeSecond());                   \
        printf("%s: Tflops = %lf\n", msg, Tflops(num_op, timer.getElapsedTimeSecond())); \
    } while (0)

#else
#define TIMER_EVENT(cmd, num_op, msg) \
    do {                              \
        cmd;                          \
    } while (0)

#endif
// clang-format on
// printf("%s: Tflops = %lf\n", msg, timer.getElapsedTimeSecond() / (1e12));