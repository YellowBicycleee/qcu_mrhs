#pragma once
#include <cublas_v2.h>
#include "qcu_public.h"

#define QCU_DEBUG
#ifdef QCU_DEBUG
#define QCU_CHECK_CUBLAS(cmd)             \
do {                                      \
  cublasStatus_t stat = cmd;              \
  check_cublas (stat, __FILE__, __LINE__);\
} while (0)
#else
#define QCU_CHECK_CUBLAS(cmd) \
do {                          \
  cmd;                        \
} while (0)

#endif

inline void check_cublas (cublasStatus_t stat, const char* file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("IN file %s, line %d, error happened\n", file, line);
    abort();
  }
}
constexpr int maxThreadsPerBlock = MAX_THREADS_PER_BLOCK;
constexpr int maxGridSize        = {2147483647};