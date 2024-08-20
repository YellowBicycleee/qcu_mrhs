#include <cublas_v2.h>

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