#include <cassert>

#include "qcu_macro.h"
#include "qcu_utils.h"

namespace qcu {

static bool gpu_id_set = false;
static int gpu_id = -1;

static bool process_rank_set = false;
static int process_rank = -1;

}  // namespace qcu

int get_gpu_id() {
    if (!qcu::gpu_id_set) {
        errorQcu("GPU ID is not set\n");
    }
    return qcu::gpu_id;
}
int get_process_id() {
    if (!qcu::process_rank_set) {
        errorQcu("Process rank is not set\n");
    }
    return qcu::process_rank;
}