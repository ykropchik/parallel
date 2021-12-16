#include <thread>
#include "omp.h"

static unsigned g_num_thread = std::thread::hardware_concurrency();

void set_num_threads(unsigned T) {
    g_num_thread = T;
    omp_set_num_threads(T);
}

unsigned get_num_threads() {
    return g_num_thread;
}

