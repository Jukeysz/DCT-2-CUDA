#include "LightFieldLoader.hpp"
#include "apply_dct1d.hpp"
#include <iostream>
#include <time.h>
#include <stdio.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    LightFieldLoader lf("../../reduced_data", 13, 13);

    lf.calculateDctDim();

    lf.exportToCsv();
    std::cout << "Exported to CSV" << std::endl;

    return 0;
}