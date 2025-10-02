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

    lf.calculateError();
    std::cout << "Calculated the error" << std::endl;

    /*
    std::ofstream out1("coefs16.csv");
    out1 << "__constant__ float BASIS16[16 * 16] = {\n";
    auto basis16 = lf.calculateBasisWaves(16);
    for (auto i = 1; i < 257; ++i) {
        out1 << basis16[i - 1] << ",";

        if (i % 60 == 0) {
            out1 << "\n";
        }
    }
    out1 << "};\n";

    std::ofstream out2("coefs13.csv");
    out2 << "__constant__ float BASIS13[13 * 13] = {\n";
    auto basis13 = lf.calculateBasisWaves(13);

    for (auto i = 1; i < 170; ++i) {
        out2 << basis13[i - 1] << ",";

        if (i % 60 == 0) {
            out2 << "\n";
        }
    }
    out2 << "};\n";
    */

    return 0;
}