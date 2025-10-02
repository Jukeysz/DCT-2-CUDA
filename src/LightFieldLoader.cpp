#include "LightFieldLoader.hpp"
#include <iostream>
#include <algorithm>
#include "apply_dct1d.hpp"
#include <fstream>
#include <vector>
#include <ranges>

#define SYNTH_SAMPLE 43264

namespace fs = std::filesystem;

LightFieldLoader::LightFieldLoader(const std::string& directory, int U, int V)
    : imageDir(directory), U(U), V(V), lightfield(U, std::vector<cv::Mat>(V)) {}

// get a sorted list of the absolute paths of each of the samples
std::vector<fs::path> LightFieldLoader::listSortedPPMs() const {
    std::vector<fs::path> ppm_files;
    // take contents from path "imageDir"
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        if (entry.path().extension() == ".ppm") {
            ppm_files.push_back(entry.path());
        }
    }

    std::sort(ppm_files.begin(), ppm_files.end());

    for (const auto& file : ppm_files) {
        std::cout << "Sorted PPM file: " << file << std::endl;
    }
    return ppm_files;
}

bool LightFieldLoader::load() {
    auto ppm_files = listSortedPPMs();

    if (ppm_files.size() != static_cast<size_t>(U * V)) {
        std::cerr << "Error: Expected " << U * V << " views but found " << ppm_files.size() << std::endl;
        return false;
    }

    for (size_t idx = 0; idx < ppm_files.size(); ++idx) {
        // placing the idxes in the correct places in the M x N hyperparameters
        int u = idx / V;
        int v = idx % V;

        cv::Mat image = cv::imread(ppm_files[idx].string(), cv::IMREAD_COLOR);

        if (image.empty()) {
            std::cerr << "Error: Failed to load " << ppm_files[idx] << std::endl;
            return false;
        }

        if (H == 0 && W == 0) {
            H = image.rows;
            W = image.cols;
        }

        lightfield[u][v] = image;
        std::cout << "Placed image onto lightfield view (" << u << ", " << v << ")" << std::endl;
        std::cout << "Image type: " << image.type() << std::endl;
        std::cout << "Channels: " << image.channels() << std::endl;

    }

    std::cout << "Finished Loading" << std::endl;
    return true;
}

cv::Mat LightFieldLoader::getView(int u, int v) const {
    if (u < 0 || u >= U || v < 0 || v >= V) {
        throw std::out_of_range("Invalid (u, v) view access");
    }
    return lightfield[u][v];
}

void LightFieldLoader::getFlattenedSyntheticLF(std::string& imageDir) {
    std::ifstream infile(imageDir);
    if (!infile.is_open()) {
        std::cerr << "Could not open Synthetic LF" << std::endl;
    }
    std::string line;

    std::getline(infile, line);
    this->U = 16;
    this->V = 16;

    this->H = 13;
    this->W = 13;
    this->flattenedLf.assign(SYNTH_SAMPLE, 0.0f);
    size_t offset = 0;
    
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string token;
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, token, ',');
        }
        std::getline(ss, token, ',');
        double value = std::stod(token);
        flattenedLf[offset++] = value;

    }

    infile.close();
}

void LightFieldLoader::getFlattenedLightField(int channel) {
    if (lightfield.empty()) return;

    // size_t totalSize = U * V * H * W * 1; // single rgb channel
    constexpr size_t totalSize = 8 * 8 * 8 * 8;
    // I gotta introduce a way to get the channels out of the global RGBRGB... array
    this->flattenedLf.assign(totalSize, 0.0f);
    size_t offset = 0;

    // iter over the MATs by catching the exact positions
    //for (int u = 0; u < BLOCK_SIZE; ++u) {
    //    for (int v = 0; v < BLOCK_SIZE; ++v) {
    //        const cv::Mat& img = lightfield[u][v];
    //
    //        CV_Assert(img.channels() == 3 && img.type() == CV_8UC3);
    //
    //        for (int y = 0; y < BLOCK_SIZE; ++y) {
    //            for (int x = 0; x < BLOCK_SIZE; ++x) {
    //                const cv::Vec3b& pixel = img.at<cv::Vec3b>(y, x);
    //                float val = 0.0f;
    //                switch (channel) {
    //                    case 0: val = static_cast<float>(pixel[2]); break;
    //                    case 1: val = static_cast<float>(pixel[1]); break;
    //                    case 2: val = static_cast<float>(pixel[0]); break;
    //                }
    //
    //                this->flattenedLf[offset++] = val;
    //            }
    //        }
    //    }
    //}

    // I'm rather taking the first sample
    const cv::Mat& img = lightfield[0][0];

    CV_Assert(img.channels() == 3 && img.type() == CV_8UC3);

    for (int s = 0; s < BLOCK_SIZE; ++s) {
        for (int t = 0; t < BLOCK_SIZE; ++t) {
            const cv::Vec3b& pixel = img.at<cv::Vec3b>(s, t);

            float val = static_cast<float>(pixel[2]);
            this->flattenedLf[offset++] = val;
        }
    }
}

std::vector<double> LightFieldLoader::calculateBasisWaves(int dimSize) const {
    std::vector<double> flattened_vector(dimSize * dimSize, 0);
    const double DCT_PI = 3.141592653589793;
    if (dimSize == 0) return flattened_vector;

    for (auto i = 0; i < dimSize; ++i) {
        flattened_vector[i] = 1.0;
    }
    for (auto i = 1; i < dimSize; ++i) {
        for (auto j = 0; j < dimSize; ++j) {
            double angle = DCT_PI * (i * (2.0 * j + 1.0)) / (2.0 * dimSize);
            flattened_vector[i * dimSize + j] = std::sqrt(2.0) * std::cos(angle);
        }
    }

    return flattened_vector;
}

void LightFieldLoader::calculateDctDim() {
    std::string synth_folder = "../../Danger_test/initial_Danger_test.csv";
    getFlattenedSyntheticLF(synth_folder);


    for (int i = 0; i < 4; ++i) {
        apply_dct1d_gpu(flattenedLf.data(), U, V, H, W, i);
    }
}

void LightFieldLoader::exportToCsv() {
    std::cout << "About to export to csv!\n";

    std::ofstream outfile("four_passes.csv");
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);

    if (!outfile.is_open()) {
        std::cerr << "It was not possible to open the file into ofstream\n";
    }

    outfile << "i,j,k,l,value\n";

    for (int i = 0; i < U; ++i) {
        for (int j = 0; j < V; ++j) {
            for (int k = 0; k < H; ++k) {
                for (int l = 0; l < W; ++l) {
                    int ind = i * V * H * W + j * H * W + k * W + l;
                    outfile << i << ","
                           << j << ","
                           << k << ","
                           << l << ","
                           << flattenedLf[ind] << "\n";
                }
            }
        }
    }

    outfile.close();
}

void LightFieldLoader::calculateError() {
    std::ifstream file("../../Danger_test/final_dct.csv");
    if (!file.is_open()) {
        std::cerr << "Could not open the final_dct.csv file" << std::endl;
        return;
    }
    std::vector<double> cpu_results;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        double value = 0.0;

        while (std::getline(ss, cell, ',')) {
            if (col == 4) {
                try {
                    value = std::stod(cell);
                    cpu_results.push_back(value);
                } catch (const std::invalid_argument&) {
                    // skip the row
                }
                break;
            }
            ++col;
        }
    }

    std::ifstream file1("./four_passes.csv");
    if (!file1.is_open()) {
        std::cerr << "Could not open the four_passes.csv file" << std::endl;
        return;
    }
    std::vector<double> gpu_results;
    std::string line1;

    while (std::getline(file1, line1)) {
        std::stringstream ss(line1);
        std::string cell;
        int col = 0;
        double value = 0.0;

        while (std::getline(ss, cell, ',')) {
            if (col == 4) {
                try {
                    value = std::stod(cell);
                    gpu_results.push_back(value);
                } catch (const std::invalid_argument&) {
                    // skip
                }
                break;
            }
            ++col;
        }
    }

    // Once I have both coefs loaded in gpu_results and cpu_results
    int size = this->U * this->V * this->H * this->W;
    std::vector<double> absolute_error;
    std::vector<double> relative_error;
    for (int i = 0; i < size; ++i) {
        absolute_error.push_back(std::abs(gpu_results[i] - cpu_results[i]));
        relative_error.push_back((std::abs(cpu_results[i] - gpu_results[i]) / std::abs(cpu_results[i])) * 100.0);
    }

    std::ofstream out0("relative_error");
    std::ofstream out1("absolute_error");

    out0 << std::setprecision(std::numeric_limits<double>::max_digits10);
    out1 << std::setprecision(std::numeric_limits<double>::max_digits10);

    for (int i = 0; i < size; ++i) {
        out0 << relative_error[i] << "\n";
        out1 << absolute_error[i] << "\n";
    }

    // find the max value from both lists
    auto [min_it, max_it] = std::ranges::minmax_element(absolute_error);
    std::cout << "Max absolute error: " << std::setprecision(std::numeric_limits<double>::max_digits10) << *max_it << "\n";

    auto [min_it2, max_it2] = std::ranges::minmax_element(relative_error);
    std::cout << "Max relative error: " << std::setprecision(std::numeric_limits<double>::max_digits10) << *max_it2 << "\n";
}

int LightFieldLoader::getHeight() const { return H; }
int LightFieldLoader::getWidth() const { return W; }
int LightFieldLoader::getAngularRows() const { return U; }
int LightFieldLoader::getAngularCols() const { return V; }