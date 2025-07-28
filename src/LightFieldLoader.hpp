#ifndef LIGHT_FIELD_LOADER_HPP
#define LIGHT_FIELD_LOADER_HPP

#define BLOCK_SIZE 8

#include <filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class LightFieldLoader {
public:
    LightFieldLoader(const std::string& directory, int U, int V);
    ~LightFieldLoader() = default;

    bool load();
    cv::Mat getView(int u, int v) const;
    
    int getHeight() const;
    int getWidth() const;
    int getAngularRows() const;
    int getAngularCols() const;

    void getFlattenedLightField(int channel);
    std::vector<float> calculateBasisWaves(int dimSize) const;
    void calculateDctDim();
    void exportToCsv();

    std::vector<float> flattenedLf;

private:
    std::string imageDir;
    int U, V;
    int H = 0, W = 0;
    std::vector<std::vector<cv::Mat>> lightfield;
    std::vector<std::filesystem::path> listSortedPPMs() const;
    // the 13x13x16x16
    void getFlattenedSyntheticLF(std::string& csvFolder);
};

#endif