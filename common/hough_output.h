#pragma once
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "pgm.h"

// El acumulador exportado permite dibujar sin volver a ejecutar el kernel.
// Se conservan las tablas exactas usadas durante la votación.
inline void exportHoughAccumulator(const std::string& directory, const PGMImage& image,
                       const std::vector<int>& acc, float rMax, float rScale,
                       const std::vector<float>& cosines, const std::vector<float>& sines,
                       bool gpuExecuted, const std::string& variant,
                       const std::string& filename, int degreeBins, int rBins) {
    std::filesystem::create_directories(directory);
    std::ofstream out(directory + "/" + filename);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(9)
        << "{\n\"variant\": " << std::quoted(variant)
        << ",\n\"gpu_executed\": " << (gpuExecuted ? "true" : "false")
        << ",\n\"width\": " << image.x_dim << ",\n\"height\": " << image.y_dim
        << ",\n\"degree_bins\": " << degreeBins << ",\n\"r_bins\": " << rBins
        << ",\n\"r_max\": " << rMax << ",\n\"r_scale\": " << rScale;
    auto writeArray = [&](const char* name, const auto& values) {
        out << ",\n" << std::quoted(name) << ": [";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) out << ',';
            out << +values[i];
        }
        out << ']';
    };
    writeArray("cosines", cosines);
    writeArray("sines", sines);
    writeArray("votes", acc);
    out << "\n}\n";
    out.close();
}
