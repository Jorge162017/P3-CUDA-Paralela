#pragma once
#include <vector>

// Imagen PGM de 8 bits; el vector libera los píxeles automáticamente.
class PGMImage {
public:
    int x_dim = 0;
    int y_dim = 0;
    std::vector<unsigned char> pixels;
    explicit PGMImage(const char* filename);
};
