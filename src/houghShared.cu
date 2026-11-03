
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/pgm.h"
#ifndef HOUGH_CPU_ONLY
#include <cuda_runtime.h>
#endif

constexpr int degreeInc = 2;
constexpr int degreeBins = 180 / degreeInc;
constexpr int rBins = 100;
constexpr int accumulatorSize = degreeBins * rBins;
constexpr float radInc = degreeInc * 3.14159265358979323846 / 180;
constexpr std::size_t sharedBytesPerBlock = accumulatorSize * sizeof(int);

std::vector<int> CPU_HoughTran(const PGMImage& image, float rMax,
                             float rScale, const float* cosines, const float* sines) {
    std::vector<int> acc(accumulatorSize, 0);
    for (int id = 0; id < static_cast<int>(image.pixels.size()); ++id) {
        if (image.pixels[id] == 0) continue;
        const int x = id % image.x_dim - image.x_dim / 2;
        const int y = image.y_dim / 2 - id / image.x_dim;
        for (int t = 0; t < degreeBins; ++t) {
            const float r = x * cosines[t] + y * sines[t];
            int bin = static_cast<int>((r + rMax) / rScale);
            if (bin < 0) bin = 0;
            if (bin >= rBins) bin = rBins - 1;
            ++acc[bin * degreeBins + t];
        }
    }
    return acc;
}

#ifndef HOUGH_CPU_ONLY
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct DeviceSession {
    ~DeviceSession() {
        const cudaError_t status = cudaDeviceReset();
        if (status != cudaSuccess)
            std::fprintf(stderr, "cudaDeviceReset: %s\n", cudaGetErrorString(status));
    }
};
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (data) {
            const cudaError_t status = cudaFree(data);
            if (status != cudaSuccess)
                std::fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(status));
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void GPU_HoughTranShared(const unsigned char* pic, int w, int h,
                                   int* acc, float rMax, float rScale,
                                   const float* cosines, const float* sines) {
    __shared__ int localAcc[accumulatorSize];
    const unsigned int locID = threadIdx.x;
    const unsigned int gloID = blockIdx.x * blockDim.x + locID;
    const unsigned int totalPixels = static_cast<unsigned int>(w) * static_cast<unsigned int>(h);

    for (int i = locID; i < accumulatorSize; i += blockDim.x)
        localAcc[i] = 0;
    __syncthreads();

    if (gloID < totalPixels) {
        const int xCoord = static_cast<int>(gloID % w) - w / 2;
        const int yCoord = h / 2 - static_cast<int>(gloID / w);
        if (pic[gloID] != 0) {
            for (int t = 0; t < degreeBins; ++t) {
                const float r = xCoord * cosines[t] + yCoord * sines[t];
                int bin = static_cast<int>((r + rMax) / rScale);
                if (bin < 0) bin = 0;
                if (bin >= rBins) bin = rBins - 1;
                atomicAdd(&localAcc[bin * degreeBins + t], 1);
            }
        }
    }
    __syncthreads();

    for (int i = locID; i < accumulatorSize; i += blockDim.x) {
        const int votes = localAcc[i];
        if (votes > 0) atomicAdd(acc + i, votes);
    }
}
#endif

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Uso: %s imagen.pgm\n", argv[0]);
        return 1;
    }
    try {
        const PGMImage image(argv[1]);
        const int w = image.x_dim, h = image.y_dim;
        const float rMax = std::sqrt(1.0 * w * w + 1.0 * h * h) / 2;
        const float rScale = 2 * rMax / rBins;
        std::vector<float> cosines(degreeBins), sines(degreeBins);
        float angle = 0;
        for (int t = 0; t < degreeBins; ++t, angle += radInc) {
            cosines[t] = std::cos(angle);
            sines[t] = std::sin(angle);
        }
        const auto cpu = CPU_HoughTran(image, rMax, rScale, cosines.data(), sines.data());
        std::printf("Variante: global + compartida (tablas aun en memoria global)\n");
        std::printf("Imagen: %d x %d; acumulador: %d bins\n", w, h, accumulatorSize);
#ifdef HOUGH_CPU_ONLY
        long long votes = 0, active = 0;
        for (int value : cpu) votes += value;
        for (unsigned char pixel : image.pixels) active += pixel > 0;
        if (votes != active * degreeBins) throw std::runtime_error("Conteo CPU incorrecto");
        std::printf("CPU OK: %lld pixeles activos, %lld votos. GPU NO ejecutada.\n", active, votes);
#else
        DeviceSession session;
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::printf("GPU: %s\n", properties.name);
        std::printf("Memoria compartida por bloque: requerida %zu bytes, limite GPU %zu bytes\n",
                    sharedBytesPerBlock, static_cast<std::size_t>(properties.sharedMemPerBlock));
        if (properties.sharedMemPerBlock < sharedBytesPerBlock)
            throw std::runtime_error("la GPU no ofrece suficiente memoria compartida por bloque");
        DeviceBuffer<unsigned char> input(image.pixels.size());
        DeviceBuffer<int> acc(accumulatorSize);
        DeviceBuffer<float> dCos(degreeBins), dSin(degreeBins);
        checkCuda(cudaMemcpy(input.data, image.pixels.data(), image.pixels.size(), cudaMemcpyHostToDevice), "copiar imagen");
        checkCuda(cudaMemcpy(dCos.data, cosines.data(), degreeBins * sizeof(float), cudaMemcpyHostToDevice), "copiar cosenos");
        checkCuda(cudaMemcpy(dSin.data, sines.data(), degreeBins * sizeof(float), cudaMemcpyHostToDevice), "copiar senos");
        checkCuda(cudaMemset(acc.data, 0, accumulatorSize * sizeof(int)), "inicializar acumulador");
        const unsigned int threads = 256;
        const unsigned int blocks = (image.pixels.size() + threads - 1) / threads;
        GPU_HoughTranShared<<<blocks, threads>>>(input.data, w, h, acc.data, rMax, rScale, dCos.data, dSin.data);
        checkCuda(cudaGetLastError(), "lanzar kernel");
        checkCuda(cudaDeviceSynchronize(), "ejecutar kernel");
        std::vector<int> gpu(accumulatorSize);
        checkCuda(cudaMemcpy(gpu.data(), acc.data, accumulatorSize * sizeof(int), cudaMemcpyDeviceToHost), "copiar acumulador");
        int mismatches = 0;
        for (int i = 0; i < accumulatorSize; ++i) {
            if (cpu[i] != gpu[i]) {
                if (mismatches < 5)
                    std::fprintf(stderr, "Bin %d: CPU=%d GPU=%d\n", i, cpu[i], gpu[i]);
                ++mismatches;
            }
        }
        std::printf("Bloques: %u; hilos por bloque: %u; pixeles en ultimo bloque: %zu\n",
                    blocks, threads, image.pixels.size() - static_cast<std::size_t>(blocks - 1) * threads);
        if (mismatches) throw std::runtime_error(std::to_string(mismatches) + " bins distintos entre CPU y GPU");
        std::puts("VALIDACION OK: CPU y GPU coinciden en todos los bins.");
#endif
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
