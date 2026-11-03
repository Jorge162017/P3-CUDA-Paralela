/*
 * Basado en el ejemplo de G. Barlas (diciembre 2014), licencia GNU GPL 3.0.
 * Fases 3 y 4: Hough con memoria global y constante, medida con CUDA Events.
 * Las tablas de cosenos y senos pasan a memoria constante; la imagen y el
 * acumulador siguen en memoria global. El protocolo de medición es el mismo
 * de la versión global para que los tiempos sean comparables.
 */
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
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

// Ambas versiones comparten las mismas tablas y el mismo orden aritmético.
// Esto evita que diferencias trigonométricas cambien el bin de una votación.
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
            // El extremo positivo y el redondeo pueden producir exactamente rBins.
            if (bin < 0) bin = 0;
            if (bin >= rBins) bin = rBins - 1;
            ++acc[bin * degreeBins + t];
        }
    }
    return acc;
}

#ifndef HOUGH_CPU_ONLY
// Memoria constante: se declara con alcance global para que el kernel la lea
// sin recibir punteros. Ocupa 2 * 90 * 4 = 720 bytes de los 64 KB disponibles.
// En cada iteración todos los hilos de un warp leen el mismo índice t, así que
// la caché constante entrega el valor a todo el warp en una sola lectura.
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct CudaEvent {
    cudaEvent_t value{};
    CudaEvent() { checkCuda(cudaEventCreate(&value), "crear evento"); }
    ~CudaEvent() {
        const cudaError_t status = cudaEventDestroy(value);
        if (status != cudaSuccess)
            std::fprintf(stderr, "cudaEventDestroy: %s\n", cudaGetErrorString(status));
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};

// RAII: libera cada asignación incluso cuando se lanza una excepción.
// Se declara antes del resto de recursos GPU para reiniciar el dispositivo al final.
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

// Las tablas ya no son argumentos: el kernel usa directamente d_Cos y d_Sin.
__global__ void GPU_HoughTranConst(const unsigned char* pic, int w, int h,
                                  int* acc, float rMax, float rScale) {
    const unsigned int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= static_cast<unsigned int>(w * h)) return;
    // % recupera la columna; / recupera la fila. Se traslada el origen al centro
    // y se invierte Y porque las filas de una imagen crecen hacia abajo.
    const int xCoord = static_cast<int>(gloID % w) - w / 2;
    const int yCoord = h / 2 - static_cast<int>(gloID / w);
    if (pic[gloID] == 0) return;
    for (int t = 0; t < degreeBins; ++t) {
        const float r = xCoord * d_Cos[t] + yCoord * d_Sin[t];
        int bin = static_cast<int>((r + rMax) / rScale);
        if (bin < 0) bin = 0;
        if (bin >= rBins) bin = rBins - 1;
        // Distintos píxeles votan por la misma línea: atomicAdd evita perder votos.
        atomicAdd(acc + bin * degreeBins + t, 1);
    }
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Uso: %s imagen.pgm [--runs N] [--output directorio]\n", argv[0]);
        return 1;
    }
    try {
        int runs = 1;
        std::string outputDirectory;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("Falta valor para " + option);
            const std::string value = argv[++i];
            if (option == "--runs") {
                std::size_t used = 0;
                runs = std::stoi(value, &used);
                if (used != value.size() || runs < 1 || runs > 10000)
                    throw std::runtime_error("--runs debe estar entre 1 y 10000");
            } else if (option == "--output") {
                if (value.empty()) throw std::runtime_error("Directorio de salida vacio");
                outputDirectory = value;
            } else throw std::runtime_error("Opcion desconocida: " + option);
        }
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
        std::printf("Imagen: %d x %d; acumulador: %d bins\n", w, h, accumulatorSize);
#ifdef HOUGH_CPU_ONLY
        long long votes = 0, active = 0;
        for (int value : cpu) votes += value;
        for (unsigned char pixel : image.pixels) active += pixel > 0;
        if (votes != active * degreeBins) throw std::runtime_error("Conteo CPU incorrecto");
        if (runs != 1) throw std::runtime_error("Las mediciones requieren compilar con CUDA");
        if (!outputDirectory.empty()) throw std::runtime_error("--output requiere compilar con CUDA");
        std::printf("CPU OK: %lld pixeles activos, %lld votos. GPU NO ejecutada.\n", active, votes);
#else
        if (!outputDirectory.empty()) std::filesystem::create_directories(outputDirectory);
        DeviceSession session;
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::printf("GPU: %s\n", properties.name);
        std::puts("Variante: memoria global + constante");
        DeviceBuffer<unsigned char> input(image.pixels.size());
        DeviceBuffer<int> acc(accumulatorSize);
        checkCuda(cudaMemcpy(input.data, image.pixels.data(), image.pixels.size(), cudaMemcpyHostToDevice), "copiar imagen");
        // Los símbolos constantes no se reservan con cudaMalloc ni se liberan con
        // cudaFree: existen mientras el módulo esté cargado y se llenan desde el host.
        checkCuda(cudaMemcpyToSymbol(d_Cos, cosines.data(), degreeBins * sizeof(float)), "copiar cosenos a memoria constante");
        checkCuda(cudaMemcpyToSymbol(d_Sin, sines.data(), degreeBins * sizeof(float)), "copiar senos a memoria constante");
        const unsigned int threads = 256;
        const unsigned int blocks = (image.pixels.size() + threads - 1) / threads;
        auto launch = [&]() {
            GPU_HoughTranConst<<<blocks, threads>>>(input.data, w, h, acc.data, rMax, rScale);
            checkCuda(cudaGetLastError(), "lanzar kernel");
        };
        // Una ejecución de calentamiento, excluida de las mediciones.
        checkCuda(cudaMemset(acc.data, 0, accumulatorSize * sizeof(int)), "limpiar calentamiento");
        launch();
        checkCuda(cudaDeviceSynchronize(), "calentamiento");
        CudaEvent start, stop;
        std::vector<float> times;
        std::vector<int> gpu(accumulatorSize);
        for (int run = 1; run <= runs; ++run) {
            // Mismo stream: el borrado termina antes de la marca de inicio.
            // Copias, memset, validación y exportación no entran en el tiempo.
            checkCuda(cudaMemset(acc.data, 0, accumulatorSize * sizeof(int)), "reiniciar acumulador");
            checkCuda(cudaEventRecord(start.value), "registrar inicio");
            launch();
            checkCuda(cudaEventRecord(stop.value), "registrar fin");
            checkCuda(cudaEventSynchronize(stop.value), "esperar fin");
            float milliseconds = 0;
            checkCuda(cudaEventElapsedTime(&milliseconds, start.value, stop.value), "medir kernel");
            checkCuda(cudaMemcpy(gpu.data(), acc.data, accumulatorSize * sizeof(int), cudaMemcpyDeviceToHost), "copiar acumulador");
            int mismatches = 0;
            for (int i = 0; i < accumulatorSize; ++i) {
                if (cpu[i] != gpu[i]) {
                    if (mismatches < 5)
                        std::fprintf(stderr, "Bin %d: CPU=%d GPU=%d\n", i, cpu[i], gpu[i]);
                    ++mismatches;
                }
            }
            if (mismatches) throw std::runtime_error(std::to_string(mismatches) + " bins distintos entre CPU y GPU");
            times.push_back(milliseconds);
            std::printf("Ejecucion %02d: %.6f ms; CPU/GPU OK\n", run, milliseconds);
        }
        std::printf("Bloques: %u; hilos por bloque: %u; calentamientos: 1\n", blocks, threads);
        const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::printf("Promedio del kernel (%d ejecuciones): %.6f ms\n", runs, mean);
        if (!outputDirectory.empty()) {
            int runtime = 0, driver = 0;
            checkCuda(cudaRuntimeGetVersion(&runtime), "version runtime");
            checkCuda(cudaDriverGetVersion(&driver), "version driver");
            // Mismas columnas que global-times.csv para comparar ambas bitácoras.
            std::ofstream csv(outputDirectory + "/constante-times.csv");
            csv.exceptions(std::ios::failbit | std::ios::badbit);
            csv << "variant,run,kernel_ms,gpu,cuda_runtime,cuda_driver,width,height,blocks,threads,warmup_runs,validated\n";
            for (std::size_t i = 0; i < times.size(); ++i)
                csv << "constante," << i + 1 << ',' << std::fixed << std::setprecision(6) << times[i]
                    << ',' << std::quoted(properties.name) << ',' << runtime << ',' << driver
                    << ',' << w << ',' << h << ',' << blocks << ',' << threads << ",1,true\n";
            csv.close();
        }
        std::puts("VALIDACION OK: CPU y GPU coinciden en todos los bins.");
#endif
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
