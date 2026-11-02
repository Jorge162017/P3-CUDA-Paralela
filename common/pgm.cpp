#include "pgm.h"
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
std::string token(std::istream& input) {
    std::string value;
    while (true) {
        input >> std::ws;
        if (input.peek() != '#') break;
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    if (!(input >> value)) throw std::runtime_error("Cabecera PGM incompleta");
    return value;
}
int number(std::istream& input) {
    const std::string value = token(input);
    std::size_t used = 0;
    int result = std::stoi(value, &used);
    if (used != value.size()) throw std::runtime_error("Numero PGM invalido");
    return result;
}
}

PGMImage::PGMImage(const char* filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("No se pudo abrir la imagen PGM");
    const std::string format = token(input);
    if (format != "P5" && format != "P2")
        throw std::runtime_error("Solo se admiten PGM P2 y P5");
    x_dim = number(input);
    y_dim = number(input);
    const int maximum = number(input);
    // Se limita a INT_MAX para que los indices lineales del kernel sean seguros.
    if (x_dim <= 0 || y_dim <= 0 || maximum <= 0 || maximum > 255 ||
        x_dim > std::numeric_limits<int>::max() / y_dim)
        throw std::runtime_error("Dimensiones o profundidad PGM no admitidas");
    pixels.resize(static_cast<std::size_t>(x_dim) * y_dim);
    if (format == "P5") {
        char separator = 0;
        if (!input.get(separator) || !std::isspace(static_cast<unsigned char>(separator)))
            throw std::runtime_error("Falta separador del raster PGM");
        // No usar std::ws: un byte de pixel puede ser espacio o salto de linea.
        if (separator == '\r' && input.peek() == '\n') input.get();
        input.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
        if (input.gcount() != static_cast<std::streamsize>(pixels.size()))
            throw std::runtime_error("Datos PGM incompletos");
        for (unsigned char pixel : pixels)
            if (pixel > maximum) throw std::runtime_error("Pixel fuera del rango PGM");
    } else {
        for (unsigned char& pixel : pixels) {
            const int value = number(input);
            if (value < 0 || value > maximum)
                throw std::runtime_error("Pixel fuera del rango PGM");
            pixel = static_cast<unsigned char>(value);
        }
    }
}
