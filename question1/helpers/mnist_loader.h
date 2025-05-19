#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>

class MNISTLoader {
public:
    static std::vector<uint8_t> loadImages(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        // Read magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), 4);
        magic = __builtin_bswap32(magic);
        
        if (magic != 2051) {
            throw std::runtime_error("Invalid MNIST image file format. Expected 2051, got " + 
                                   std::to_string(magic));
        }

        // Read header
        uint32_t num_images, rows, cols;
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);
        
        num_images = __builtin_bswap32(num_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        // Read image data
        std::vector<uint8_t> images(num_images * rows * cols);
        file.read(reinterpret_cast<char*>(images.data()), images.size());
        
        return images;
    }

    static std::vector<uint8_t> loadLabels(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        // Read magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), 4);
        magic = __builtin_bswap32(magic);
        
        if (magic != 2049) {
            throw std::runtime_error("Invalid MNIST label file format. Expected 2049, got " + 
                                   std::to_string(magic));
        }

        // Read header
        uint32_t num_labels;
        file.read(reinterpret_cast<char*>(&num_labels), 4);
        num_labels = __builtin_bswap32(num_labels);

        // Read label data
        std::vector<uint8_t> labels(num_labels);
        file.read(reinterpret_cast<char*>(labels.data()), labels.size());
        
        return labels;
    }
};