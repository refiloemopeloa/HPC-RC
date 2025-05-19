#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint> 
#include <cmath>

class ImageUtils {
public:
    static void rotate(std::vector<uint8_t>& image, float degrees) {
        const int size = 28;
        std::vector<uint8_t> rotated(size*size, 0);
        float rad = degrees * M_PI / 180.0f;
        int center = size / 2;

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int xp = (x - center) * cos(rad) - (y - center) * sin(rad) + center;
                int yp = (x - center) * sin(rad) + (y - center) * cos(rad) + center;
                
                if (xp >= 0 && xp < size && yp >= 0 && yp < size) {
                    rotated[y*size + x] = image[yp*size + xp];
                }
            }
        }
        image = rotated;
    }

    static void saveBinary(const std::vector<uint8_t>& data, const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
};