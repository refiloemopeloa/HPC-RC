#include "../helpers/mnist_loader.h"
#include "../helpers/image_utils.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;


int preprocessData(int num_workers) {
    auto train_images = MNISTLoader::loadImages("../data/train-images.idx3-ubyte");
    auto train_labels = MNISTLoader::loadLabels("../data/train-labels.idx1-ubyte");

    std::cout << "Loaded " << train_images.size() / (28 * 28) << " training images" << std::endl;

    const int total_samples = train_images.size() / (28 * 28);
    const int samples_per_worker = total_samples / num_workers;

    std::cout << "Creating " << num_workers << " worker datasets with "
              << samples_per_worker << " samples each" << std::endl;

    // Create non-IID distributions
    for (int worker = 1; worker <= num_workers; worker++) {
        std::vector<float> worker_images;
        std::vector<uint8_t> worker_labels;
        float rotation = 90.0f * worker;

        std::cout << "Processing worker " << worker << " with rotation " << rotation << "°" << std::endl;

        for (int i = 0; i < samples_per_worker; i++) {
            int idx = (worker - 1) * samples_per_worker + i;

            std::vector<uint8_t> img_uint8(28 * 28);
            std::copy(train_images.begin() + idx * 28 * 28,
                      train_images.begin() + (idx + 1) * 28 * 28,
                      img_uint8.begin());

            ImageUtils::rotate(img_uint8, rotation);

            for (uint8_t pixel : img_uint8) {
                worker_images.push_back(pixel / 255.0f);
            }

            worker_labels.push_back(train_labels[idx]);
        }

        std::string image_filename = "worker_" + std::to_string(worker) + "_images.bin";
        std::ofstream img_file(image_filename, std::ios::binary);
        if (!img_file) {
            std::cerr << "Error: Cannot create " << image_filename << std::endl;
            return 1;
        }
        img_file.write(reinterpret_cast<const char*>(worker_images.data()),
                       worker_images.size() * sizeof(float));
        img_file.close();

        std::string label_filename = "worker_" + std::to_string(worker) + "_labels.bin";
        std::ofstream label_file(label_filename, std::ios::binary);
        if (!label_file) {
            std::cerr << "Error: Cannot create " << label_filename << std::endl;
            return 1;
        }
        label_file.write(reinterpret_cast<const char*>(worker_labels.data()),
                         worker_labels.size());
        label_file.close();

        std::cout << "Worker " << worker << ": Saved " << worker_images.size()
                  << " float values (" << worker_images.size() / 784 << " images)" << std::endl;

        float min_val = *std::min_element(worker_images.begin(), worker_images.end());
        float max_val = *std::max_element(worker_images.begin(), worker_images.end());
        std::cout << "Worker " << worker << " data range: [" << min_val << ", " << max_val << "]" << std::endl;
    }

    // === Process Test Set ===
    std::cout << "Processing test set..." << std::endl;

    auto test_images_u8 = MNISTLoader::loadImages("../data/t10k-images.idx3-ubyte");
    auto test_labels = MNISTLoader::loadLabels("../data/t10k-labels.idx1-ubyte");

    size_t num_test_samples = test_labels.size();
    const int image_size = 28 * 28;

    std::vector<float> normalized_test_images;
    normalized_test_images.reserve(num_test_samples * image_size);

    for (size_t i = 0; i < num_test_samples; ++i) {
        for (int j = 0; j < image_size; ++j) {
            normalized_test_images.push_back(test_images_u8[i * image_size + j] / 255.0f);
        }
    }

    std::ofstream test_img_file("test_images.bin", std::ios::binary);
    if (!test_img_file) {
        std::cerr << "Error: Cannot create test_images.bin" << std::endl;
        return 1;
    }
    test_img_file.write(reinterpret_cast<const char*>(normalized_test_images.data()),
                        normalized_test_images.size() * sizeof(float));
    test_img_file.close();

    std::ofstream test_label_file("test_labels.bin", std::ios::binary);
    if (!test_label_file) {
        std::cerr << "Error: Cannot create test_labels.bin" << std::endl;
        return 1;
    }
    test_label_file.write(reinterpret_cast<const char*>(test_labels.data()),
                          test_labels.size());
    test_label_file.close();

    std::cout << "Saved test set: " << num_test_samples << " images and labels." << std::endl;

    std::cout << "Preprocessing complete. Created " << num_workers << " worker datasets." << std::endl;
    std::cout << "All images are normalized to [0.0, 1.0] range and saved as float values." << std::endl;

    return 0;
}


/*int main() {
    // Load original MNIST
    int m;
    cin >> m;
    kmeans(m);
    return 0;
}*/