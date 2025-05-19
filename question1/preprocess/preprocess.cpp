#include "../helpers/mnist_loader.h"  
#include "../helpers/image_utils.h" 
#include <iostream>
#include <vector>

int main() {
    // Load original MNIST
    auto train_images = MNISTLoader::loadImages("../data/train-images.idx3-ubyte");
    auto train_labels = MNISTLoader::loadLabels("../data/train-labels.idx1-ubyte");
    
    const int num_workers = 4;
    const int samples_per_worker = train_images.size() / (28*28) / num_workers;
    
    // Create non-IID distributions
    for (int worker = 0; worker < num_workers; worker++) {
        std::vector<uint8_t> worker_images;
        std::vector<uint8_t> worker_labels;
        float rotation = 90.0f * worker; // 0°, 90°, 180°, 270°

        for (int i = 0; i < samples_per_worker; i++) {
            int idx = worker * samples_per_worker + i;
            std::vector<uint8_t> img(28*28);
            std::copy(train_images.begin() + idx*28*28, 
                     train_images.begin() + (idx+1)*28*28,
                     img.begin());
            
            ImageUtils::rotate(img, rotation);
            worker_images.insert(worker_images.end(), img.begin(), img.end());
            worker_labels.push_back(train_labels[idx]);
        }

        // Save worker-specific data
        ImageUtils::saveBinary(worker_images, "worker_" + std::to_string(worker) + "_images.bin");
        ImageUtils::saveBinary(worker_labels, "worker_" + std::to_string(worker) + "_labels.bin");
    }

    std::cout << "Preprocessing complete. Created " << num_workers << " worker datasets.\n";
    return 0;
}