// centralized_training.cpp
#include "centralized.h"
#include "../logisticRegression/logistic.cpp"
#include "../helpers/mnist_loader.h"
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

struct Dataset {
    std::vector<std::vector<float>> train_images;
    std::vector<uint8_t> train_labels;
    std::vector<std::vector<float>> validation_images;
    std::vector<uint8_t> validation_labels;
};

Dataset loadAndSplitData(const std::string& images_path, 
                         const std::string& labels_path,
                         float validation_ratio) {
    Dataset result;

    auto images_u8 = MNISTLoader::loadImages(images_path);
    auto labels = MNISTLoader::loadLabels(labels_path);

    const size_t num_samples = labels.size();
    const int image_size = 28 * 28;

    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), g);

    size_t validation_size = static_cast<size_t>(num_samples * validation_ratio);

    result.train_images.reserve(num_samples - validation_size);
    result.train_labels.reserve(num_samples - validation_size);
    result.validation_images.reserve(validation_size);
    result.validation_labels.reserve(validation_size);

    for (size_t i = 0; i < num_samples; i++) {
        size_t idx = indices[i];
        std::vector<float> image(image_size);
        for (int j = 0; j < image_size; j++) {
            image[j] = images_u8[idx * image_size + j] / 255.0f;
        }

        if (i < validation_size) {
            result.validation_images.push_back(std::move(image));
            result.validation_labels.push_back(labels[idx]);
        } else {
            result.train_images.push_back(std::move(image));
            result.train_labels.push_back(labels[idx]);
        }
    }

    return result;
}


void runCentralizedTraining(const std::string& train_images_path,
                           const std::string& train_labels_path,
                           const std::string& test_images_path,
                           const std::string& test_labels_path,
                           float validation_ratio) {
    // Load and split training data
    Dataset data = loadAndSplitData(train_images_path, train_labels_path, validation_ratio);

    // Load test data
    auto test_images_u8 = MNISTLoader::loadImages(test_images_path);
    auto test_labels = MNISTLoader::loadLabels(test_labels_path);

    // Convert test images to float
    std::vector<std::vector<float>> test_images;
    const int image_size = 28 * 28;
    test_images.reserve(test_labels.size());

    for (size_t i = 0; i < test_labels.size(); i++) {
        std::vector<float> image(image_size);
        for (int j = 0; j < image_size; j++) {
            image[j] = test_images_u8[i * image_size + j] / 255.0f;
        }
        test_images.push_back(std::move(image));
    }

    std::cout << "Dataset sizes:" << std::endl;
    std::cout << "- Training: " << data.train_images.size() << " samples" << std::endl;
    std::cout << "- Validation: " << data.validation_images.size() << " samples" << std::endl;
    std::cout << "- Test: " << test_images.size() << " samples" << std::endl;

    // Create and train model
    FederatedLogisticRegression model;
    model.printModelInfo();

    const int epochs = 15;
    const int batch_size = 64;

    std::cout << "Training model for " << epochs << " epochs..." << std::endl;

    float best_val_acc = 0.0f;
    int epochs_without_improvement = 0;
    const int patience = 3; // Early stopping patience

    for (int epoch = 0; epoch < epochs; epoch++) {
        model.trainEpoch(data.train_images, data.train_labels, batch_size);

        float train_acc = model.evaluateAccuracy(data.train_images, data.train_labels);
        float val_acc   = model.evaluateAccuracy(data.validation_images, data.validation_labels);

        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << " - Train Accuracy: " << train_acc * 100 << "%"
                  << " - Val Accuracy: " << val_acc * 100 << "%" << std::endl;

        // Early stopping check
        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement++;
            if (epochs_without_improvement >= patience) {
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                break;
            }
        }
    }

    // Final evaluation
    float test_acc = model.evaluateAccuracy(test_images, test_labels);
    std::cout << "\nFinal Test Accuracy: " << test_acc * 100 << "%" << std::endl;
}
