
#include "centralized.h"
#include "../logisticRegression/logistic.cpp"
#include "../helpers/mnist_loader.h"
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <string>
#include <deque>
#include <cmath>

struct Dataset {
    std::vector<std::vector<float>> train_images;
    std::vector<uint8_t> train_labels;
    std::vector<std::vector<float>> validation_images;
    std::vector<uint8_t> validation_labels;
};

struct ConvergenceChecker {
    std::deque<float> loss_history;
    std::deque<float> round_loss_history;  // For round-level tracking
    int window_size;
    int round_window_size;
    float tolerance;
    float round_tolerance;
    int min_epochs;
    int min_rounds;
    int current_round;
    
    ConvergenceChecker(int window = 5, float tol = 1e-4, int min_ep = 5,
                      int round_window = 3, float round_tol = 1e-3, int min_r = 3) 
        : window_size(window), tolerance(tol), min_epochs(min_ep),
          round_window_size(round_window), round_tolerance(round_tol), 
          min_rounds(min_r), current_round(0) {}
    
    void addLoss(float loss) {
        loss_history.push_back(loss);
        if (loss_history.size() > window_size) {
            loss_history.pop_front();
        }
    }
    
    void addRoundLoss(float round_loss) {
        round_loss_history.push_back(round_loss);
        if (round_loss_history.size() > round_window_size) {
            round_loss_history.pop_front();
        }
        current_round++;
    }
    
    bool hasConverged(int current_epoch) const {
        // Need minimum epochs and full window
        if (current_epoch < min_epochs || loss_history.size() < window_size) {
            return false;
        }
        
        // Calculate moving average and variance
        float sum = 0.0f;
        for (float loss : loss_history) {
            sum += loss;
        }
        float mean = sum / window_size;
        
        float variance = 0.0f;
        for (float loss : loss_history) {
            variance += (loss - mean) * (loss - mean);
        }
        variance /= window_size;
        
        // Check if variance is below tolerance (stable loss)
        return variance < tolerance;
    }
    
    bool hasRoundConverged() const {
        // Need minimum rounds and full window
        if (current_round < min_rounds || round_loss_history.size() < round_window_size) {
            return false;
        }
        
        // Calculate round-level variance
        float sum = 0.0f;
        for (float loss : round_loss_history) {
            sum += loss;
        }
        float mean = sum / round_window_size;
        
        float variance = 0.0f;
        for (float loss : round_loss_history) {
            variance += (loss - mean) * (loss - mean);
        }
        variance /= round_window_size;
        
        return variance < round_tolerance;
    }
    
    float getRecentLossChange() const {
        if (loss_history.size() < 2) return std::numeric_limits<float>::max();
        return std::abs(loss_history.back() - loss_history.front());
    }
    
    float getRecentRoundLossChange() const {
        if (round_loss_history.size() < 2) return std::numeric_limits<float>::max();
        return std::abs(round_loss_history.back() - round_loss_history.front());
    }
    
    int getCurrentRound() const { return current_round; }
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
                           float validation_ratio,
                           int max_rounds = 10) {  // Added max_rounds parameter
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

    const int epochs_per_round = 5;  // Epochs per training round
    const int batch_size = 64;

    std::cout << "Training model for " << max_rounds << " rounds (" 
              << epochs_per_round << " epochs per round)..." << std::endl;

    // Initialize tracking variables
    float best_val_acc = 0.0f;
    int rounds_without_improvement = 0;
    const int round_patience = 2; // Early stopping patience for rounds
    
    // Initialize convergence checker with round consideration
    ConvergenceChecker convergence_checker(5, 1e-4, 5, 3, 1e-3, 3); // Added round parameters
    
    // Main training loop - by rounds
    for (int round = 0; round < max_rounds; round++) {
        std::cout << "\n=== Training Round " << round + 1 << "/" << max_rounds << " ===" << std::endl;
        
        float round_start_loss = 0.0f;
        float round_end_loss = 0.0f;
        bool round_converged_early = false;
        
        // Train for specified epochs in this round
        for (int epoch_in_round = 0; epoch_in_round < epochs_per_round; epoch_in_round++) {
            int global_epoch = round * epochs_per_round + epoch_in_round;
            
            // Train for one epoch
            model.trainEpoch(data.train_images, data.train_labels, batch_size);

            // Evaluate performance
            float train_acc = model.evaluateAccuracy(data.train_images, data.train_labels);
            float val_acc   = model.evaluateAccuracy(data.validation_images, data.validation_labels);
            
            // Calculate validation loss for convergence checking
            float val_loss = model.computeLoss(data.validation_images, data.validation_labels);
            convergence_checker.addLoss(val_loss);
            
            if (epoch_in_round == 0) round_start_loss = val_loss;
            round_end_loss = val_loss;

            std::cout << "  Epoch " << epoch_in_round + 1 << "/" << epochs_per_round 
                      << " (Global: " << global_epoch + 1 << ")"
                      << " - Train Acc: " << train_acc * 100 << "%"
                      << " - Val Acc: " << val_acc * 100 << "%"
                      << " - Val Loss: " << val_loss << std::endl;

            // Check epoch-level convergence within round
            bool epoch_converged = convergence_checker.hasConverged(global_epoch + 1);
            if (epoch_converged) {
                std::cout << "    Epoch-level convergence detected in round " << round + 1 << std::endl;
                round_converged_early = true;
                break;
            }
        }
        
        // Add round-level loss for round convergence tracking
        convergence_checker.addRoundLoss(round_end_loss);
        
        // Calculate round-level metrics
        float final_train_acc = model.evaluateAccuracy(data.train_images, data.train_labels);
        float final_val_acc = model.evaluateAccuracy(data.validation_images, data.validation_labels);
        
        std::cout << "Round " << round + 1 << " Summary:"
                  << " Train Acc: " << final_train_acc * 100 << "%"
                  << " - Val Acc: " << final_val_acc * 100 << "%"
                  << " - Loss Change: " << round_start_loss - round_end_loss << std::endl;

        // Round-level early stopping check (accuracy-based)
        if (final_val_acc > best_val_acc) {
            best_val_acc = final_val_acc;
            rounds_without_improvement = 0;
        } else {
            rounds_without_improvement++;
        }

        // Round-level convergence check
        bool round_converged = convergence_checker.hasRoundConverged();
        if (round_converged) {
            std::cout << "Round-level convergence achieved! Training complete." << std::endl;
            std::cout << "Round loss variance below tolerance." << std::endl;
            std::cout << "Recent round loss change: " << convergence_checker.getRecentRoundLossChange() << std::endl;
            break;
        }

        // Round-level early stopping
        if (rounds_without_improvement >= round_patience) {
            std::cout << "Early stopping: No validation accuracy improvement for " 
                      << round_patience << " rounds." << std::endl;
            break;
        }

        // Print convergence status
        std::cout << "Convergence Status: "
                  << "Epoch-level change = " << convergence_checker.getRecentLossChange()
                  << ", Round-level change = " << convergence_checker.getRecentRoundLossChange()
                  << std::endl;
    }

    // Final evaluation
    float test_acc = model.evaluateAccuracy(test_images, test_labels);
    std::cout << "\nFinal Test Accuracy: " << test_acc * 100 << "%" << std::endl;
}


// Function to create non-IID data using the same strategy as preprocessing
Dataset createNonIIDData(const std::string& train_images_path,
                        const std::string& train_labels_path,
                        int num_workers,
                        float validation_ratio) {
    
    std::cout << "=== Creating Non-IID Dataset ===" << std::endl;
    std::cout << "Simulating " << num_workers << " workers with non-IID characteristics" << std::endl;
    
    // Load original MNIST data
    auto train_images_u8 = MNISTLoader::loadImages(train_images_path);
    auto train_labels = MNISTLoader::loadLabels(train_labels_path);
    
    const int total_samples = train_images_u8.size() / (28 * 28);
    const int samples_per_worker = total_samples / num_workers;
    const int image_size = 28 * 28;
    
    std::cout << "Original dataset: " << total_samples << " samples" << std::endl;
    std::cout << "Target samples per worker: " << samples_per_worker << std::endl;
    
    std::vector<std::vector<float>> all_images;
    std::vector<uint8_t> all_labels;
    
    // Set random seed for reproducibility
    std::srand(42);
    
    // Create data for each worker using the same strategy as preprocessing
    for (int worker = 1; worker <= num_workers; worker++) {
        std::cout << "\nProcessing worker " << worker << "..." << std::endl;
        
        std::vector<std::vector<float>> worker_images;
        std::vector<uint8_t> worker_labels;
        
        // Strategy 1: Class imbalance - each worker focuses on 2-3 classes
        std::vector<int> preferred_classes;
        if (worker == 1) preferred_classes = {0, 1, 2};
        else if (worker == 2) preferred_classes = {2, 3, 4};
        else if (worker == 3) preferred_classes = {4, 5, 6};
        else if (worker == 4) preferred_classes = {6, 7, 8};
        else preferred_classes = {8, 9, 0};
        
        // Strategy 2: Small rotation
        float rotation = 10.0f * (worker - 1); // 0°, 10°, 20°, 30°, 40°
        std::cout << "Worker " << worker << " - Preferred classes: ";
        for (int cls : preferred_classes) std::cout << cls << " ";
        std::cout << "- Rotation: " << rotation << "°" << std::endl;
        
        int samples_collected = 0;
        int attempts = 0;
        const int max_attempts = total_samples * 2; // Prevent infinite loop
        
        // First pass: collect samples with class preference
        while (samples_collected < samples_per_worker && attempts < max_attempts) {
            int idx = attempts % total_samples;
            uint8_t label = train_labels[idx];
            
            bool accept = false;
            if (std::find(preferred_classes.begin(), preferred_classes.end(), label) != preferred_classes.end()) {
                accept = (rand() % 100) < 70; // 70% chance for preferred classes
            } else {
                accept = (rand() % 100) < 30; // 30% chance for other classes
            }
            
            if (accept) {
                std::vector<uint8_t> img_uint8(image_size);
                std::copy(train_images_u8.begin() + idx * image_size,
                          train_images_u8.begin() + (idx + 1) * image_size,
                          img_uint8.begin());

                // Apply rotation if specified
                if (rotation > 0) {
                    ImageUtils::rotate(img_uint8, rotation);
                }

                // Convert to float and normalize
                std::vector<float> img_float(image_size);
                for (int j = 0; j < image_size; j++) {
                    img_float[j] = img_uint8[j] / 255.0f;
                }

                worker_images.push_back(std::move(img_float));
                worker_labels.push_back(label);
                samples_collected++;
            }
            attempts++;
        }
        
        // Second pass: fill remaining samples randomly if needed
        while (samples_collected < samples_per_worker) {
            int idx = rand() % total_samples;
            
            std::vector<uint8_t> img_uint8(image_size);
            std::copy(train_images_u8.begin() + idx * image_size,
                      train_images_u8.begin() + (idx + 1) * image_size,
                      img_uint8.begin());

            if (rotation > 0) {
                ImageUtils::rotate(img_uint8, rotation);
            }

            // Convert to float and normalize
            std::vector<float> img_float(image_size);
            for (int j = 0; j < image_size; j++) {
                img_float[j] = img_uint8[j] / 255.0f;
            }

            worker_images.push_back(std::move(img_float));
            worker_labels.push_back(train_labels[idx]);
            samples_collected++;
        }
        
        // Add worker data to combined dataset
        all_images.insert(all_images.end(), worker_images.begin(), worker_images.end());
        all_labels.insert(all_labels.end(), worker_labels.begin(), worker_labels.end());
        
        // Print class distribution for this worker
        std::vector<int> class_counts(10, 0);
        for (uint8_t label : worker_labels) {
            class_counts[label]++;
        }
        std::cout << "Worker " << worker << " class distribution: ";
        for (int i = 0; i < 10; i++) {
            std::cout << i << ":" << class_counts[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Print combined class distribution
    std::vector<int> combined_class_counts(10, 0);
    for (uint8_t label : all_labels) {
        combined_class_counts[label]++;
    }
    std::cout << "\nCombined non-IID dataset class distribution: ";
    for (int i = 0; i < 10; i++) {
        std::cout << i << ":" << combined_class_counts[i] << " ";
    }
    std::cout << std::endl << "Total samples: " << all_images.size() << std::endl;
    
    // Now split the combined non-IID data into train/validation
    const size_t total_noniid_samples = all_images.size();
    
    // Create indices for shuffling
    std::vector<size_t> indices(total_noniid_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Split into training and validation
    size_t validation_size = static_cast<size_t>(total_noniid_samples * validation_ratio);
    
    Dataset result;
    result.train_images.reserve(total_noniid_samples - validation_size);
    result.train_labels.reserve(total_noniid_samples - validation_size);
    result.validation_images.reserve(validation_size);
    result.validation_labels.reserve(validation_size);
    
    for (size_t i = 0; i < total_noniid_samples; i++) {
        size_t idx = indices[i];
        
        if (i < validation_size) {
            result.validation_images.push_back(all_images[idx]);
            result.validation_labels.push_back(all_labels[idx]);
        } else {
            result.train_images.push_back(all_images[idx]);
            result.train_labels.push_back(all_labels[idx]);
        }
    }
    
    std::cout << "\nFinal dataset split:" << std::endl;
    std::cout << "- Training: " << result.train_images.size() << " samples" << std::endl;
    std::cout << "- Validation: " << result.validation_images.size() << " samples" << std::endl;
    
    return result;
}

void runCentralizedTrainingOnNonIID(const std::string& train_images_path,
                                   const std::string& train_labels_path,
                                   const std::string& test_images_path,
                                   const std::string& test_labels_path,
                                   int num_workers,
                                   float validation_ratio,
                                   int max_rounds = 10) {
    
    std::cout << "=== Centralized Training on Non-IID Data ===" << std::endl;
    
    // Create non-IID dataset using the same strategy as worker preprocessing
    Dataset data = createNonIIDData(train_images_path, train_labels_path, 
                                   num_workers, validation_ratio);
    
    // Load test data (same as original centralized training)
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
    
    std::cout << "\nDataset sizes:" << std::endl;
    std::cout << "- Training: " << data.train_images.size() << " samples" << std::endl;
    std::cout << "- Validation: " << data.validation_images.size() << " samples" << std::endl;
    std::cout << "- Test: " << test_images.size() << " samples" << std::endl;
    
    // Create and train model
    FederatedLogisticRegression model;
    model.printModelInfo();
    
    const int epochs_per_round = 5;  // Epochs per training round
    const int batch_size = 64;
    
    std::cout << "\nTraining model for " << max_rounds << " rounds (" 
              << epochs_per_round << " epochs per round)..." << std::endl;
    
    // Initialize tracking variables
    float best_val_acc = 0.0f;
    int rounds_without_improvement = 0;
    const int round_patience = 2; // Early stopping patience for rounds
    
    // Initialize convergence checker with round consideration
    ConvergenceChecker convergence_checker(5, 1e-4, 5, 3, 1e-3, 3);
    
    // Main training loop - by rounds
    for (int round = 0; round < max_rounds; round++) {
        std::cout << "\n=== Training Round " << round + 1 << "/" << max_rounds << " ===" << std::endl;
        
        float round_start_loss = 0.0f;
        float round_end_loss = 0.0f;
        bool round_converged_early = false;
        
        // Train for specified epochs in this round
        for (int epoch_in_round = 0; epoch_in_round < epochs_per_round; epoch_in_round++) {
            int global_epoch = round * epochs_per_round + epoch_in_round;
            
            // Train for one epoch
            model.trainEpoch(data.train_images, data.train_labels, batch_size);
            
            // Evaluate performance
            float train_acc = model.evaluateAccuracy(data.train_images, data.train_labels);
            float val_acc   = model.evaluateAccuracy(data.validation_images, data.validation_labels);
            
            // Calculate validation loss for convergence checking
            float val_loss = model.computeLoss(data.validation_images, data.validation_labels);
            convergence_checker.addLoss(val_loss);
            
            if (epoch_in_round == 0) round_start_loss = val_loss;
            round_end_loss = val_loss;
            
            std::cout << "  Epoch " << epoch_in_round + 1 << "/" << epochs_per_round 
                      << " (Global: " << global_epoch + 1 << ")"
                      << " - Train Acc: " << train_acc * 100 << "%"
                      << " - Val Acc: " << val_acc * 100 << "%"
                      << " - Val Loss: " << val_loss << std::endl;
            
            // Check epoch-level convergence within round
            bool epoch_converged = convergence_checker.hasConverged(global_epoch + 1);
            if (epoch_converged) {
                std::cout << "    Epoch-level convergence detected in round " << round + 1 << std::endl;
                round_converged_early = true;
                break;
            }
        }
        
        // Add round-level loss for round convergence tracking
        convergence_checker.addRoundLoss(round_end_loss);
        
        // Calculate round-level metrics
        float final_train_acc = model.evaluateAccuracy(data.train_images, data.train_labels);
        float final_val_acc = model.evaluateAccuracy(data.validation_images, data.validation_labels);
        
        std::cout << "Round " << round + 1 << " Summary:"
                  << " Train Acc: " << final_train_acc * 100 << "%"
                  << " - Val Acc: " << final_val_acc * 100 << "%"
                  << " - Loss Change: " << round_start_loss - round_end_loss << std::endl;
        
        // Round-level early stopping check (accuracy-based)
        if (final_val_acc > best_val_acc) {
            best_val_acc = final_val_acc;
            rounds_without_improvement = 0;
        } else {
            rounds_without_improvement++;
        }
        
        // Round-level convergence check
        bool round_converged = convergence_checker.hasRoundConverged();
        if (round_converged) {
            std::cout << "Round-level convergence achieved! Training complete." << std::endl;
            std::cout << "Round loss variance below tolerance." << std::endl;
            std::cout << "Recent round loss change: " << convergence_checker.getRecentRoundLossChange() << std::endl;
            break;
        }
        
        // Round-level early stopping
        if (rounds_without_improvement >= round_patience) {
            std::cout << "Early stopping: No validation accuracy improvement for " 
                      << round_patience << " rounds." << std::endl;
            break;
        }
        
        // Print convergence status
        std::cout << "Convergence Status: "
                  << "Epoch-level change = " << convergence_checker.getRecentLossChange()
                  << ", Round-level change = " << convergence_checker.getRecentRoundLossChange()
                  << std::endl;
    }
    
    // Final evaluation
    float test_acc = model.evaluateAccuracy(test_images, test_labels);
    std::cout << "\nFinal Test Accuracy: " << test_acc * 100 << "%" << std::endl;
}
