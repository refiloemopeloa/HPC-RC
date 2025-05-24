#include <iostream>              // std::cout, std::endl
#include <vector>                // std::vector
#include <cmath>                 // std::exp, std::log, std::abs
#include <random>                // std::random_device, std::mt19937, std::normal_distribution
#include <numeric>               // std::iota
#include <algorithm>            // std::shuffle, std::max_element
#include <cstdint>              // uint8_t
#include <mpi.h> 

class FederatedLogisticRegression {
private:
    // One-vs-all: 10 binary classifiers for digits 0-9
    std::vector<std::vector<float>> weights; // [class][feature] - 10 x 785 (784 + bias)
    float learning_rate = 0.01f;
    int num_classes = 10;
    int num_features = 784;
    
public:
    FederatedLogisticRegression() {
        initializeWeights();
    }
    
    void initializeWeights() {
        weights.resize(num_classes, std::vector<float>(num_features + 1, 0.0f));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.01f); // Small random initialization
        
        for (int c = 0; c < num_classes; c++) {
            for (int f = 0; f <= num_features; f++) {
                weights[c][f] = dist(gen);
            }
        }
        
        std::cout << "Logistic regression initialized with " << num_classes 
                  << " classes and " << num_features << " features" << std::endl;
    }
    
    float sigmoid(float x) {
        // Stable sigmoid to prevent overflow
        if (x > 500) return 1.0f;
        if (x < -500) return 0.0f;
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    std::vector<float> predict_proba(const std::vector<float>& features) {
        std::vector<float> probabilities(num_classes);
        
        for (int c = 0; c < num_classes; c++) {
            float logit = weights[c][num_features]; // bias term
            
            for (int f = 0; f < num_features; f++) {
                logit += weights[c][f] * features[f];
            }
            
            probabilities[c] = sigmoid(logit);
        }
        
        return probabilities;
    }
    
    int predict(const std::vector<float>& features) {
        std::vector<float> probs = predict_proba(features);
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }
    
    void trainBatch(const std::vector<std::vector<float>>& batch_features,
                   const std::vector<uint8_t>& batch_labels) {
        
        int batch_size = batch_features.size();
        
        // Initialize gradients
        std::vector<std::vector<float>> gradients(num_classes, 
            std::vector<float>(num_features + 1, 0.0f));
        
        // Compute gradients for each sample
        for (int sample = 0; sample < batch_size; sample++) {
            const auto& features = batch_features[sample];
            int true_label = batch_labels[sample];
            
            // Get predictions for all classes
            std::vector<float> predictions = predict_proba(features);
            
            // Update gradients for each class (one-vs-all)
            for (int c = 0; c < num_classes; c++) {
                float y_true = (c == true_label) ? 1.0f : 0.0f;
                float y_pred = predictions[c];
                float error = y_pred - y_true;
                
                // Gradient for bias
                gradients[c][num_features] += error;
                
                // Gradient for weights
                for (int f = 0; f < num_features; f++) {
                    gradients[c][f] += error * features[f];
                }
            }
        }
        
        // Update weights using averaged gradients
        for (int c = 0; c < num_classes; c++) {
            for (int f = 0; f <= num_features; f++) {
                weights[c][f] -= learning_rate * gradients[c][f] / batch_size;
            }
        }
    }
    
    void trainEpoch(const std::vector<std::vector<float>>& all_features,
                   const std::vector<uint8_t>& all_labels,
                   int batch_size = 32) {
        
        // Shuffle data
        std::vector<int> indices(all_features.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Process in batches
        for (size_t start = 0; start < all_features.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, all_features.size());
            
            std::vector<std::vector<float>> batch_features;
            std::vector<uint8_t> batch_labels;
            
            for (size_t i = start; i < end; i++) {
                batch_features.push_back(all_features[indices[i]]);
                batch_labels.push_back(all_labels[indices[i]]);
            }
            
            trainBatch(batch_features, batch_labels);
        }
    }
    
    float evaluateAccuracy(const std::vector<std::vector<float>>& test_features,
                          const std::vector<uint8_t>& test_labels) {
        int correct = 0;
        
        for (size_t i = 0; i < test_features.size(); i++) {
            int predicted = predict(test_features[i]);
            if (predicted == test_labels[i]) {
                correct++;
            }
        }
        
        return static_cast<float>(correct) / test_features.size();
    }
    
    float computeLoss(const std::vector<std::vector<float>>& features,
                     const std::vector<uint8_t>& labels) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < features.size(); i++) {
            std::vector<float> probs = predict_proba(features[i]);
            int true_label = labels[i];
            
            // Cross-entropy loss for the true class
            float prob = std::max(probs[true_label], 1e-10f); // Prevent log(0)
            total_loss -= std::log(prob);
        }
        
        return total_loss / features.size();
    }
    
    // === FEDERATED LEARNING METHODS ===
    
    std::vector<float> serializeWeights() {
        std::vector<float> serialized;
        serialized.reserve(num_classes * (num_features + 1));
        
        for (int c = 0; c < num_classes; c++) {
            for (int f = 0; f <= num_features; f++) {
                serialized.push_back(weights[c][f]);
            }
        }
        
        return serialized;
    }
    
    void deserializeWeights(const std::vector<float>& serialized) {
        int idx = 0;
        for (int c = 0; c < num_classes; c++) {
            for (int f = 0; f <= num_features; f++) {
                weights[c][f] = serialized[idx++];
            }
        }
    }
    
    void averageWeights(const std::vector<std::vector<float>>& worker_weights,
                       const std::vector<int>& worker_data_sizes) {
        
        int total_samples = 0;
        for (int size : worker_data_sizes) total_samples += size;
        
        // Initialize averaged weights to zero
        std::vector<float> averaged_weights(num_classes * (num_features + 1), 0.0f);
        
        // Weighted average based on worker data sizes
        for (size_t worker = 0; worker < worker_weights.size(); worker++) {
            float weight = static_cast<float>(worker_data_sizes[worker]) / total_samples;
            
            for (size_t i = 0; i < worker_weights[worker].size(); i++) {
                averaged_weights[i] += weight * worker_weights[worker][i];
            }
        }
        
        deserializeWeights(averaged_weights);
        
        std::cout << "Averaged weights from " << worker_weights.size() 
                  << " workers (total samples: " << total_samples << ")" << std::endl;
    }
    
    void printModelInfo() {
        std::cout << "=== Logistic Regression Model Info ===" << std::endl;
        std::cout << "Classes: " << num_classes << std::endl;
        std::cout << "Features: " << num_features << std::endl;
        std::cout << "Parameters: " << (num_classes * (num_features + 1)) << std::endl;
        std::cout << "Learning rate: " << learning_rate << std::endl;
        
        // Print weight statistics
        for (int c = 0; c < num_classes; c++) {
            float weight_sum = 0.0f;
            for (int f = 0; f <= num_features; f++) {
                weight_sum += std::abs(weights[c][f]);
            }
            std::cout << "Class " << c << " - Avg |weight|: " 
                      << (weight_sum / (num_features + 1)) << std::endl;
        }
    }
};

// Simple federated training protocol for Logistic Regression
class FederatedLogisticTraining {
public:
    static void runFederatedTraining(int rank, int size, int num_rounds = 15) {
        if (rank == 0) {
            runServer(size - 1, num_rounds);
        } else {
            runWorker(rank, num_rounds);
        }
    }
    
};