#include "common.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

/*void sendCentroids(const std::vector<Centroid>& centroids, int dest) {
    // First send number of centroids
    int num_centroids = centroids.size();
    MPI_Send(&num_centroids, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

    // Then send dimension (handle empty centroids case)
    int dim = centroids.empty() ? 0 : centroids[0].values.size();
    MPI_Send(&dim, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);

    // Send flattened data only if we have centroids
    if (!centroids.empty() && dim > 0) {
        std::vector<float> flat_data;
        flat_data.reserve(num_centroids * dim);
        for (const auto& c : centroids) {
            flat_data.insert(flat_data.end(), c.values.begin(), c.values.end());
        }
        MPI_Send(flat_data.data(), flat_data.size(), MPI_FLOAT, dest, 2, MPI_COMM_WORLD);
    }
}

vector<Centroid> receiveCentroids(int from_rank) {
    MPI_Status status;
    
    // Receive number of centroids
    int num_centroids;
    MPI_Recv(&num_centroids, 1, MPI_INT, from_rank, 0, MPI_COMM_WORLD, &status);
    
    // Check for termination signal
    if (num_centroids == -1) {
        return {}; // Return empty vector as termination signal
    }
    
    // Receive dimension
    int dim;
    MPI_Recv(&dim, 1, MPI_INT, from_rank, 1, MPI_COMM_WORLD, &status);
    
    // Handle empty centroids case
    if (num_centroids == 0 || dim == 0) {
        return {};
    }
    
    // Receive flattened data
    std::vector<float> flat_data(num_centroids * dim);
    MPI_Recv(flat_data.data(), num_centroids * dim, MPI_FLOAT, from_rank, 2, MPI_COMM_WORLD, &status);
    
    // Reconstruct centroids
    std::vector<Centroid> centroids(num_centroids);
    for (int i = 0; i < num_centroids; i++) {
        centroids[i].values.assign(
            flat_data.begin() + i * dim,
            flat_data.begin() + (i + 1) * dim
        );
    }
    
    return centroids;
}

// ---- Server Functions ----
void initializeRandomCentroids(std::vector<Centroid>& centroids, int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    
    std::cout << "Initializing " << centroids.size() << " centroids with dimension " << dim << std::endl;
    
    for (auto& c : centroids) {
        c.values.resize(dim);
        for (float& val : c.values) {
            val = dist(gen);
            // Ensure initialization is reasonable for MNIST (0-1 range)
            if (!std::isfinite(val)) {
                val = 0.5f; // fallback value
            }
        }
    }
    
    // Print first few values for debugging
    if (!centroids.empty() && !centroids[0].values.empty()) {
        std::cout << "First centroid sample values: ";
        for (int i = 0; i < std::min(5, (int)centroids[0].values.size()); i++) {
            std::cout << centroids[0].values[i] << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<Centroid> averageCentroids(const std::vector<std::vector<Centroid>>& all_centroids) {
    if (all_centroids.empty() || all_centroids[0].empty()) return {};
    
    std::vector<Centroid> avg(all_centroids[0].size());
    int num_workers = all_centroids.size();
    int dim = all_centroids[0][0].values.size();

    // Initialize averaged centroids
    for (size_t c = 0; c < avg.size(); c++) {
        avg[c].values.resize(dim, 0.0f);
    }

    // Sum all worker centroids
    for (size_t c = 0; c < avg.size(); c++) {
        for (int w = 0; w < num_workers; w++) {
            if (all_centroids[w].size() != avg.size()) {
                std::cerr << "Warning: Worker " << w << " has different number of centroids" << std::endl;
                continue;
            }
            if (all_centroids[w][c].values.size() != dim) {
                std::cerr << "Warning: Worker " << w << " centroid " << c << " has wrong dimension" << std::endl;
                continue;
            }
            
            for (int d = 0; d < dim; d++) {
                float val = all_centroids[w][c].values[d];
                if (std::isfinite(val)) {
                    avg[c].values[d] += val;
                } else {
                    std::cerr << "Warning: Non-finite value in worker centroid" << std::endl;
                }
            }
        }
        
        // Average the values
        for (int d = 0; d < dim; d++) {
            avg[c].values[d] /= num_workers;
            
            // Ensure the result is finite
            if (!std::isfinite(avg[c].values[d])) {
                std::cerr << "Warning: Non-finite average centroid value, resetting to 0" << std::endl;
                avg[c].values[d] = 0.0f;
            }
        }
    }
    return avg;
}*/

void sendModel(const Logistic& model, int dest) {
    int dim = model.weights.size();
    MPI_Send(&dim, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(model.weights.data(), dim, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);
    MPI_Send(&model.bias, 1, MPI_FLOAT, dest, 2, MPI_COMM_WORLD);
}

Logistic receiveModel(int from_rank) {
    MPI_Status status;
    int dim;
    MPI_Recv(&dim, 1, MPI_INT, from_rank, 0, MPI_COMM_WORLD, &status);

    Logistic model;
    model.weights.resize(dim);
    MPI_Recv(model.weights.data(), dim, MPI_FLOAT, from_rank, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&model.bias, 1, MPI_FLOAT, from_rank, 2, MPI_COMM_WORLD, &status);

    return model;
}


int32_t read_int(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    if (file.gcount() != 4) {
        throw std::runtime_error("Failed to read 4 bytes for integer");
    }
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<float> loadMNISTImages(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MNIST file: " + path);
    }

    // Read and verify magic number
    int32_t magic = read_int(file);
    if (magic != 2051) {
        std::ostringstream oss;
        oss << "Invalid MNIST image file. Expected magic 2051, got " << magic;
        throw std::runtime_error(oss.str());
    }

    // Read number of images
    int32_t num_images = read_int(file);
    if (num_images != 10000) {
        std::ostringstream oss;
        oss << "Unexpected image count. Expected 10000, got " << num_images;
        throw std::runtime_error(oss.str());
    }

    // Read dimensions
    int32_t rows = read_int(file);
    int32_t cols = read_int(file);
    if (rows != 28 || cols != 28) {
        throw std::runtime_error("Unexpected image dimensions");
    }

    const int image_size = rows * cols;
    std::vector<float> images;
    images.reserve(num_images * image_size);

    // Read image data
    std::vector<uint8_t> buffer(image_size);
    for (int i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), image_size);
        if (file.gcount() != image_size) {
            throw std::runtime_error("Incomplete image data at index " + std::to_string(i));
        }

        // Convert to normalized float (0-1)
        for (uint8_t pixel : buffer) {
            images.push_back(pixel / 255.0f);
        }
    }

    return images;
}

// Load MNIST labels (binary format)
std::vector<uint8_t> loadMNISTLabels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open MNIST labels: " + path);
    }

    // Read and verify magic number
    int32_t magic = read_int(file);
    if (magic != 2049) { // Fixed: labels magic number is 2049, not 10000
        throw std::runtime_error("Invalid MNIST labels file. Expected magic 2049, got " + std::to_string(magic));
    }

    int32_t num_labels = read_int(file);
    std::vector<uint8_t> labels(num_labels);
    
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);
    if (file.gcount() != num_labels) {
        throw std::runtime_error("Failed to read all labels");
    }
    return labels;
}

std::pair<std::vector<std::vector<float>>, std::vector<uint8_t>> loadTestData() {
    std::string images_file = "../federated/test_images.bin";
    std::string labels_file = "../federated/test_labels.bin";

    // Load labels first to determine number of samples
    std::ifstream labels_in(labels_file, std::ios::binary);
    if (!labels_in.is_open()) {
        throw std::runtime_error("Cannot open test labels file: " + labels_file);
    }

    // Get the number of labels by file size
    labels_in.seekg(0, std::ios::end);
    size_t num_samples = labels_in.tellg();
    labels_in.seekg(0, std::ios::beg);

    // Read all labels
    std::vector<uint8_t> labels(num_samples);
    labels_in.read(reinterpret_cast<char*>(labels.data()), num_samples);
    labels_in.close();

    std::cout << "Loaded " << num_samples << " labels" << std::endl;

    // Load images - they are stored as a flat array of floats
    std::ifstream images_in(images_file, std::ios::binary);
    if (!images_in.is_open()) {
        throw std::runtime_error("Cannot open test images file: " + images_file);
    }

    const size_t image_size = 784; // 28 * 28
    const size_t total_floats = num_samples * image_size;

    // Read all image data as a flat array first
    std::vector<float> flat_images(total_floats);
    images_in.read(reinterpret_cast<char*>(flat_images.data()), 
                   total_floats * sizeof(float));
    
    if (!images_in) {
        throw std::runtime_error("Error reading test image data");
    }
    images_in.close();

    // Convert flat array to vector of vectors
    std::vector<std::vector<float>> images(num_samples, std::vector<float>(image_size));
    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < image_size; ++j) {
            images[i][j] = flat_images[i * image_size + j];
        }
    }

    std::cout << "Loaded " << images.size() << " images, each with " << image_size << " pixels" << std::endl;

    return {images, labels};
}

std::pair<std::vector<std::vector<float>>, std::vector<uint8_t>> loadWorkerData(int worker_id) {
    // File names based on worker ID
    std::string images_file = "worker_" + std::to_string(worker_id) + "_images.bin";
    std::string labels_file = "worker_" + std::to_string(worker_id) + "_labels.bin";

    // Load labels using your existing function
    std::vector<uint8_t> labels = loadWorkerLabels(worker_id);

    // Load images:
    std::ifstream file(images_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open images file: " + images_file);
    }

    // Read all image floats from the binary file
    // Assuming each image is 784 floats (28x28), and total images = labels.size()

    size_t num_images = labels.size();
    size_t image_size = 784; // adjust if different

    std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));

    for (size_t i = 0; i < num_images; i++) {
        file.read(reinterpret_cast<char*>(images[i].data()), image_size * sizeof(float));
        if (!file) {
            throw std::runtime_error("Error reading image data from " + images_file);
        }
    }

    file.close();

    return {images, labels};
}

void deleteWorkerData(int num_workers, int rank) {
    if (rank == 0) {
        std::cout << "Process 0 finished, deleting worker data" << std::endl;

        for (int worker_id = 1; worker_id <= num_workers; ++worker_id) {
            std::string image_file = "worker_" + std::to_string(worker_id) + "_images.bin";
            std::string label_file = "worker_" + std::to_string(worker_id) + "_labels.bin";

            if (std::remove(image_file.c_str()) != 0) {
                std::cerr << "Warning: Failed to delete " << image_file << std::endl;
            }

            if (std::remove(label_file.c_str()) != 0) {
                std::cerr << "Warning: Failed to delete " << label_file << std::endl;
            }
        }
    }
}

std::vector<uint8_t> loadWorkerLabels(int rank) {
    int worker_id = rank - 1;
    std::string path = "../federated/worker_" + std::to_string(worker_id + 1) + "_labels.bin";

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Worker " + std::to_string(rank) + 
                                 " failed to open labels: " + path);
    }

    size_t size = file.tellg();
    if (size == 0) {
        throw std::runtime_error("Worker " + std::to_string(rank) + " labels file is empty: " + path);
    }

    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> labels(size);
    if (!file.read(reinterpret_cast<char*>(labels.data()), size)) {
        throw std::runtime_error("Failed to read worker labels from: " + path);
    }

    if (labels.empty()) {
        std::cerr << "ERROR: Worker " << rank << " loaded EMPTY labels!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::cout << "Worker " << rank << " loaded " << labels.size() << " labels" << std::endl;

    return labels;
}


/*std::vector<Centroid> localKMeans(const std::vector<float>& data, 
                                const std::vector<Centroid>& centroids,
                                int dim) {

    if (data.empty()) {
        std::cerr << "ERROR: localKMeans received empty data!" << std::endl;
        return centroids;
    }
    
    if (centroids.empty()) {
        std::cerr << "ERROR: localKMeans received empty centroids!" << std::endl;
        return centroids;
    }
    
    if (data.size() % dim != 0) {
        std::cerr << "ERROR: Data size " << data.size() 
                 << " not divisible by dimension " << dim << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    std::vector<Centroid> updated = centroids;
    std::vector<int> counts(centroids.size(), 0);
    
    // Initialize updated centroids to zero
    for (size_t c = 0; c < updated.size(); c++) {
        std::fill(updated[c].values.begin(), updated[c].values.end(), 0.0f);
    }
    
    int num_samples = data.size() / dim;
    std::cout << "Processing " << num_samples << " samples in localKMeans" << std::endl;
    
    // Assignment and accumulation step
    for (size_t i = 0; i < data.size(); i += dim) {
        float min_dist = std::numeric_limits<float>::max();
        int best_c = 0;
        
        // Find closest centroid
        for (size_t c = 0; c < centroids.size(); c++) {
            float dist = 0;
            for (int d = 0; d < dim; d++) {
                float diff = data[i+d] - centroids[c].values[d];
                if (std::isfinite(diff)) {
                    dist += diff * diff;
                } else {
                    std::cerr << "Warning: Non-finite difference in distance calculation" << std::endl;
                    dist = std::numeric_limits<float>::max();
                    break;
                }
            }
            if (dist < min_dist && std::isfinite(dist)) {
                min_dist = dist;
                best_c = c;
            }
        }
        
        // Accumulate to closest centroid
        for (int d = 0; d < dim; d++) {
            if (std::isfinite(data[i+d])) {
                updated[best_c].values[d] += data[i+d];
            }
        }
        counts[best_c]++;
    }
    
    // Average step
    for (size_t c = 0; c < updated.size(); c++) {
        if (counts[c] > 0) {
            for (int d = 0; d < dim; d++) {
                updated[c].values[d] /= counts[c];
                
                // Ensure result is finite
                if (!std::isfinite(updated[c].values[d])) {
                    std::cerr << "Warning: Non-finite centroid value after averaging" << std::endl;
                    updated[c].values[d] = centroids[c].values[d]; // Keep original
                }
            }
        } else {
            // Keep original centroid if no points assigned
            std::cout << "Warning: No points assigned to centroid " << c << std::endl;
            updated[c] = centroids[c];
        }
    }
    
    // Print assignment counts for debugging
    std::cout << "Centroid assignment counts: ";
    for (size_t c = 0; c < counts.size(); c++) {
        std::cout << counts[c] << " ";
    }
    std::cout << std::endl;
    
    return updated;
}

// Calculate loss as average distance from centroids
double calculateLoss(const std::vector<Centroid>& global_centroids,
                   const std::vector<std::vector<Centroid>>& worker_updates) {
    if (global_centroids.empty() || worker_updates.empty()) {
        return 0.0;
    }
    
    double total_loss = 0.0;
    int count = 0;
    const int dim = global_centroids[0].values.size();
    
    for (const auto& worker_centroids : worker_updates) {
        if (worker_centroids.size() != global_centroids.size()) {
            std::cerr << "Warning: Centroid size mismatch in loss calculation" << std::endl;
            continue;
        }
        
        for (size_t c = 0; c < global_centroids.size(); c++) {
            if (worker_centroids[c].values.size() != dim) {
                std::cerr << "Warning: Dimension mismatch in loss calculation" << std::endl;
                continue;
            }
            
            double dist = 0.0;
            for (int d = 0; d < dim; d++) {
                float diff = global_centroids[c].values[d] - worker_centroids[c].values[d];
                if (std::isfinite(diff)) {  // Check for valid numbers
                    dist += diff * diff;
                } else {
                    std::cerr << "Warning: Non-finite value detected in loss calculation" << std::endl;
                    dist = 0.0;
                    break;
                }
            }
            
            if (std::isfinite(dist) && dist >= 0) {
                total_loss += sqrt(dist);
                count++;
            }
        }
    }
    
    double result = count > 0 ? total_loss / count : 0.0;
    if (!std::isfinite(result)) {
        std::cerr << "Warning: Loss calculation resulted in non-finite value" << std::endl;
        return 0.0;
    }
    
    return result;
}

// Check if loss has converged
bool hasConverged(double prev_loss, double current_loss, double tolerance) {
    return fabs(prev_loss - current_loss) < tolerance;
}

// Evaluate model accuracy on test data
void evaluateModel(const std::vector<Centroid>& centroids, 
                  const std::vector<float>& test_data) {
    if (centroids.empty() || test_data.empty()) {
        std::cout << "Cannot evaluate: empty centroids or test data" << std::endl;
        return;
    }
    
    const int dim = centroids[0].values.size();
    int correct = 0;
    int total = 0;
    
    // Note: This assumes test_data has labels in the last position
    for (size_t i = 0; i < test_data.size(); i += dim + 1) {
        if (i + dim >= test_data.size()) break; // Safety check
        
        float min_dist = std::numeric_limits<float>::max();
        int predicted = -1;
        
        for (size_t c = 0; c < centroids.size(); c++) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = test_data[i+d] - centroids[c].values[d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                predicted = c;
            }
        }
        
        int true_label = static_cast<int>(test_data[i+dim] * 255);
        if (predicted == true_label) {
            correct++;
        }
        total++;
    }
    
    if (total > 0) {
        std::cout << "Model Accuracy: " << (100.0 * correct / total) << "% (" 
                  << correct << "/" << total << ")" << std::endl;
    }
}*/

Logistic averageModels(const std::vector<Logistic>& models) {
    if (models.empty()) return {};

    int dim = models[0].weights.size();
    Logistic avg;
    avg.weights.resize(dim, 0.0f);
    avg.bias = 0.0f;

    for (const auto& m : models) {
        for (int i = 0; i < dim; ++i) {
            avg.weights[i] += m.weights[i];
        }
        avg.bias += m.bias;
    }

    for (int i = 0; i < dim; ++i) {
        avg.weights[i] /= models.size();
    }
    avg.bias /= models.size();

    return avg;
}


// Send termination signal to workers
void sendTerminationSignal(int worker_rank) {
    int signal = -1; // Special termination signal
    MPI_Send(&signal, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
    
    // Send dummy dimension to complete the protocol
    int dummy = 0;
    MPI_Send(&dummy, 1, MPI_INT, worker_rank, 1, MPI_COMM_WORLD);
}