#ifndef CENTRALIZED_H
#define CENTRALIZED_H

#include <vector>
#include <cstdint>
#include <string>

void runCentralizedTraining(const std::string& train_images_path, 
                           const std::string& train_labels_path,
                           const std::string& test_images_path,
                           const std::string& test_labels_path,
                           float validation_ratio = 0.1f);

#endif