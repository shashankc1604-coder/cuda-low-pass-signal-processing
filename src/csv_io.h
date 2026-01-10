#pragma once
#include <vector>
#include <string>
// Load a CSV file containing numeric samples.
// Supports:
//  - One value per line
//  - Comma-separated values
//
// Returns:
//  - std::vector<float> with parsed samples
//  - Empty vector if file cannot be read
std::vector<float> load_csv(const std::string& filepath);

// Save numeric samples to a CSV file.
// Each value is written on a new line.
void save_csv(const std::string& filepath,
              const std::vector<float>& data);
