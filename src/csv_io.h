#pragma once
#include <vector>
#include <string>
// Load a CSV file containing numeric samples.
// Supports:
//  - One value per line
//  - Comma-separated values
std::vector<float> load_csv(const std::string& filepath);
void save_csv_two_columns(const std::string& filepath,
                          const std::vector<float>& col1,
                          const std::vector<float>& col2,
                          const std::string& name1,
                          const std::string& name2);
