#include "csv_io.h"

#include <fstream>
#include <sstream>
#include <iostream>


// Load CSV file into vector<float>
std::vector<float> load_csv(const std::string& filepath)
{
    std::vector<float> data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "ERROR: Unable to open CSV file: "
                  << filepath << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                float value = std::stof(token);
                data.push_back(value);
            } catch (const std::exception&) {
                // Skip invalid entries
                continue;
            }
        }
    }

    file.close();
    return data;
}


// Save vector<float> to CSV file
void save_csv_two_columns(const std::string& filepath,
                          const std::vector<float>& col1,
                          const std::vector<float>& col2,
                          const std::string& name1,
                          const std::string& name2)
{
    if (col1.size() != col2.size()) {
        std::cerr << "ERROR: Column size mismatch\n";
        return;
    }

    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: Unable to write CSV: "
                  << filepath << std::endl;
        return;
    }

    // Header
    file << name1 << "," << name2 << "\n";

    // Data
    for (size_t i = 0; i < col1.size(); ++i) {
        file << col1[i] << "," << col2[i] << "\n";
    }

    file.close();
}

