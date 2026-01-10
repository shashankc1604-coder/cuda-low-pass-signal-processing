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
void save_csv(const std::string& filepath,
              const std::vector<float>& data)
{
    std::ofstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "ERROR: Unable to write CSV file: "
                  << filepath << std::endl;
        return;
    }

    for (const float& value : data) {
        file << value << "\n";
    }

    file.close();
}
