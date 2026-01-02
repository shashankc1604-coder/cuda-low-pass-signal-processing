#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

/*
 * Load sensor column from CSV file
 */
std::vector<float> load_sensor_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    std::vector<float> data;

    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filepath << std::endl;
        return data;
    }

    std::string line;
    std::getline(file, line);  // Read header

    // Find index of "sensor" column
    std::stringstream header_stream(line);
    std::string column;
    int sensor_col_index = -1;
    int col_index = 0;

    while (std::getline(header_stream, column, ',')) {
        if (column == "sensor") {
            sensor_col_index = col_index;
            break;
        }
        col_index++;
    }

    if (sensor_col_index == -1) {
        std::cerr << "CSV does not contain 'sensor' column\n";
        return data;
    }

    // Read data rows
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream line_stream(line);
        std::string cell;
        int current_col = 0;

        while (std::getline(line_stream, cell, ',')) {
            if (current_col == sensor_col_index) {
                try {
                    data.push_back(std::stof(cell));
                } catch (...) {
                    data.push_back(0.0f);  // Graceful fallback
                }
                break;
            }
            current_col++;
        }
    }

    file.close();
    return data;
}

/*
 * Save filtered signal to CSV file
 */
void save_sensor_csv(
    const std::string& filepath,
    const std::vector<float>& data
) {
    std::ofstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Failed to write CSV file: " << filepath << std::endl;
        return;
    }

    file << "index,filtered_sensor\n";

    for (size_t i = 0; i < data.size(); i++) {
        file << i << "," << data[i] << "\n";
    }

    file.close();
}
