#include <iostream>
#include <fstream>
#include <cstring>  
#include <chrono>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>   
#include "gol_1D.hpp"
#include <cuda_runtime.h>
#include "gol_cu.hpp"

using namespace std;
using namespace chrono;
namespace fs = std::filesystem;

// Print usage instructions for the program
void print_usage() {
    cout << "Usage: gameoflife [-i input_file] [-o output_dir] [-g generations] [-s] [-t]" << endl;
    cout << "Options:" << endl;
    cout << "  -i input_file    Input file (required)" << endl;
    cout << "  -o output_dir    Output directory (default: ./output)" << endl;
    cout << "  -g generations   Number of generations (default: 5)" << endl;
    cout << "  -s              Print statistics" << endl;
    cout << "  -t              Use torus topology" << endl;
}

// Helper function to create directories
bool create_directory(const string& path) {
    try {
        fs::create_directories(path);
        return true;
    } catch (const fs::filesystem_error& e) {
        cerr << "Error creating directory: " << e.what() << endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    string input_file;
    string output_dir = "./output";  // Default output directory
    int generations = 5;
    bool print_stats = false;
    bool use_torus = false;

    // Process command-line arguments
    int opt;
    while ((opt = getopt(argc, argv, "i:o:g:sth")) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_dir = optarg;
                break;
            case 'g':
                generations = stoi(optarg);
                if (generations <= 0) {
                    cerr << "Generations must be positive" << endl;
                    return 1;
                }
                break;
            case 's':
                print_stats = true;
                break;
            case 't':
                use_torus = true;
                break;
            case 'h':
                print_usage();
                return 0;
            default:
                print_usage();
                return 1;
        }
    }

    // Validate mandatory parameters
    if (input_file.empty()) {
        cerr << "Input file is required" << endl;
        print_usage();
        return 1;
    }

    // Create output directory
    if (!create_directory(output_dir)) {
        return 1;
    }

    // Create result subdirectory with timestamp
    auto now = system_clock::now();
    auto now_time = system_clock::to_time_t(now);
    string timestamp = to_string(now_time);
    string result_dir = output_dir + "/run_" + timestamp;
    
    if (!create_directory(result_dir)) {
        return 1;
    }

    // Open the input file
    ifstream infile(input_file);
    if (!infile.is_open()) {
        cerr << "Error opening input file: " << input_file << endl;
        return 1;
    }

    // Read grid dimensions
    int rows, cols;
    if (!(infile >> rows >> cols) || rows <= 0 || cols <= 0) {
        cerr << "Invalid grid dimensions" << endl;
        return 1;
    }
    infile.ignore();

    // Initialize the universe
    auto init_start = high_resolution_clock::now();
    Universe u(rows, cols, result_dir);  // Pass output directory to Universe
    read_in_file(infile, u);
    infile.close();
    u.save_generation(1);
    auto init_end = high_resolution_clock::now();

    // Create and open the statistics file
    ofstream stats_file(result_dir + "/statistics.txt");
    if (!stats_file.is_open()) {
        cerr << "Error creating statistics file" << endl;
        return 1;
    }

    // Allocate host memory
    char *h_grid = (char*)malloc(rows * cols * sizeof(char));
    int *h_infection_life = (int*)malloc(rows * cols * sizeof(int));
    int *h_immune_life = (int*)malloc(rows * cols * sizeof(int));

    // Initialize allocated memory
    memset(h_grid, 0, rows * cols * sizeof(char));
    memset(h_infection_life, 0, rows * cols * sizeof(int));
    memset(h_immune_life, 0, rows * cols * sizeof(int));

    // Check memory allocation
    if (h_grid == nullptr || h_infection_life == nullptr || h_immune_life == nullptr) {
        cerr << "Memory allocation failed!" << endl;
        if (h_grid) free(h_grid);
        if (h_infection_life) free(h_infection_life);
        if (h_immune_life) free(h_immune_life);
        return 1;
    }

    // Copy data from Universe to host memory
    memcpy(h_grid, u.grid, rows * cols * sizeof(char));
    memcpy(h_infection_life, u.infection_life, rows * cols * sizeof(int));
    memcpy(h_immune_life, u.immune_life, rows * cols * sizeof(int));

    // Initialize CUDA grid
    init(rows, cols, h_grid, h_infection_life, h_immune_life, use_torus);

    // Start simulation
    auto evolve_start = high_resolution_clock::now();
    for (int gen = 2; gen <= generations; ++gen) {
        step();
        transfer(h_grid);
        //memcpy(u.grid, h_grid, rows * cols * sizeof(char));
        //u.save_generation(gen);
    }
    cudaDeviceSynchronize();
    auto evolve_end = high_resolution_clock::now();

    // Calculate performance statistics
    double init_time = duration_cast<duration<double>>(init_end - init_start).count();
    double evolve_time = duration_cast<duration<double>>(evolve_end - evolve_start).count();

    // Write performance statistics to file
    stats_file << "\nPerformance Statistics:" << endl;
    stats_file << "Initialization time: " << init_time << " seconds" << endl;
    stats_file << "Evolution time for " << generations << " generations: " << evolve_time << " seconds" << endl;
    stats_file << "Average time per generation: " << evolve_time / (generations - 1) << " seconds" << endl;

    // Print statistics to console
    cout << "\nPerformance Statistics:" << endl;
    cout << "Initialization time: " << init_time << " seconds" << endl;
    cout << "Evolution time for " << generations << " generations: " << evolve_time << " seconds" << endl;
    cout << "Average time per generation: " << evolve_time / (generations - 1) << " seconds" << endl;
    cout << "\nResults saved in: " << result_dir << endl;

    // Free resources
    free_memory();
    stats_file.close();
    free(h_grid);
    free(h_infection_life);
    free(h_immune_life);

    return 0;
}
