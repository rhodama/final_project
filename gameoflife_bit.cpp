#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>   
#include "gol_bit.hpp"

using namespace std;
using namespace chrono;
namespace fs = std::filesystem;

void print_usage() {
    cout << "Usage: gameoflife [-i input_file] [-o output_dir] [-g generations] [-s] [-t]" << endl;
    cout << "Options:" << endl;
    cout << "  -i input_file    Input file (required)" << endl;
    cout << "  -o output_dir    Output directory (default: ./output)" << endl;
    cout << "  -g generations   Number of generations (default: 5)" << endl;
    cout << "  -s              Print statistics" << endl;
    cout << "  -t              Use torus topology" << endl;
}

// Helper function to create directory
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

    // Process command line arguments
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

    // Validate required parameters
    if (input_file.empty()) {
        cerr << "Input file is required" << endl;
        print_usage();
        return 1;
    }

    // Create output directory
    if (!create_directory(output_dir)) {
        return 1;
    }

    // Create result subdirectory (using timestamp)
    auto now = system_clock::now();
    auto now_time = system_clock::to_time_t(now);
    string timestamp = to_string(now_time);
    string result_dir = output_dir + "/run_" + timestamp;
    
    if (!create_directory(result_dir)) {
        return 1;
    }

    // Open input file
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

    // Initialize universe
    auto init_start = high_resolution_clock::now();
    Universe u(rows, cols, result_dir);  // Pass output directory
    read_in_file(infile, u);
    infile.close();
    u.save_generation(1);
    auto init_end = high_resolution_clock::now();

    // Create and open statistics file
    ofstream stats_file(result_dir + "/statistics.txt");
    if (!stats_file.is_open()) {
        cerr << "Error creating statistics file" << endl;
        return 1;
    }

    // Evolution phase
    auto evolve_start = high_resolution_clock::now();
    for (int gen = 2; gen <= generations; ++gen) {
        // Record reproduce time
        auto reproduce_start = high_resolution_clock::now();
        reproduce(u, use_torus);
        auto reproduce_end = high_resolution_clock::now();
        double reproduce_time = duration_cast<duration<double>>(reproduce_end - reproduce_start).count();
        u.total_reproduce_time += reproduce_time;

        // Record spread_infection time
        auto spread_start = high_resolution_clock::now();
        spread_infection(u, use_torus);
        auto spread_end = high_resolution_clock::now();
        double spread_time = duration_cast<duration<double>>(spread_end - spread_start).count();
        u.total_spread_time += spread_time;

        // Record save_generation time
        auto save_start = high_resolution_clock::now();
        //u.save_generation(gen);
        auto save_end = high_resolution_clock::now();
        double save_time = duration_cast<duration<double>>(save_end - save_start).count();
        u.total_save_time += save_time;

        u.generation_count++;

        if (print_stats) {
            cout << "\nGeneration " << gen << " timing:" << endl;
            cout << "  reproduce time: " << reproduce_time * 1000 << " ms" << endl;
            cout << "  spread time:    " << spread_time * 1000 << " ms" << endl;
            cout << "  save time:      " << save_time * 1000 << " ms" << endl;
        }
    }
    auto evolve_end = high_resolution_clock::now();

    // Calculate performance statistics
    double init_time = duration_cast<duration<double>>(init_end - init_start).count();
    double evolve_time = duration_cast<duration<double>>(evolve_end - evolve_start).count();

    // Write performance statistics to file
    stats_file << "\nPerformance Statistics:" << endl;
    stats_file << "Initialization time: " << init_time << " seconds" << endl;
    stats_file << "Evolution time for " << generations << " generations: " << evolve_time << " seconds" << endl;
    stats_file << "Average time per generation: " << evolve_time / (generations - 1) << " seconds" << endl;

    // Output to console
    cout << "\nPerformance Statistics:" << endl;
    cout << "Initialization time: " << init_time << " seconds" << endl;
    cout << "Evolution time for " << generations << " generations: " << evolve_time << " seconds" << endl;
    cout << "Average time per generation: " << evolve_time / (generations - 1) << " seconds" << endl;
    cout << "\nResults saved in: " << result_dir << endl;

    double avg_reproduce=0;
    double avg_spread=0;
    double avg_save=0;
    u.get_performance_stats(avg_reproduce, avg_spread, avg_save);

    stats_file << "\nDetailed Performance Statistics:" << endl;
    stats_file << "Average reproduce time:        " << avg_reproduce * 1000 << " ms" << endl;
    stats_file << "Average spread infection time: " << avg_spread * 1000 << " ms" << endl;
    stats_file << "Average save generation time:  " << avg_save * 1000 << " ms" << endl;
    stats_file << "Total reproduce time:        " << u.total_reproduce_time * 1000 << " ms" << endl;
    stats_file << "Total spread infection time: " << u.total_spread_time * 1000 << " ms" << endl;
    stats_file << "Total save generation time:  " << u.total_save_time * 1000 << " ms" << endl;

    // Output detailed statistics to console
    cout << "\nDetailed Performance Statistics:" << endl;
    cout << "Average reproduce time:        " << avg_reproduce * 1000 << " ms" << endl;
    cout << "Average spread infection time: " << avg_spread * 1000 << " ms" << endl;
    cout << "Average save generation time:  " << avg_save * 1000 << " ms" << endl;
    cout << "Total reproduce time:        " << u.total_reproduce_time * 1000 << " ms" << endl;
    cout << "Total spread infection time: " << u.total_spread_time * 1000 << " ms" << endl;
    cout << "Total save generation time:  " << u.total_save_time * 1000 << " ms" << endl;
    stats_file.close();
    return 0;
}
