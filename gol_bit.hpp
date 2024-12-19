#ifndef GOL_BIT_HPP
#define GOL_BIT_HPP

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>

class Universe {
private:
    std::vector<uint8_t> state_grid;      // Using uint8_t to store cell states
    std::vector<uint8_t> infection_life;   
    std::vector<uint8_t> immune_life;      
    int rows, columns;
    std::string result_dir;
    
public:
    // State encoding: using integers to represent 4 states
    static const uint8_t DEAD     = 0;
    static const uint8_t HEALTHY  = 1;
    static const uint8_t INFECTED = 2;
    static const uint8_t IMMUNE   = 3;

    // Performance statistics variables remain unchanged
    double total_reproduce_time = 0;
    double total_spread_time = 0;
    double total_save_time = 0;
    int generation_count = 0;

    // Lifecycle constants remain unchanged
    static const int INFECTION_LIFETIME = 10;
    static const int IMMUNE_LIFETIME = 5;

    // Constructor
    Universe(int r, int c, const std::string& output_dir) 
        : rows(r), columns(c), result_dir(output_dir) {
        state_grid.resize(r * c, 0);
        infection_life.resize(r * c, 0);
        immune_life.resize(r * c, 0);
    }

    // Get and set states
    inline uint8_t get_state(int row, int col) const {
        return state_grid[row * columns + col];
    }
    void save_generation(int generation);

    inline void set_state(int row, int col, uint8_t state) {
        state_grid[row * columns + col] = state;
    }

    // State conversion functions
    static uint8_t char_to_state(char c) {
        switch (c) {
            case '*': return HEALTHY;
            case '+': return INFECTED;
            case '-': return IMMUNE;
            case '.': return DEAD;
            default: return DEAD;
        }
    }

    static char state_to_char(uint8_t state) {
        switch (state) {
            case HEALTHY:  return '*';
            case INFECTED: return '+';
            case IMMUNE:   return '-';
            case DEAD:     return '.';
            default: return '.';
        }
    }

    void get_performance_stats(double& avg_reproduce, double& avg_spread, double& avg_save) {
        if (generation_count == 0) return;
        avg_reproduce = total_reproduce_time / generation_count;
        avg_spread = total_spread_time / generation_count;
        avg_save = total_save_time / generation_count;
    }

    // Declare friend functions
    friend void reproduce(Universe& universe, bool use_torus);
    friend void spread_infection(Universe& universe, bool use_torus);
    friend void read_in_file(std::ifstream& infile, Universe& u);
    friend void print_statistics(const Universe& u, std::ofstream& stats_file);
};

// Declare external functions
void reproduce(Universe& universe, bool use_torus);
void spread_infection(Universe& universe, bool use_torus);
void read_in_file(std::ifstream& infile, Universe& u);
void print_statistics(const Universe& u, std::ofstream& stats_file);

#endif // GOL_BIT_HPP
