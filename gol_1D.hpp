#ifndef GOL_H
#define GOL_H

#include <string>
#include <fstream>

// Define which implementation to use
#define USE_BITWISE_IMPL 0  // Set to 0 to use character-based implementation

#if USE_BITWISE_IMPL
    // Bit-based cell states
    constexpr uint8_t CELL_DEAD     = 0b00000000;
    constexpr uint8_t CELL_HEALTHY  = 0b00000001;
    constexpr uint8_t CELL_INFECTED = 0b00000010;
    constexpr uint8_t CELL_IMMUNE   = 0b00000100;
    constexpr uint8_t CELL_ALIVE    = 0b00000101;
    
    // Helper functions for bit operations
    inline bool is_alive(uint8_t cell) { return cell & CELL_ALIVE; }
    inline bool is_healthy(uint8_t cell) { return cell & CELL_HEALTHY; }
    inline bool is_infected(uint8_t cell) { return cell & CELL_INFECTED; }
    inline bool is_immune(uint8_t cell) { return cell & CELL_IMMUNE; }
#else
    // Character-based cell states
    const char HEALTHY = '*';
    const char INFECTED = '+';
    const char IMMUNE = '-';
    const char DEAD = '.';
#endif

// Common constants
const int INFECTION_LIFETIME = 10;
const int IMMUNE_LIFETIME = 5;

class Universe {
public:
    int rows;
    int columns;
    int aliveCells;
    int totalAliveCells;
    
#if USE_BITWISE_IMPL
    uint8_t** grid;
#else
    char* grid;
#endif

    int* infection_life;
    int* immune_life;
    std::string output_dir;

    Universe(int r, int c, const std::string& output_dir);
    ~Universe();
    void save_generation(int generation) const;

    // Helper function to calculate 1D array index
    inline int index(int i, int j) const { return i * columns + j; }
};

void read_in_file(std::ifstream& infile, Universe& u);
void reproduce(Universe& u, bool use_torus);
void spread_infection(Universe& u, bool use_torus);
void print_statistics(const Universe& u, std::ofstream& stats_file);

#endif
