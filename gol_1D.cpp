#include "gol_1D.hpp"
#include <iostream>
#include <cstring>

// Constructor
Universe::Universe(int r, int c, const std::string& out_dir) 
    : rows(r), columns(c), aliveCells(0), totalAliveCells(0), output_dir(out_dir) {
    grid = (char*)malloc(rows * columns * sizeof(char));
    infection_life = (int*)malloc(rows * columns * sizeof(int));
    immune_life = (int*)malloc(rows * columns * sizeof(int));
    
    std::memset(grid, DEAD, rows * columns);
    std::memset(infection_life, 0, rows * columns * sizeof(int));
    std::memset(immune_life, 0, rows * columns * sizeof(int));
}

// Destructor
Universe::~Universe() {
    free(grid);
    free(infection_life);
    free(immune_life);
}

void Universe::save_generation(int generation) const {
    // Create filename for this generation
    std::string filename = output_dir + "/generation_" + std::to_string(generation) + ".txt";
    std::ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write grid dimensions first
    outfile << rows << " " << columns << std::endl;

    // Write the current grid state
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            outfile << grid[index(i, j)];
        }
        outfile << std::endl;
    }

    outfile.close();
}

// Read initial state from file
void read_in_file(std::ifstream& infile, Universe& u) {
    std::string line;
    for (int i = 0; i < u.rows && std::getline(infile, line); ++i) {
        for (int j = 0; j < u.columns && j < static_cast<int>(line.size()); ++j) {
            u.grid[u.index(i, j)] = line[j];
            if (line[j] == HEALTHY) {
                u.aliveCells++;
                u.totalAliveCells++;
            } else if (line[j] == INFECTED) {
                u.infection_life[u.index(i, j)] = INFECTION_LIFETIME;
            }
        }
    }
}

// Reproduction logic
void reproduce(Universe& u, bool use_torus) {
    char* new_grid = (char*)malloc(u.rows * u.columns * sizeof(char));
    std::memcpy(new_grid, u.grid, u.rows * u.columns * sizeof(char));

    for (int i = 0; i < u.rows; ++i) {
        for (int j = 0; j < u.columns; ++j) {
            if (u.grid[u.index(i, j)] == INFECTED) continue;

            int healthy_or_immune_neighbors = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0) continue;

                    int ni = i + di;
                    int nj = j + dj;

                    if (use_torus) {
                        ni = (ni + u.rows) % u.rows;
                        nj = (nj + u.columns) % u.columns;
                    } else if (ni < 0 || ni >= u.rows || nj < 0 || nj >= u.columns) {
                        continue;
                    }

                    if (u.grid[u.index(ni, nj)] == HEALTHY || u.grid[u.index(ni, nj)] == IMMUNE) {
                        healthy_or_immune_neighbors++;
                    }
                }
            }

            if (u.grid[u.index(i, j)] == HEALTHY) {
                if (healthy_or_immune_neighbors < 2 || healthy_or_immune_neighbors > 4) {
                    new_grid[u.index(i, j)] = DEAD;
                }
            } else if (u.grid[u.index(i, j)] == DEAD) {
                if (healthy_or_immune_neighbors == 3) {
                    new_grid[u.index(i, j)] = HEALTHY;
                    u.aliveCells++;
                }
            }
        }
    }

    std::memcpy(u.grid, new_grid, u.rows * u.columns * sizeof(char));
    free(new_grid);
}

// Infection spread logic
void spread_infection(Universe& u, bool use_torus) {
    char* new_grid = (char*)malloc(u.rows * u.columns * sizeof(char));
    std::memcpy(new_grid, u.grid, u.rows * u.columns * sizeof(char));

    for (int i = 0; i < u.rows; ++i) {
        for (int j = 0; j < u.columns; ++j) {
            if (u.grid[u.index(i, j)] == INFECTED) {
                // Decrease infection lifetime
                u.infection_life[u.index(i, j)]--;

                // When lifetime ends, become immune
                if (u.infection_life[u.index(i, j)]<= 0) {
                    new_grid[u.index(i, j)] = IMMUNE;
                    u.immune_life[u.index(i, j)] = IMMUNE_LIFETIME;
                    continue;
                }

                // Spread infection
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;

                        int ni = i + di;
                        int nj = j + dj;

                        if (use_torus) {
                            ni = (ni + u.rows) % u.rows;
                            nj = (nj + u.columns) % u.columns;
                        } else if (ni < 0 || ni >= u.rows || nj < 0 || nj >= u.columns) {
                            continue;
                        }

                        // Healthy cells can be infected, immune cells cannot
                        if (u.grid[u.index(ni, nj)] == HEALTHY) {
                            new_grid[u.index(ni, nj)] = INFECTED;
                            u.infection_life[u.index(ni, nj)] = INFECTION_LIFETIME;
                            u.aliveCells--;
                        }
                    }
                }
            } else if (u.grid[u.index(i, j)] == IMMUNE) {
                // Decrease immune lifetime
                u.immune_life[u.index(i, j)]--;
                if (u.immune_life[u.index(i, j)] <= 0) {
                    new_grid[u.index(i, j)] = HEALTHY; // Convert to healthy cell
                    u.aliveCells++;
                }
            }
        }
    }

    std::memcpy(u.grid, new_grid, u.rows * u.columns * sizeof(char));
    free(new_grid);
}

// Print statistics
void print_statistics(const Universe& u, std::ofstream& stats_file) {
    // Calculate cell counts
    int healthy_count = 0;
    int infected_count = 0;
    int immune_count = 0;
    int dead_count = 0;

    for (int i = 0; i < u.rows*u.columns; ++i) {
        switch (u.grid[i]) {
            case HEALTHY:
                healthy_count++;
                break;
            case INFECTED:
                infected_count++;
                break;
            case IMMUNE:
                immune_count++;
                break;
            case DEAD:
                dead_count++;
                break;
        }
    }

    // Calculate percentages
    int total_cells = u.rows * u.columns;
    double healthy_percent = (healthy_count * 100.0) / total_cells;
    double infected_percent = (infected_count * 100.0) / total_cells;
    double immune_percent = (immune_count * 100.0) / total_cells;
    double dead_percent = (dead_count * 100.0) / total_cells;

    // Output to console
    std::cout << "Cell Statistics:" << std::endl;
    std::cout << "Healthy cells:  " << healthy_count << " (" << healthy_percent << "%)" << std::endl;
    std::cout << "Infected cells: " << infected_count << " (" << infected_percent << "%)" << std::endl;
    std::cout << "Immune cells:   " << immune_count << " (" << immune_percent << "%)" << std::endl;
    std::cout << "Dead cells:     " << dead_count << " (" << dead_percent << "%)" << std::endl;
    std::cout << "Total alive:    " << u.aliveCells << std::endl;

    // Output to file
    stats_file << "Cell Statistics:" << std::endl;
    stats_file << "Healthy cells:  " << healthy_count << " (" << healthy_percent << "%)" << std::endl;
    stats_file << "Infected cells: " << infected_count << " (" << infected_percent << "%)" << std::endl;
    stats_file << "Immune cells:   " << immune_count << " (" << immune_percent << "%)" << std::endl;
    stats_file << "Dead cells:     " << dead_count << " (" << dead_percent << "%)" << std::endl;
    stats_file << "Total alive:    " << u.aliveCells << std::endl;
    stats_file << "----------------------------------------" << std::endl;
}
