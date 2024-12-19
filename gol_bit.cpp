#include "gol_bit.hpp"
#include <iostream>

void read_in_file(std::ifstream& infile, Universe& u) {
    std::string line;
    int row = 0;
    
    while (std::getline(infile, line) && row < u.rows) {
        for (size_t col = 0; col < static_cast<size_t>(u.columns) && col < line.length(); ++col) {
            uint64_t state = Universe::char_to_state(line[col]);
            u.set_state(row, static_cast<int>(col), state);
            
            // Set infection lifetime for infected cells
            if (state == Universe::INFECTED) {
                size_t index = row * u.columns + static_cast<int>(col);
                u.infection_life[index] = Universe::INFECTION_LIFETIME;
            }
        }
        row++;
    }
}

void Universe::save_generation(int gen_number) {
    std::string filename = result_dir + "/generation_" + std::to_string(gen_number) + ".txt";
    std::ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        std::cerr << "Error creating generation file: " << filename << std::endl;
        return;
    }

    // Write grid dimensions
    outfile << rows << " " << columns << std::endl;

    // Write current state
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            uint64_t state = get_state(i, j);
            outfile << state_to_char(state);
        }
        outfile << std::endl;
    }
    
    outfile.close();
}

void reproduce(Universe& universe, bool use_torus) {
    std::vector<uint8_t> new_state = universe.state_grid;
    int rows = universe.rows;
    int cols = universe.columns;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uint8_t current_state = universe.get_state(i, j);
            if (current_state == Universe::INFECTED) continue;

            // Calculate neighbor coordinates
            int n1i = use_torus ? (i - 1 + rows) % rows : i - 1;    // up
            int n2i = use_torus ? (i - 1 + rows) % rows : i - 1;    // up-left
            int n3i = use_torus ? (i - 1 + rows) % rows : i - 1;    // up-right
            int n4i = i;                                             // left
            int n5i = i;                                             // right
            int n6i = use_torus ? (i + 1) % rows : i + 1;           // down
            int n7i = use_torus ? (i + 1) % rows : i + 1;           // down-left
            int n8i = use_torus ? (i + 1) % rows : i + 1;           // down-right

            int n1j = j;                                             // up
            int n2j = use_torus ? (j - 1 + cols) % cols : j - 1;    // up-left
            int n3j = use_torus ? (j + 1) % cols : j + 1;           // up-right
            int n4j = use_torus ? (j - 1 + cols) % cols : j - 1;    // left
            int n5j = use_torus ? (j + 1) % cols : j + 1;           // right
            int n6j = j;                                             // down
            int n7j = use_torus ? (j - 1 + cols) % cols : j - 1;    // down-left
            int n8j = use_torus ? (j + 1) % cols : j + 1;           // down-right

            // Count living neighbors
            int healthy_or_immune_neighbors = 0;
            
            if (use_torus || (n1i >= 0 && n1j >= 0 && n1i < rows && n1j < cols))
                healthy_or_immune_neighbors += universe.get_state(n1i, n1j) & 0x1;
            if (use_torus || (n2i >= 0 && n2j >= 0 && n2i < rows && n2j < cols))
                healthy_or_immune_neighbors += universe.get_state(n2i, n2j) & 0x1;
            if (use_torus || (n3i >= 0 && n3j >= 0 && n3i < rows && n3j < cols))
                healthy_or_immune_neighbors += universe.get_state(n3i, n3j) & 0x1;
            if (use_torus || (n4i >= 0 && n4j >= 0 && n4i < rows && n4j < cols))
                healthy_or_immune_neighbors += universe.get_state(n4i, n4j) & 0x1;
            if (use_torus || (n5i >= 0 && n5j >= 0 && n5i < rows && n5j < cols))
                healthy_or_immune_neighbors += universe.get_state(n5i, n5j) & 0x1;
            if (use_torus || (n6i >= 0 && n6j >= 0 && n6i < rows && n6j < cols))
                healthy_or_immune_neighbors += universe.get_state(n6i, n6j) & 0x1;
            if (use_torus || (n7i >= 0 && n7j >= 0 && n7i < rows && n7j < cols))
                healthy_or_immune_neighbors += universe.get_state(n7i, n7j) & 0x1;
            if (use_torus || (n8i >= 0 && n8j >= 0 && n8i < rows && n8j < cols))
                healthy_or_immune_neighbors += universe.get_state(n8i, n8j) & 0x1;

            int index = i * universe.columns + j;
            if (current_state == Universe::HEALTHY) {
                if (healthy_or_immune_neighbors < 2 || healthy_or_immune_neighbors > 4) {
                    new_state[index] = Universe::DEAD;
                }
            } else if (current_state == Universe::DEAD) {
                if (healthy_or_immune_neighbors == 3) {
                    new_state[index] = Universe::HEALTHY;
                }
            }
        }
    }
    universe.state_grid = new_state;
}

void spread_infection(Universe& universe, bool use_torus) {
    std::vector<uint8_t> new_state = universe.state_grid;

    for (int i = 0; i < universe.rows; ++i) {
        for (int j = 0; j < universe.columns; ++j) {
            uint8_t current_state = universe.get_state(i, j);
            int index = i * universe.columns + j;

            if (current_state == Universe::INFECTED) {
                universe.infection_life[index]--;
                if (universe.infection_life[index] <= 0) {
                    new_state[index] = Universe::IMMUNE;
                    universe.immune_life[index] = Universe::IMMUNE_LIFETIME;
                    continue;
                }

                // Spread infection to neighbors
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;

                        int ni = use_torus ? (i + di + universe.rows) % universe.rows : i + di;
                        int nj = use_torus ? (j + dj + universe.columns) % universe.columns : j + dj;

                        if (!use_torus && (ni < 0 || ni >= universe.rows || nj < 0 || nj >= universe.columns))
                            continue;

                        if (universe.get_state(ni, nj) == Universe::HEALTHY) {
                            int neighbor_index = ni * universe.columns + nj;
                            new_state[neighbor_index] = Universe::INFECTED;
                            universe.infection_life[neighbor_index] = Universe::INFECTION_LIFETIME;
                        }
                    }
                }
            }
            else if (current_state == Universe::IMMUNE) {
                universe.immune_life[index]--;
                if (universe.immune_life[index] <= 0) {
                    new_state[index] = Universe::HEALTHY;
                }
            }
        }
    }
    universe.state_grid = new_state;
}

void print_statistics(const Universe& u, std::ofstream& stats_file) {
    int healthy_count = 0;
    int infected_count = 0;
    int immune_count = 0;
    int dead_count = 0;

    for (int i = 0; i < u.rows; ++i) {
        for (int j = 0; j < u.columns; ++j) {
            uint64_t state = u.get_state(i, j);
            switch (state) {
                case Universe::HEALTHY: healthy_count++; break;
                case Universe::INFECTED: infected_count++; break;
                case Universe::IMMUNE: immune_count++; break;
                case Universe::DEAD: dead_count++; break;
            }
        }
    }

    int total_cells = u.rows * u.columns;
    stats_file << "Cell Statistics:\n"
              << "Healthy cells:  " << healthy_count << " (" 
              << (healthy_count * 100.0) / total_cells << "%)\n"
              << "Infected cells: " << infected_count << " (" 
              << (infected_count * 100.0) / total_cells << "%)\n"
              << "Immune cells:   " << immune_count << " (" 
              << (immune_count * 100.0) / total_cells << "%)\n"
              << "Dead cells:     " << dead_count << " (" 
              << (dead_count * 100.0) / total_cells << "%)\n";
}
