// Define constants for cell lifetimes
#define INFECTION_LIFETIME 10 // Duration for which a cell remains infected
#define IMMUNE_LIFETIME 5     // Duration for which a cell remains immune

// Function declarations
void init(int rx_, int cy_, char* grid_, int* infection_life_, int* immune_life_, bool use_torus_);
// Initializes the simulation with the given grid dimensions, initial grid state,
// infection and immune lifetimes, and whether to use a torus topology.

void transfer(char* h_grid);
// Transfers the current grid state from the device to the host for further processing or visualization.

void step();
// Executes one simulation step, including reproduction and infection spread, and updates the grid state.

void free_memory();
// Frees allocated memory on the device and host, ensuring proper cleanup after the simulation ends.
