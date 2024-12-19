#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "gol_cu.hpp"
int rx;
int cy;

char* grid;
int* infection_life;
int* immune_life;
char* new_grid;
bool use_torus;
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define INFECTION_LIFETIME 10
#define IMMUNE_LIFETIME 5
#define HEALTHY '*'
#define INFECTED '+'
#define IMMUNE '-'
#define DEAD '.'

void init(int rx_, int cy_, char* grid_ , int* infection_life_, int* immune_life_,bool use_torus_){
    int size;
    rx = rx_;
    cy = cy_;
    size = rx*cy;
    use_torus=use_torus_;
    cudaMalloc(&grid, size * sizeof(char));
    cudaMalloc(&infection_life, size * sizeof(int));
    cudaMalloc(&immune_life, size * sizeof(int));
    cudaMalloc(&new_grid, size * sizeof(char));


    cudaMemcpy(grid, grid_, size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(infection_life, infection_life_, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(immune_life, immune_life_, size * sizeof(int), cudaMemcpyHostToDevice);

}

__global__ void reproduce(int rx, int cy, char* grid, char* new_grid, int* infection_life, int* immune_life, bool use_torus) {
    // Expand shared memory to include the halo region for boundaries
    __shared__ char local_grid[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    // Calculate global thread ID
    int global_i = blockIdx.y * blockDim.y + threadIdx.y;
    int global_j = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate local thread ID in shared memory
    int local_i = threadIdx.y + 1; // Offset by 1 to account for the halo
    int local_j = threadIdx.x + 1; // Offset by 1 to account for the halo

    // Load cell data within the current block into shared memory
    if (global_i < rx && global_j < cy) {
        local_grid[local_i][local_j] = grid[global_i * cy + global_j];
    } else {
        local_grid[local_i][local_j] = DEAD;
        return;
    }

    int index = global_i * cy + global_j;

    // Load halo region data (including edges and diagonal neighbors)
    if (threadIdx.y == 0) { // Top halo
        int global_ni = global_i - 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
        }
        local_grid[0][local_j] = (global_ni >= 0) ? grid[global_ni * cy + global_j] : DEAD;
    }
    if (threadIdx.y == blockDim.y - 1) { // Bottom halo
        int global_ni = global_i + 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
        }
        local_grid[BLOCK_DIM_Y + 1][local_j] = (global_ni < rx) ? grid[global_ni * cy + global_j] : DEAD;
    }
    if (threadIdx.x == 0) { // Left halo
        int global_nj = global_j - 1;
        if (use_torus) {
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[local_i][0] = (global_nj >= 0) ? grid[global_i * cy + global_nj] : DEAD;
    }
    if (threadIdx.x == blockDim.x - 1) { // Right halo
        int global_nj = global_j + 1;
        if (use_torus) {
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[local_i][BLOCK_DIM_X + 1] = (global_nj < cy) ? grid[global_i * cy + global_nj] : DEAD;
    }

    // Load diagonal halo regions
    if (threadIdx.y == 0 && threadIdx.x == 0) { // Top-left corner
        int global_ni = global_i - 1;
        int global_nj = global_j - 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[0][0] = (global_ni >= 0 && global_nj >= 0) ? grid[global_ni * cy + global_nj] : DEAD;
    }
    if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1) { // Top-right corner
        int global_ni = global_i - 1;
        int global_nj = global_j + 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[0][BLOCK_DIM_X + 1] = (global_ni >= 0 && global_nj < cy) ? grid[global_ni * cy + global_nj] : DEAD;
    }
    if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0) { // Bottom-left corner
        int global_ni = global_i + 1;
        int global_nj = global_j - 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[BLOCK_DIM_Y + 1][0] = (global_ni < rx && global_nj >= 0) ? grid[global_ni * cy + global_nj] : DEAD;
    }
    if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1) { // Bottom-right corner
        int global_ni = global_i + 1;
        int global_nj = global_j + 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[BLOCK_DIM_Y + 1][BLOCK_DIM_X + 1] = (global_ni < rx && global_nj < cy) ? grid[global_ni * cy + global_nj] : DEAD;
    }

    // Synchronize threads to ensure shared memory is fully loaded
    __syncthreads();

    // Retrieve the current cell state
    char current_state = local_grid[local_i][local_j];

    // Update the state for infected cells
    if (current_state == INFECTED) {
        new_grid[index] = INFECTED;
        return;
    }

    // Count healthy or immune neighbors
    int healthy_or_immune_neighbors = 0;
    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            if (di == 0 && dj == 0) continue;
            int ni = global_i + di;
            int nj = global_j + dj;
            if (ni < 0 || ni >= rx || nj < 0 || nj >= cy) continue;

            char neighbor_state = local_grid[local_i + di][local_j + dj];
            if (neighbor_state == HEALTHY || neighbor_state == IMMUNE) {
                healthy_or_immune_neighbors++;
            }
        }
    }

    // Update state based on rules
    if (current_state == HEALTHY) {
        if (healthy_or_immune_neighbors < 2 || healthy_or_immune_neighbors > 4) {
            new_grid[index] = DEAD;
        } else {
            new_grid[index] = HEALTHY;
        }
    } else if (current_state == DEAD) {
        if (healthy_or_immune_neighbors == 3) {
            new_grid[index] = HEALTHY;
        } else {
            new_grid[index] = DEAD;
        }
    }

    // Synchronize threads before finishing
    __syncthreads();
}


__global__ void spread_infection(int rx, int cy, char* grid, char* new_grid, int* infection_life, int* immune_life, bool use_torus) {
    // Define shared memory to include the halo region
    __shared__ char local_grid[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    // Calculate global thread ID
    int global_i = blockIdx.y * blockDim.y + threadIdx.y;
    int global_j = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate local thread ID in shared memory
    int local_i = threadIdx.y + 1; // Offset by 1 to account for the halo
    int local_j = threadIdx.x + 1; // Offset by 1 to account for the halo

    // Load cell data within the current block into shared memory
    if (global_i < rx && global_j < cy) {
        local_grid[local_i][local_j] = grid[global_i * cy + global_j];
    } else {
        local_grid[local_i][local_j] = DEAD;
        return;
    }

    int index = global_i * cy + global_j;

    // Load halo region data (including edges and diagonals)
    if (threadIdx.y == 0) { // Top halo
        int global_ni = global_i - 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
        }
        local_grid[0][local_j] = (global_ni >= 0) ? grid[global_ni * cy + global_j] : DEAD;
    }
    if (threadIdx.y == blockDim.y - 1) { // Bottom halo
        int global_ni = global_i + 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
        }
        local_grid[BLOCK_DIM_Y + 1][local_j] = (global_ni < rx) ? grid[global_ni * cy + global_j] : DEAD;
    }
    if (threadIdx.x == 0) { // Left halo
        int global_nj = global_j - 1;
        if (use_torus) {
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[local_i][0] = (global_nj >= 0) ? grid[global_i * cy + global_nj] : DEAD;
    }
    if (threadIdx.x == blockDim.x - 1) { // Right halo
        int global_nj = global_j + 1;
        if (use_torus) {
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[local_i][BLOCK_DIM_X + 1] = (global_nj < cy) ? grid[global_i * cy + global_nj] : DEAD;
    }

    // Load diagonal halo regions
    if (threadIdx.y == 0 && threadIdx.x == 0) { // Top-left corner
        int global_ni = global_i - 1;
        int global_nj = global_j - 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[0][0] = (global_ni >= 0 && global_nj >= 0) ? grid[global_ni * cy + global_nj] : DEAD;
    }
    if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1) { // Top-right corner
        int global_ni = global_i - 1;
        int global_nj = global_j + 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[0][BLOCK_DIM_X + 1] = (global_ni >= 0 && global_nj < cy) ? grid[global_ni * cy + global_nj] : DEAD;
    }
    if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0) { // Bottom-left corner
        int global_ni = global_i + 1;
        int global_nj = global_j - 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[BLOCK_DIM_Y + 1][0] = (global_ni < rx && global_nj >= 0) ? grid[global_ni * cy + global_nj] : DEAD;
    }
    if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1) { // Bottom-right corner
        int global_ni = global_i + 1;
        int global_nj = global_j + 1;
        if (use_torus) {
            global_ni = (global_ni + rx) % rx;
            global_nj = (global_nj + cy) % cy;
        }
        local_grid[BLOCK_DIM_Y + 1][BLOCK_DIM_X + 1] = (global_ni < rx && global_nj < cy) ? grid[global_ni * cy + global_nj] : DEAD;
    }
    __syncthreads();

    int healthy_or_immune_neighbors = 0;
    char current_state = local_grid[local_i][local_j];

    // Handle infected cells
    if (current_state == INFECTED) {
        // Decrease infection lifetime
        infection_life[index]--;

        // If lifetime ends, become immune
        if (infection_life[index] <= 0) {
            new_grid[index] = IMMUNE;
            immune_life[index] = IMMUNE_LIFETIME;
            return;
        }

        // Spread infection to neighbors
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                if (di == 0 && dj == 0) continue; // Skip the current cell
                int ni = global_i + di;
                int nj = global_j + dj;
                if (ni < 0 || ni >= rx || nj < 0 || nj >= cy) continue;

                int neighbor_index = ni * cy + nj;
                char neighbor_state = local_grid[local_i + di][local_j + dj];

                if (neighbor_state == HEALTHY || neighbor_state == IMMUNE) {
                    healthy_or_immune_neighbors++;
                }

                // Infect healthy neighbors
                if (neighbor_state == HEALTHY) {
                    new_grid[neighbor_index] = INFECTED;
                    infection_life[neighbor_index] = INFECTION_LIFETIME;
                }
            }
        }
    }
    // Handle immune cells
    else if (current_state == IMMUNE) {
        // Decrease immune lifetime
        immune_life[index]--;

        // If lifetime ends, become healthy
        if (immune_life[index] <= 0) {
            new_grid[index] = HEALTHY;
        }
    }
    __syncthreads();
}

void step() {
    // Calculate grid dimensions
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize(
        (cy + BLOCK_DIM_X - 1) / BLOCK_DIM_X, // Number of blocks in x-direction
        (rx + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y  // Number of blocks in y-direction
    );

    // Launch kernels with 2D thread blocks and grids
    reproduce<<<gridSize, blockSize>>>(rx, cy, grid, new_grid, infection_life, immune_life, use_torus);
    cudaDeviceSynchronize();

    // Copy results to the grid
    cudaMemcpy(grid, new_grid, rx * cy * sizeof(char), cudaMemcpyDeviceToDevice);

    spread_infection<<<gridSize, blockSize>>>(rx, cy, grid, new_grid, infection_life, immune_life, use_torus);
    cudaDeviceSynchronize();

    // Copy the results back to the original grid
    cudaMemcpy(grid, new_grid, rx * cy * sizeof(char), cudaMemcpyDeviceToDevice);
}

void transfer(char *h_grid)
{
    cudaMemcpy(h_grid, grid, rx * cy * sizeof(char),
                               cudaMemcpyDeviceToHost);
}



void free_memory()
{
    cudaFree(grid);
    cudaFree(infection_life);
    cudaFree(immune_life);
    cudaFree(new_grid);
} 