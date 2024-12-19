import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

def read_generation_files(directory):
    """Read all generation files from the directory"""
    generations = []
    
    # Get all generation files from directory
    for i in range(1, 101):  # Assume maximum 100 generation files
        file_path = os.path.join(directory, f'generation_{i}.txt')
        if not os.path.exists(file_path):
            break
            
        with open(file_path, 'r') as file:
            # Read grid size from first line
            rows, cols = map(int, file.readline().split())
            current_gen = []
            
            # Read grid data
            for line in file:
                row = [
                    2 if c == '*' else 1 if c == '+' else 3 if c == '-' else 0 
                    for c in line.strip()
                ]
                current_gen.append(row)
                
        generations.append(np.array(current_gen))
    
    return generations

def create_gif_animation(generations, output_filename='evolution.gif', interval=200, dpi=100):
    """Create GIF animation"""
    # Set figure size
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define color mapping
    colors = ['black', 'red', 'green', 'blue']  # Corresponding to Dead, Infected, Healthy, Immune
    cmap = mcolors.ListedColormap(colors)
    
    # Initialize image
    im = ax.imshow(generations[0], cmap=cmap, vmin=0, vmax=3)
    
    # Add color bar
    cbar = plt.colorbar(im)
    cbar.set_ticks([0.4, 1.2, 2.0, 2.8])
    cbar.set_ticklabels(['Dead', 'Infected', 'Healthy', 'Immune'])
    
    # Set title
    title = ax.set_title('Generation 0')
    
    def update(frame):
        """Update animation frame"""
        im.set_array(generations[frame])
        title.set_text(f'Generation {frame}')
        return [im, title]
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(generations),
        interval=interval, 
        blit=True
    )
    
    # Save animation
    anim.save(
        output_filename, 
        writer='pillow', 
        dpi=dpi
    )
    
    plt.close()

def main():
    try:
        # Use provided directory name
        directory = 'run_1732860464'
        generations = read_generation_files(directory)
        
        if not generations:
            print("Warning: No generation data read")
            return
            
        # Print shape of each generation for debugging
        print("Shape of read generation data:")
        for i, gen in enumerate(generations):
            print(f"Generation {i+1} shape: {gen.shape}")
            
        # Print numeric grids for first few timesteps
        print("\nFirst few numeric grids:")
        for i in range(min(3, len(generations))):  # Only print first 3 timesteps
            print(f"\nGeneration {i+1}:")
            print(generations[i])
            print("\nUnique values in this grid:", np.unique(generations[i]))
        
        # Create GIF animation
        create_gif_animation(
            generations, 
            output_filename='evolution_100x100.gif', 
            interval=200, 
            dpi=200
        )
        
    except Exception as e:
        print(f"Program execution error: {str(e)}")

if __name__ == "__main__":
    main()
