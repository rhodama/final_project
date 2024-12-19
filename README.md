How to compile the code:
Run serial baseline: make build/serial
Run tuned serial version: make build/bit
Run CUDA version: make build/gpu

How to run the code on the perlmutter:
sbatch submit_serial.sh/submit_bit.sh/submit_cuda.sh

How to verify the results:
1. uncomment the save_generation function in gameoflife_1D.cpp/gameoflife_bit.cpp/gameoflife_cuda.cpp
2. compile the code and submit them on perlmutter, change the -g option from 10000 to 100(or the saving time would be too long to crash)
3. When the perlmutter returns a .out file, it will show that the generation file is saved to ./results/run_xxxx directory. Get into this directory and copy the relative path of the 100th generation.txt file.
4. Open the diff.py and change the path of File2 to what you just copied, and run the diff.py. If they are identical. then it is correct!




