# HPC_Code
This repository documents the hybrid parallelised C code I wrote as part of uni coursework. The course focused on parallel programming on CPUs and looked at inter and intra-node communication using a combination of OpenMP and MPI. This is a reduction of the working repository which was used for the course, which was a mess due to needing duplication of versions for assessment purposes. 

Trade offs in terms both compute resources & efficiency, along with developer resources - having working code in a feasibly short amount of time were explored. This was predominatly done through an extended project, where a serial implementation was developed initially and then a parallel adaptation. Finally the scalability of the code was examined using the cluster computing resources.

My allocated project was solving the non-linear Schrodinger (Gross Piteavski ) equation in 3D using finite differences. Whilst the code needed to produce correct results physical interpretation wasn't the focus here. Most of the other projects were graph theory problems without nice physical interpretation.

I wrote a report about the interpretations, parallelisation scheme implemented and scaling, which is far too long to be read, so in short
- Results were physically meaningful for the simple test cases checked, they were self consistent and corresponded to 1D cases with analytic solutions. 
- The scheme sliced the problem in 1 direction to split across MPI ranks. 
	- It would have been better to use a tiling splitting scheme to reduce communication with increased resources, but the focus was having code work over a short time frame. 
	- There was neat iterative matrix inversion scheme used to avoid having to compute inverses on 1 rank and broadcast results
- Non trivial speedups were exhibited in runtime although scaling was far from ideal strong or weak scaling. Less than ideal parallelisation scheme was a limiting factor here. Runtime still significantly faster than serial code. 
## Plots
A couple of animations from testing the 1D code, there are more cases in the report and they're actually explained, but they're not animated.
- No external potential is applied so wave function is unconstrained and disperses
![Dispersion Plot](https://github.com/m-richards/HPC_Code/blob/master/plots/no-potential-dispersion.gif)
- Forced potential with analytic solution found by Husimi (https://doi.org/10.1143/ptp/9.4.381).
![Husimi Plot](https://github.com/m-richards/HPC_Code/blob/master/plots/husimi-potential.gif)



## Running Code
This is here mainly just for reference. 
Serial code will run directly on the command line (it will throw some -Wunused-variable warnings but they're fine)
```
make serial
./3d_serial.exe <inputs/ws2-80-40-40.inp>reference.out
```
Parallel code requires exporting of OMP environment variable to set the number of OMP threads. 
Note that it will also work with an appropriately configured queue management system (i.e. slurm) script, 
provided the module for mpirun is imported and the argument to -n and OMP_NUM_THREADS are set correspondingly. 
```
make hybrid
export OMP_NUM_THREADS=2
mpirun --bind-to none -x OMP_NUM_THREADS -n 4 3d_hybrid.exe <inputs/ws2-80-40-40.inp>hybrid.out
```