## How to run the code

The code should be first compiled to create an executable. The executable needs only a text file with the necessary inputs.

### Compile the code

Open a terminal, go to the Source folder and type "Make". 
N.B. The makefile creates the executable, called "main" in a folder called "RUNS", that is placed at the same level as "Source". If the folder does not exist, the compilation process will not succeed. 
Modify the makefile to change this setting.

### Run the code

The code needs an input file, called "param.dat", that should be placed in the same folder of the executable. This input file is a simple text file; each entry should be placed on a different line and should be formatted as "key = value"
An example of the input file is in this folder, called "param_template.dat"
- box_x = 20.                 length of the track
- temperature = 0.01          temperature of the heath bath 
- theta_temperature = 0.1     "temperature" of the chemical fluctuations
- friction = 1.               friction coefficient of the heath bath 
- timestep = 0.001            time step for the integration algorithm
- N_particles = 10            number of beads/cells in the filament
- N_steps = 2000000           length of the simulation (in time steps) 
- ratio_samplings = 1000      time lag for configuration dump            
- active_force = 1.           magnitude of the active force
- restart_key = 0             start from a fixed initial configuration or restart from a file 
- r0 = 1.                     spring rest length (or average cell length)
- k_compression = 50.         spring coefficient upon compression 
- k_stretching = 50.          spring coefficient upon extension
- myseed = 42                 seed for PNRGs
- model_type = 0              type of model: 0: all cells self-propel; 1: only the two ends self-propel; 2: only one end self-propel 
- external_field = 5.         magnitude of the extenal input, triggers at the ends of the track
- theta_threshold = 10.       confinement parameter for the variable $\omega$ ($\omega_{\mathrm{max}}$ in the manuscript)
