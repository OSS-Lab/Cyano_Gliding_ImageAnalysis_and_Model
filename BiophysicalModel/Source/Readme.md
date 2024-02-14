Source code for the Biophysical model reported in Cellular (de)coordination in gliding motility and plectoneme formation
https://doi.org/10.1101/2024.02.06.579126 

Author: Emanuele Locatelli

Description: 
the code for simulating the biophysical model for filament reversal comprises
main.c : main file
definitions.h : data structures definitions 
init.h : read and assign external values for model paramters and set the initial conditions
random.h : library for PRNGs (taken from Numerical Recipes) 
utils.h : miscallaneous routines, such as write dump files in the style of LAMMPS, allocating memory etc
MD.h : routines for performing Brownian Dynamics simulations of the biophysical model
makefile : simple Make file to compile the code
