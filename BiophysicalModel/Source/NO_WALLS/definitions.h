#define PI 3.141592653589793

typedef struct
{
    double x;      //colloid coordinate
    double ox;      //old colloid coordinates

    double px;     //predictor coordinate
    double nx;     //cumulative colloid coordinates
    
    double ds;
    double vx, fx;
    double pfx;
    double fa;
    double theta; 
    
    int stop_field;

    long int ix;       
    int is_active;
    double is_active_d;
    
} Particle;

Particle *myparticle = NULL; 

//defining the System

typedef struct
{
    int NPart;
    long int step;
    long int NSteps, NPrint, Nrestart;      //number of IPCs, number of MC sweeps, frequency of configurations
    long int restart_step;
    int restart;
    long seed;

    double box_x;
    double T, Tt, zeta_null, D0;
    double Dt;
    double *zeta;
    double myfa;
    double efield;
    double theta_th;
    double Kom;

    int _model;
    int active_end;
    int v_lag;

    int mypot;
    int myalg;
    int myanalysis; 

    double T1, Tr;

    char restart_file[100];
    char potential_file[100];

} System;

System Sys;

typedef struct
{
    double X0;
    double Kcomp;
    double Kstr;
    int N;
    double *kcomp;
    double *x0;
    double *kstr;

} DoubleHarmonic;

DoubleHarmonic DHpot;




