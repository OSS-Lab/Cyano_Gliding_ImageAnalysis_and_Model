void read_input_file()
{
    FILE *_myfile = fopen("param.dat", "r");
    
    char line[100];
    int i=0;
    char *token;
    char *_key=NULL;
    char *_value = NULL;
    
    Sys.restart_step = 0;

    while(fgets(line, sizeof(line), _myfile) != NULL){

      token = strtok(line, " ");
      i = 0;
      while (token != NULL) {
          if (i == 0) _key = token;
	  if (i == 2) _value = token;
          token = strtok(NULL, " ");
          i++;
      }     
   
      if(strcmp("N_particles",_key) == 0){ Sys.NPart = atoi(_value); continue; }
      if(strcmp("box_x",_key) == 0){ Sys.box_x = atof(_value); continue; } 
      if(strcmp("temperature",_key) == 0){ Sys.T = atof(_value); continue; }
      if(strcmp("theta_temperature",_key) == 0){ Sys.Tt = atof(_value); continue; }
      if(strcmp("friction",_key) == 0){ Sys.zeta_null = atof(_value); continue; } 
      if(strcmp("timestep",_key) == 0){ Sys.Dt = atof(_value); continue; }
      
      if(strcmp("model_type",_key) == 0){ Sys._model = atoi(_value); continue; }
          
      if(strcmp("N_steps",_key) == 0){ Sys.NSteps = atoi(_value); continue; }
      if(strcmp("ratio_samplings",_key) == 0){ Sys.NPrint = atoi(_value); continue; }
      if(strcmp("ratio_restartfile",_key) == 0){ Sys.Nrestart = atoi(_value); continue; } 
      if(strcmp("step_restart",_key) == 0){ Sys.restart_step = atoi(_value); continue; }
      if(strcmp("restart_key",_key) == 0){ Sys.restart = atoi(_value); continue; }      
     
      if(strcmp("external_field",_key) == 0){ Sys.efield = atof(_value); continue; }
      if(strcmp("delta_Komega",_key) == 0){ Sys.Kom = atof(_value); continue; }
      if(strcmp("theta_threshold",_key) == 0){ Sys.theta_th = atof(_value); continue; } 
      if(strcmp("active_force",_key) == 0){ Sys.myfa = atof(_value); continue; } 

      if(strcmp("myseed",_key) == 0){ Sys.seed = atoi(_value); continue; } 
      
      if( Sys.mypot == 0){

          DHpot.N = Sys.NPart;
	  if(strcmp("r0",_key) == 0){ DHpot.X0 = atof(_value); continue; }
	  if(strcmp("k_compression",_key) == 0){ DHpot.Kcomp = atof(_value); continue; }
	  if(strcmp("k_stretching",_key) == 0){ DHpot.Kstr = atof(_value); continue; }

          DHpot.x0 = (double*) malloc (DHpot.N*sizeof(double));
          DHpot.kstr = (double*) malloc (DHpot.N*sizeof(double));
          DHpot.kcomp = (double*) malloc (DHpot.N*sizeof(double));
        }

    }
    
    fclose(_myfile);

    sprintf(Sys.restart_file,"restart_file.dat");

}

void init_system()
{
    long int j;
   
    if(Sys.NPart*DHpot.X0 > Sys.box_x){ printf("Error: box too small! enlarging ... \n"); Sys.box_x = 1.2*Sys.NPart*DHpot.X0+2; }
 
    if(Sys.restart == 0){ 
        for(j = 0; j < Sys.NPart; j++){
            myparticle[j].nx = myparticle[j].px = j+1;
	    myparticle[j].ox = myparticle[j].nx;
	    myparticle[j].x = P_Img(myparticle[j].nx,Sys.box_x); 
	    myparticle[j].theta = 1.;
	    myparticle[j].stop_field = 0; 
	}
       
        
    }else{
        ReadConf();
    }
 
    /*ZETA*/
    Sys.zeta = (double *) malloc (3*sizeof(double));
    Sys.zeta[0] = Sys.zeta_null; Sys.zeta[1] = 1*Sys.zeta_null; Sys.zeta[2] = Sys.zeta_null;

    // for model=1
    Sys.v_lag = 100;
    Sys.active_end = 0;

    /***SET HARMONIC PARAMETERS***/
    for(j = 0; j < Sys.NPart; j++)
    {
      DHpot.kcomp[j] = DHpot.Kcomp;
      DHpot.kstr[j] = DHpot.Kstr;
      DHpot.x0[j] = DHpot.X0;	          
    }

    
    for(j = 0; j < Sys.NPart; j++) { 
	//myparticle[j].theta = 0;   
	if(Sys._model == 0){ 
	    myparticle[j].is_active = threshold(tanh(myparticle[j].theta));
            myparticle[j].is_active_d = tanh(myparticle[j].theta);
	}
	else if(Sys._model == 1){
	    if(j == 0 || j == Sys.NPart-1){ 
		myparticle[j].is_active = 0;
		myparticle[j].is_active_d = 0;
	    	myparticle[j].stop_field = 0.; 
	    }
	    else myparticle[j].is_active = 0;
	}
	else if(Sys._model == 2){
            myparticle[j].is_active = 0;
            if(j == Sys.active_end){ 
	        myparticle[j].is_active = 0;
		myparticle[j].is_active_d = 0;
		myparticle[j].stop_field = 0.; 
	    }
        }
        	
	myparticle[j].fa = myparticle[j].is_active_d*Sys.myfa;
    } 
    
    for(j = 0; j < Sys.NPart; j++) myparticle[j].fx = myparticle[j].pfx = 0.;

    Sys.D0 = Sys.T/(Sys.zeta_null); 
    
}




