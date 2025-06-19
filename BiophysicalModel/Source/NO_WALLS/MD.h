
/*_idx == monomer id */


void harmonic_interaction(int _id)
{
    double d;
    int j;
    // use nx for pfx if _id = 0  (False) -- predictor
    // use px for fx  if _id = 1 (True) -- corrector
    
    for(j = 0; j < Sys.NPart-1; j++){
	if(_id) d = myparticle[j+1].px-myparticle[j].px-DHpot.x0[j];
	else d = myparticle[j+1].nx-myparticle[j].nx-DHpot.x0[j];
	myparticle[j].ds = d;
	if( d < 0){
            if(_id){ myparticle[j].fx += DHpot.kcomp[j]*d; myparticle[j+1].fx += -DHpot.kcomp[j+1]*d;}
	    else{ myparticle[j].pfx += DHpot.kcomp[j]*d; myparticle[j+1].pfx += -DHpot.kcomp[j+1]*d;}
	}else{
	    if(_id){ myparticle[j].fx += DHpot.kstr[j]*d; myparticle[j+1].fx += -DHpot.kstr[j+1]*d;}
	    else{ myparticle[j].pfx += DHpot.kstr[j]*d; myparticle[j+1].pfx += -DHpot.kstr[j+1]*d;}
	}	
    }

    myparticle[Sys.NPart-1].ds = myparticle[Sys.NPart-2].ds;

}

double confining_field(int _id){
     double temp;
     
     if(myparticle[_id].theta > Sys.theta_th){ 
         temp = myparticle[_id].theta-Sys.theta_th;
	 return -2*(temp*temp*temp);
     }else if (myparticle[_id].theta < -Sys.theta_th){
         temp = myparticle[_id].theta+Sys.theta_th;
         return -2*(temp*temp*temp);
     }else {
         return 0.;
     }  
}

double harmonic_single(int _id)
{
    double out = 0;
    double d;

    if(_id != 0){ 
	d = myparticle[_id].nx-myparticle[_id-1].nx-DHpot.x0[_id];
        if ( d < 0){
	    out += -(Sys.efield+Sys.Kom)*d; 
	}else{
	    out += -(Sys.efield+Sys.Kom)*d; 
	}
    }

    if(_id != Sys.NPart-1){
        d = myparticle[_id+1].nx-myparticle[_id].nx-DHpot.x0[_id];
        if ( d < 0){
	    out += (Sys.efield+Sys.Kom)*d; 
        }else{
	    out += (Sys.efield+Sys.Kom)*d; 	    
        }
    }

    return out;
}

double harmonic_energy(int _id)
{
    double out = 0;
    double d;

    if (_id != 0) { 
        d = myparticle[_id].nx-myparticle[_id-1].nx-DHpot.x0[_id];
	if( d < 0 ) out += 0.5*DHpot.kcomp[_id]*d*d;
	else out += 0.5*DHpot.kstr[_id]*d*d;
    }
    if (_id != Sys.NPart-1){
        d = myparticle[_id+1].nx-myparticle[_id].nx-DHpot.x0[_id];
        if( d < 0 ) out += 0.5*DHpot.kcomp[_id]*d*d;
        else out += 0.5*DHpot.kstr[_id]*d*d;
    }

    return out;
}

void active_push(int _id)
{
   int j;
   
   for(j = 0; j < Sys.NPart; j++){
       if(_id) myparticle[j].fx += myparticle[j].fa;
       else myparticle[j].pfx += myparticle[j].fa;
   }  
}

void force_nullify(int _id)
{
    long int j;
    if(_id) for(j = 0; j < Sys.NPart; j++) myparticle[j].fx = 0.;
    else for(j = 0; j < Sys.NPart; j++) myparticle[j].pfx = 0.;
}

void forces_predictor()
{
    force_nullify(0);
    harmonic_interaction(0);
    active_push(0);

}

void forces_corrector()
{
    force_nullify(1);
    harmonic_interaction(1);
    active_push(1);
}


/* 
 * Different models:
 * Model 0: all monomers are active, 
 * Model 1: only first and last monomers are active
 * Model 2: only first or last monomer is active
 */

void update_stop_field()
{
   int j;

   for(j=0; j < Sys.NPart; j++){
	if( (myparticle[j].nx > 0 && myparticle[j].nx < Sys.box_x) ){ myparticle[j].stop_field = 0;}
	if( myparticle[j].nx < 0 ){ 
		myparticle[j].stop_field = Sys.efield; Sys.active_end = 0;  
	}
	if( myparticle[j].nx > Sys.box_x ){ 
		myparticle[j].stop_field = -Sys.efield; Sys.active_end = Sys.NPart-1;
	}
   }

}

void active_status()
{

    int j;
    int count = 0;
    double DeltaE, rnd;
 
    for(j = 0; j < Sys.NPart; j++){
        if(Sys._model == 1 && (j != 0 && j != Sys.NPart-1)) continue;
        if(Sys._model == 2 && j != Sys.active_end && (myparticle[j].is_active) == 0) continue;
	myparticle[j].theta += confining_field(j)*Sys.Dt + myparticle[j].stop_field*Sys.Dt + harmonic_single(j)*Sys.Dt + sqrt(2*Sys.Tt*Sys.Dt/1.)*gaussrand(&Sys.seed);

        myparticle[j].is_active = threshold(tanh(myparticle[j].theta)); 
	myparticle[j].is_active_d = tanh(myparticle[j].theta);
    }	
   
    if(Sys._model == 0){   
            for(j = 0; j < Sys.NPart; j++) myparticle[j].fa = myparticle[j].is_active_d*Sys.myfa;
    }else if(Sys._model == 1){ /* ONLY ENDS ARE ACTIVE */
	    myparticle[0].fa = myparticle[0].is_active_d*Sys.myfa; myparticle[Sys.NPart-1].fa = myparticle[Sys.NPart-1].is_active_d*Sys.myfa;
    }else if(Sys._model == 2){
            for(j = 0; j < Sys.NPart; j+=Sys.NPart-1) myparticle[j].fa = myparticle[j].is_active_d*Sys.myfa;
    }else{
            printf("error: entry not valid\n");
            exit(1);
    }
    
}

/*************************************************************************/

void overdamped_predictor()
{
    long int j;
    double zz;

    for(j = 0; j < Sys.NPart; j++) {
        zz = Sys.zeta[1]; 
	myparticle[j].px = myparticle[j].nx + (Sys.Dt/zz)*myparticle[j].pfx + sqrt(2*Sys.T*Sys.Dt/zz)*gaussrand(&Sys.seed);  
    }
  
}

void overdamped_corrector()
{
    double eta2;
    double zz;
    long int j;

    for(j = 0; j < Sys.NPart; j++){ 
	zz = Sys.zeta[1];
        myparticle[j].nx += 0.5*(Sys.Dt/zz)*(myparticle[j].fx+myparticle[j].pfx) + sqrt(2*Sys.T*Sys.Dt/zz)*gaussrand(&Sys.seed);   
    }
}

void print_status(){
   
    int j;

    printf("PRINT STATUS:\n");
    for(j = 0; j < Sys.NPart; j++){
        printf("%d %.5f %.5f %.5f %d %.3f\n", j, myparticle[j].nx, myparticle[j].fx, myparticle[j].ds, myparticle[j].stop_field, myparticle[j].theta/PI);
    }
    printf("----------------------------------\n");


}

void integrate_position()
{
    //update_stop_field();
    active_status();  

    forces_predictor();
    overdamped_predictor();

    forces_corrector();
    overdamped_corrector();

    if(Sys.step % Sys.v_lag == 0) compute_velocity();
}

void do_BD()
{
    char dumpname[100];
    char dumpname_all[100];
    char restartname[100];

    sprintf(dumpname,"trajectory.lammpstrj");
    FILE *f = fopen(dumpname, "w"); fclose(f);

    sprintf(dumpname_all,"trajectory_all.lammpstrj");
    f = fopen(dumpname_all, "w"); fclose(f);

    sprintf(restartname,"restartpoint.dat");

    double msdratio=1.; 

    for(Sys.step = Sys.restart_step; Sys.step < Sys.restart_step+Sys.NSteps; Sys.step++){ 
       
	if(Sys.step % 10000 == 0)  WriteConf(restartname);
	
	if(Sys.step % Sys.NPrint == 0) Write_trajectory_com(dumpname);
	//if(Sys.step % (Sys.NPrint) == 0) Write_trajectory_all(dumpname_all);

	integrate_position();
	//if(Sys.step % 1000 == 0) print_status();


    }

}

