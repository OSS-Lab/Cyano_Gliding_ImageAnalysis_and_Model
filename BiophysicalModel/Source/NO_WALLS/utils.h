
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

void allocate_particle()
{
    myparticle = (Particle *)malloc(Sys.NPart*sizeof(Particle));   
}

void clean_particle()
{
    free(myparticle);
}

double MinD(double dx, double L){

        double dx1;
        dx1 = dx - rint(dx/L)*L;
        return dx1;
}

double P_Img (double z, double L){

        double z1;
        z1 = z - floor(z/L)*L;
        return z1;
}

double sgn(double x){
    return copysign(1,x);
}

int threshold(double x){
  int out;
  if(x > 0.33) return 1;
  if (x < -0.33) return -1;
  return 0;
}

double poisson_rand(double in, double _lambda){
    return -log(1-in)/_lambda;
}

void ReadConf()
{
    FILE* fp = fopen(Sys.restart_file, "r");
    if(fp == NULL){ printf("File does not exist!\n"); exit(1); }

    double a, b, c;

    for(int i = 0; i < Sys.NPart; i++)
    {
        fscanf(fp, "%lf", &a);
        myparticle[i].ox = a;
        myparticle[i].nx = a;
        myparticle[i].x = P_Img(a,Sys.box_x);
    }
    fclose(fp);
}


void WriteConf(char filename[])
{
    FILE* fp = fopen(filename, "w");

    for(int i = 0; i < Sys.NPart; i++)
    {
        fprintf(fp, "%lf\n", myparticle[i].nx);
    }
    fflush(fp); fclose(fp); // Close file when we're done
}



void WriteLammpstrj(char filename[])
{
    FILE* fp = fopen(filename, "w");

    fprintf(fp,"ITEM: TIMESTEP\n%ld\n",Sys.step);
    fprintf(fp,"ITEM: NUMBER OF ATOMS\n%d\n", Sys.NPart);
    fprintf(fp,"ITEM: BOX BOUNDS pp pp pp\n%.0f %.1f\n%.0f %.1f\n%.0f %.1f\n",0.,Sys.box_x,0.,10.,0.,10.);
    fprintf(fp,"ITEM: ATOMS id type q xu yu zu vx vy vz\n");

    int w = 1; int k = 1;
    for(int i = 0; i < Sys.NPart; i++){
        fprintf(fp,"%d %d %d %.5f %.5f %.5f %.5f %.5f %.5f\n", w, 1, myparticle[i].is_active, myparticle[i].nx, 0., 0., myparticle[i].vx, 0., 0.); k++;
        w++;
    }
    fflush(fp); fclose(fp); // Close file when we're done
}

void Write_trajectory(char filename[])
{
    FILE* fp = fopen(filename, "a");

    fprintf(fp,"ITEM: TIMESTEP\n%ld\n",Sys.step);
    fprintf(fp,"ITEM: NUMBER OF ATOMS\n%d\n", 2);
    fprintf(fp,"ITEM: BOX BOUNDS pp pp pp\n%.0f %.1f\n%.0f %.1f\n%.0f %.1f\n",0.,Sys.box_x,0.,10.,0.,10.);
    fprintf(fp,"ITEM: ATOMS id type q xu yu zu vx vy vz\n");

    int w = 1; int k = 1;
    for(int i = 0; i < Sys.NPart; i+=Sys.NPart-1){
        fprintf(fp,"%d %d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n", w, myparticle[i].is_active+2, myparticle[i].theta, myparticle[i].nx, 0., 0., myparticle[i].vx, 0., 0.); k++;
        w++;
    }
    fflush(fp); fclose(fp); // Close file when we're done

}

void Write_trajectory_all(char filename[])
{
    FILE* fp = fopen(filename, "a");

    fprintf(fp,"ITEM: TIMESTEP\n%ld\n",Sys.step);
    fprintf(fp,"ITEM: NUMBER OF ATOMS\n%d\n", Sys.NPart);
    fprintf(fp,"ITEM: BOX BOUNDS pp pp pp\n%.0f %.1f\n%.0f %.1f\n%.0f %.1f\n",0.,Sys.box_x,0.,10.,0.,10.);
    fprintf(fp,"ITEM: ATOMS id type q xu yu zu vx vy vz\n");

    int w = 1; int k = 1;
    for(int i = 0; i < Sys.NPart; i++){
        fprintf(fp,"%d %d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n", w, myparticle[i].is_active+2, myparticle[i].theta, myparticle[i].nx, 0., 0., myparticle[i].vx, 0., 0.); k++;
        w++;
    }
    fflush(fp); fclose(fp); // Close file when we're done

}

void Write_trajectory_com(char filename[])
{
    FILE* fp = fopen(filename, "a");

    fprintf(fp,"ITEM: TIMESTEP\n%ld\n",Sys.step);
    fprintf(fp,"ITEM: NUMBER OF ATOMS\n%d\n", 1);
    fprintf(fp,"ITEM: BOX BOUNDS pp pp pp\n%.0f %.1f\n%.0f %.1f\n%.0f %.1f\n",0.,Sys.box_x,0.,10.,0.,10.);
    fprintf(fp,"ITEM: ATOMS id type q xu yu zu vx vy vz\n");

    int w = 1; int k = 1;
    double com=0; int syncro=0; double q = 0; double v=0; double syncro_abs=0;
    for(int i = 0; i < Sys.NPart; i++){
        com += myparticle[i].nx;
	syncro_abs += fabs((float) myparticle[i].is_active);
	syncro += myparticle[i].is_active;
	q += myparticle[i].theta;
	v += myparticle[i].vx;
	//w++;
    }
    fprintf(fp,"%d %d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n", w, (int) syncro/Sys.NPart+2, (float) syncro_abs/Sys.NPart, com/Sys.NPart, 0., 0., v/Sys.NPart, 0., 0.); k++;

    fflush(fp); fclose(fp); // Close file when we're done

}

void compute_velocity(){
    
    int j;
    for(j = 0; j < Sys.NPart; j++){ 
	myparticle[j].vx = (myparticle[j].nx - myparticle[j].ox)/(Sys.v_lag*Sys.Dt);
        myparticle[j].ox = myparticle[j].nx; 
    }
    
}

