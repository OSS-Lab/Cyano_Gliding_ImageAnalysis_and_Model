#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <math.h>

#include "random.h"
#include "definitions.h"
#include "utils.h"
#include "init.h"
#include "MD.h"


int main(int argc, char* argv[] ) 
{

    read_input_file();

    allocate_particle();
    init_system();     

    do_BD();

    clean_particle();
    return 0;
}
// g++ -w -fpermissive -std=c++17 -o ../main main.c parse_input.c nrutil.c -lm


