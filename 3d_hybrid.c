/*
* 3D solver for the gross-pitaveski equation in natural units.
*
* A quick note on code convention. Functions with MPI function calls in them
* are prefixed with PARALLEL_MPI. Perhaps this is not ideal style, but the intent is to make it explicit from the level
* up that MPI send/recv operations are going on. Note however that only the function directly containing the MPI calls
* has the flag, not functions in layers up. [Would just prefix with MPI but this is forbidden by MPI]
*
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdbool.h>
#include <omp.h>

#define PI 3.1415926535

const int ROOT = 0;
const int SERIAL_NO_RANK = -1;

// Used for neighbour rank calculation
int modulo(int a, int b) {
    return (a % b + b) % b;
}

//Utility function from Assessed lab 2
void root_print(int my_rank, char *format, ...) {
    va_list args;
    if (my_rank != ROOT) {
        return;
    }
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}
//Utility function from Assessed lab 2
void rank_print(int my_rank, char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("Rank %d: ", my_rank);
    vprintf(format, args);
    va_end(args);
}

// Simple helper function to take max of 3 ints, mainly abstracted so purpose is clear
int three_num_max(int a, int b, int c){
    int result = a>b? a:b;
    return  result>c? result:c;
}


// Establishes a gaussian initial condition
double ic_function(double x, double y, double z, double sigma0) {

    return (sqrt(1.0 / (sqrt(PI) * sigma0)) * exp((-x * x - y * y - z * z) / (2.0 * pow(sigma0, 2.0))));
}


// external potential in one spatial dimension
double linear_potential_func(double x, double freq) {
    return 0.5 * freq * freq * x * x;
}


/* Tridiagonal matrix solver. complex variable modification of code from Michael Bromley
 * under copyleft permissions given to students in PHYS3071. Originally based on code
 * From numerical recipies. gam is an array of length n to be used as memory workspace.
 * solution to system in xsoln.
 */
int tridag_cplx(double complex adiag[], const double complex alower[], const double complex aupper[],
                const double complex rrhs[], double complex xsoln[], int n, double complex *gam) {
    int j;
    double complex bet;

    if (cabs(adiag[0]) == 0.0) return -1;    // Error

    bet = adiag[0];   // bet this is used as workspace
    xsoln[0] = rrhs[0] / bet;

    for (j = 1; j < n; j++) {
        gam[j] = aupper[j - 1] / bet;
        bet = adiag[j] - alower[j] * gam[j];
        if (cabs(bet) == 0.0) return -1;   // Pivoting Error
        xsoln[j] = (rrhs[j] - alower[j] * xsoln[j - 1]) / bet;
    }
    for (j = n - 2; j >= 0; j--) {
        xsoln[j] -= gam[j + 1] * xsoln[j + 1];  // Backsubstition
    }
    return 0;
}


/* Function to return 1D flattened index of 3 variables */
int get_1d_index(int j_x, int j_y, int j_z, int n_x, int n_y) {
    int index = j_x + n_x * j_y + n_x * n_y * j_z;
    return index;
}


/* Function to perform exchange of information between boundary values
 * of the mpi ranks. Note that this function contains the send and recv calls
 * but does not have waits to check if they are actually obtained.
 * This is intended, so that we can ensure communication happens as late as
 * possible (and is why requests are passed into the function)
 * buffer_size allows function to take array of boundary values -i.e. 1 b.c for each openMP thread
 * */
void PARALLEL_MPI_exchange_boundaries(int rank, int num_mpi_ranks,
                                      double complex *l_recv_buf, double complex *r_recv_buf,
                                      double complex *l_send_buf, double complex *r_send_buf,
                                      int buffer_size,
                                      MPI_Request *recv_reqs_lhs, MPI_Request *send_reqs_lhs) {

    const int last_rank = num_mpi_ranks - 1;
    int left_rank = modulo(rank - 1, num_mpi_ranks);
    int right_rank = modulo(rank + 1, num_mpi_ranks);

    /* Communication pattern
    * recv L
    * recv R
    * send L
    * send R
    */
    const int tag_send_left_recv_right = 3; //explicit tag values
    const int tag_send_right_recv_left = 2;

    if (rank != 0) {
        MPI_Irecv(l_recv_buf, buffer_size, MPI_C_DOUBLE_COMPLEX, left_rank,
                  tag_send_right_recv_left, MPI_COMM_WORLD, &recv_reqs_lhs[0]);
    }
    if (rank != last_rank) {
        MPI_Irecv(r_recv_buf, buffer_size, MPI_C_DOUBLE_COMPLEX, right_rank,
                  tag_send_left_recv_right, MPI_COMM_WORLD, &recv_reqs_lhs[1]);
    }

    // Send operation
    if (rank != 0) {
        MPI_Isend(l_send_buf, buffer_size, MPI_C_DOUBLE_COMPLEX, left_rank,
                  tag_send_left_recv_right, MPI_COMM_WORLD, &send_reqs_lhs[0]);
    }
    if (rank != last_rank) {

        MPI_Isend(r_send_buf, buffer_size, MPI_C_DOUBLE_COMPLEX, right_rank,
                  tag_send_right_recv_left, MPI_COMM_WORLD, &send_reqs_lhs[1]);
    }
}

/* Helper function to check MPI waits on all specified request array, taking into account
 * the edge case ranks
 */
void PARALLEL_MPI_check_waits(MPI_Request *reqs, int rank, int num_mpi_ranks) {
    if (num_mpi_ranks==1) return; //no exchange so no need to wait
    if ((rank != 0) && (rank != num_mpi_ranks - 1)) { //do this first since most likely
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    } else if (rank == 0) {
        MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
    } else {  // rank==last rank
        MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);

    }
}

/*Function to build solution to compute the solution to the explict side matrix vector product
 */
void build_rhs_explicit(const double complex *z_psi_old_l, double complex *z_psi_explicit_l,
                        int num_x_pts, double x_left, double dx, double complex z_alpha, double complex z_beta,
                        double g_coeff, double potential_scaling, double dimensional_frequency, double complex left_bc,
                        double complex right_bc) {
    double x_curr;
    double complex potential_current, z_rho_left, z_rho_right;

    #pragma omp parallel for private (x_curr, z_rho_left, z_rho_right, potential_current)
    for (int j_x = 0; j_x < num_x_pts; j_x++) {
        x_curr = x_left + dx * (1 + j_x);

        //reference to point left and right of current [j_x, t] point
        if (j_x == 0) {
            z_rho_left = left_bc;
        } else {
            z_rho_left = z_psi_old_l[j_x - 1];
        }
        if (j_x == num_x_pts - 1) {
            z_rho_right = right_bc;
        } else {
            z_rho_right = z_psi_old_l[j_x + 1];
        }
        //Calculate current potential for t_curr
        potential_current = linear_potential_func(x_curr, dimensional_frequency) + potential_scaling * (
                g_coeff * pow(cabs(z_psi_old_l[j_x]), 2.0));

        z_psi_explicit_l[j_x] = (1.0 - 2.0 * z_alpha - z_beta * potential_current
                                        ) * z_psi_old_l[j_x]
                                        + z_alpha * (z_rho_left + z_rho_right);
    }
}


/*Constructs a new main diagonal for the implicit side matrix, uses the previous iterated estimate of z_psi_new
 * to do so. Does not depend on any asynchronous communication (other than that used to construct z_psi_new in
 * the first place.
 */
void build_new_main_diag(double complex *za_main_diag_l, double complex *z_psi_future_approx,
                         int num_x_global_pts, double x_left, double dx, double g_coeff,
                         double complex z_alpha, double complex z_beta, double potential_scaling,
                         double dimensional_frequency) {
    double x_curr;
    double complex potential_future;
    #pragma omp parallel for private(x_curr, potential_future)
    for (int j_x = 0; j_x < num_x_global_pts; j_x++) {
        x_curr = x_left + dx * (1 + j_x);
        // note this is estimate dependent on iteration
        potential_future = linear_potential_func(x_curr, dimensional_frequency) + potential_scaling * (
                g_coeff * pow(cabs(z_psi_future_approx[j_x]), 2.0));

        // setup main diag based on iterative approximation to future potential
        za_main_diag_l[j_x] = (double complex) 1.0
                              + (double complex) (2.0) * z_alpha
                              + z_beta * potential_future;
    }

}

/* Solver for a single time step of the 1D (nonlinear) schrodinger equation
 * Takes in rank, num_ranks, vector of previous time step values, vector to store rhs matrix evaluation in,
 * vector to store result of matrix inversion in,
 * memory workspace vector
 * main, upper and lower diagonals of lhs matrix
 * length of vectors, nonlinear GPE parameter g, leftmost_x_point, x step size,
 * matrix element collected coefficients alpha and beta,
 * dimensional potential scaling factor (1/3) to evenly divide,
 * frequency parameter of harmonic oscilator potential
 * left and right "boundary conditions" - which may be the result of MPI communication prior
 * which are used in rhs matrix construction,
 * n_iter - the number of iteration to do for nonlinear convergence and matrix inversion convergence*/
void one_dim_evolution(int rank, int num_mpi_ranks, double complex *z_psi_old_l, double complex *z_psi_explicit_l,
                       double complex *z_psi_implicit_l, double complex *z_workspace_vec_l,
                       double complex *za_main_diag_l, double complex *za_upper_diag_l, double complex *za_lower_diag_l,
                       int num_x_pts, double g_coeff,
                       double x_left, double dx, double complex z_alpha, double complex z_beta,
                       double potential_scaling, double dimensional_frequency,
                       double complex left_bc, double complex right_bc, int n_iter) {

//     Setup explicit side solution
    build_rhs_explicit(z_psi_old_l, z_psi_explicit_l, num_x_pts, x_left, dx, z_alpha, z_beta, g_coeff,
                       potential_scaling, dimensional_frequency, left_bc, right_bc);

    // we re-purpose old time vector as iteration vector since we no longer need old, and the initial
    // iter values are z_psi_old to start with
    double complex *z_psi_iter_l = z_psi_old_l;

    // static references to original top and bottom elements of rhs vector (used for parallel case
    // as this vector gets deformed by values from communication
    double complex z_explicit_top_elem = z_psi_explicit_l[0];
    double complex z_explicit_bottom_elem = z_psi_explicit_l[num_x_pts - 1];

    // Iterations for dual purpose - nonlinear convergence & matrix pseudoinverse conversion
    for (int k_iter = 0; k_iter < n_iter; k_iter++) {

        // if call is made from parallel context need to get new corner bc to update rhs vector
        if (rank != SERIAL_NO_RANK) {
            double complex l_recv_buf, r_recv_buf;
            l_recv_buf = 0.0;
            r_recv_buf = 0.0;
            MPI_Request recv_reqs_lhs[2];
            MPI_Request send_reqs_lhs[2];
            PARALLEL_MPI_exchange_boundaries(rank, num_mpi_ranks, &l_recv_buf, &r_recv_buf,
                                             &z_psi_iter_l[0], &z_psi_iter_l[num_x_pts - 1],
                                             1, recv_reqs_lhs, send_reqs_lhs);

            // can build diag without corners so run now to allow communication to happen
            build_new_main_diag(za_main_diag_l, z_psi_iter_l, num_x_pts, x_left, dx, g_coeff,
                                z_alpha, z_beta, potential_scaling, dimensional_frequency);

            PARALLEL_MPI_check_waits(recv_reqs_lhs, rank, num_mpi_ranks); //now need these values
            z_psi_explicit_l[0] = z_explicit_top_elem + z_alpha * l_recv_buf;
            z_psi_explicit_l[num_x_pts - 1] = z_explicit_bottom_elem + z_alpha * r_recv_buf;
            PARALLEL_MPI_check_waits(send_reqs_lhs, rank, num_mpi_ranks);

        } else {// Serial code does not need to worry about communication so just build main diag
            build_new_main_diag(za_main_diag_l, z_psi_iter_l, num_x_pts, x_left, dx, g_coeff,
                                z_alpha, z_beta, potential_scaling, dimensional_frequency);
        }
//
        // solve for implicit soln - returned to z_psi_implicit_l
        int ierr = tridag_cplx(za_main_diag_l, za_lower_diag_l, za_upper_diag_l,
                               z_psi_explicit_l, z_psi_implicit_l, num_x_pts,
                               z_workspace_vec_l);
        if (ierr != 0) {
            printf("Matrix inversion failed. Exiting\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
        // iteration update scheme, swap pointers to avoid copies
        double complex *z_psi_temp = z_psi_iter_l;
        z_psi_iter_l = z_psi_implicit_l;
        z_psi_implicit_l = z_psi_temp;
    }
}
/*Function to handle reading inputs from file redirect, since this needs to be done on 1 Rank.
 * output is broadcast to other ranks. Note that this function is intentionally messy
 * with pointers and dereferencing, so that we can assign the exterior input values locally,
 * rather than returning an array and having to do that outside the function*/
int PARALLEL_MPI_read_inputs(int rank, int *num_x_pts_global, int *num_y_pts, int *num_z_pts,
        int * num_t_pts, int *n_iter,
        double *x_left_global, double *x_right, double *sigma0_x, double *x_freq,
        double *y_left, double *y_right, double *sigma0_y, double *y_freq,
        double *z_left, double *z_right, double *sigma0_z, double *z_freq,
        double *dt, double *g_coeff){

    double input_params_array[19];
    if (rank == ROOT) {
        //X dir
        printf("X direction parameters:\n");
        printf("Input the IC scaling factor sigma0_x \n");
        scanf("%lf", sigma0_x);
        printf("sigma0_x = %lf\n", *sigma0_x);
        printf("Input the number of x grid points num_x_pts\n");
        scanf("%d", num_x_pts_global);
        printf("num_x_pts = %d\n", *num_x_pts_global);
        printf("Input the left end x_coordinate x_left\n");
        scanf("%lf", x_left_global);
        printf("x_left = %lf\n", *x_left_global);
        printf("Input the right end x-coordinate x_right0\n");
        scanf("%lf", x_right);
        printf("x_right = %lf\n", *x_right);
        printf("input the x_direction frequency x_freq\n");
        scanf("%lf", x_freq);
        printf("x_freq = %lf\n", *x_freq);
        input_params_array[0] = *sigma0_x;
        input_params_array[1] = (double) *num_x_pts_global;
        input_params_array[2] = *x_left_global;
        input_params_array[3] = *x_right;
        input_params_array[4] = *x_freq;


        //y_dir
        printf("y direction parameters:\n");
        printf("Input the IC scaling factor sigma0_y \n");
        scanf("%lf", sigma0_y);
        printf("sigma0_y = %lf\n", *sigma0_y);
        printf("Input the number of y grid points num_y_pts\n");
        scanf("%d", num_y_pts);
        printf("num_y_pts = %d\n", *num_y_pts);
        printf("Input the left end y_coordinate y_left\n");
        scanf("%lf", y_left);
        printf("y_left = %lf\n", *y_left);
        printf("Input the right end y-coordinate y_right0\n");
        scanf("%lf", y_right);
        printf("y_right = %lf\n", *y_right);
        printf("input the y_direction frequency y_freq\n");
        scanf("%lf", y_freq);
        printf("y_freq = %lf\n", *y_freq);
        input_params_array[5] = *sigma0_y;
        input_params_array[6] = (double) *num_y_pts;
        input_params_array[7] = *y_left;
        input_params_array[8] = *y_right;
        input_params_array[9] = *y_freq;

        //z_dir
        printf("z direction parameters:\n");
        printf("Input the IC scaling factor sigma0_z \n");
        scanf("%lf", sigma0_z);
        printf("sigma0_z = %lf\n", *sigma0_z);
        printf("Input the number of z grid points num_z_pts\n");
        scanf("%d", num_z_pts);
        printf("num_z_pts = %d\n", *num_z_pts);
        printf("Input the left end z_coordinate z_left\n");
        scanf("%lf", z_left);
        printf("z_left = %lf\n", *z_left);
        printf("Input the right end z-coordinate z_right0\n");
        scanf("%lf", z_right);
        printf("z_right = %lf\n", *z_right);
        printf("input the z_direction frequencz z_freq\n");
        scanf("%lf", z_freq);
        printf("z_freq = %lf\n", *z_freq);
        input_params_array[10] = *sigma0_z;
        input_params_array[11] = (double) *num_z_pts;
        input_params_array[12] = *z_left;
        input_params_array[13] = *z_right;
        input_params_array[14] = *z_freq;

        //General params
        printf("Input the number of temporal steps num_t_pts\n");
        scanf("%d", num_t_pts);
        printf("num_t_pts = %d\n", *num_t_pts);
        printf("Input the temporal step size dt\n");
        scanf("%lf", dt);
        printf("dt = %lf\n", *dt);
        printf("Enter the number of iterations to use to approximate the potential\n");
        scanf("%d", n_iter);
        printf("n_iter = %d\n", *n_iter);
        if (*n_iter == 0) {
            printf("assuming n_iter=0 means no extra iterations, setting n_iter=1");
            *n_iter = 1;
        }
        if (*n_iter < 0) {
            printf("n_iter < 1 illegal, exiting\n");
            return -1;
        }
        printf("enter the parameter value g\n");
        scanf("%lf", g_coeff);
        printf("g = %lf\n", *g_coeff);
        input_params_array[15] = (double) *num_t_pts;
        input_params_array[16] = *dt;
        input_params_array[17] = (double) *n_iter;
        input_params_array[18] = *g_coeff;
    }
    MPI_Bcast(input_params_array, 19, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    //Reassign to individual values
    *sigma0_x = input_params_array[0];
    *num_x_pts_global = (int) input_params_array[1];
    *x_left_global = input_params_array[2];
    *x_right = input_params_array[3];
    *x_freq = input_params_array[4];
    // y direction
    *sigma0_y = input_params_array[5];
    *num_y_pts = (int) input_params_array[6];
    *y_left = input_params_array[7];
    *y_right = input_params_array[8];
    *y_freq = input_params_array[9];
    // z direction
    *sigma0_z = input_params_array[10];
    *num_z_pts = (int) input_params_array[11];
    *z_left = input_params_array[12];
    *z_right = input_params_array[13];
    *z_freq = input_params_array[14];
    // General params
    *num_t_pts = (int) input_params_array[15];
    *dt = input_params_array[16];
    *n_iter = (int) input_params_array[17];
    *g_coeff = input_params_array[18];

    return 0;
}

/*Abstraction of the output produced at every time step. use the new psi and numerically
 * integrate to determine the new norm, standard deviation, average displacement and print to console
 * */
void PARALLEL_MPI_produce_output(int rank, int n_t, double t_curr, double complex *z_psi_new,
        int num_x_pts_rank, int num_y_pts, int num_z_pts, double x_left_rank, double y_left,
        double dx, double dy, double dz,
        double* reduction_array_local, double* reduction_array_result) {
    int j_z, j_y, j_x, index;
    double x_curr, y_curr, mod_psi_curr;
    double norm = 0.0, local_norm = 0.0;
    //Accumulation quantities across all ranks
    double x_average = 0, x_squared_average = 0, sqrtvar_x;
    double y_average = 0, y_squared_average = 0, sqrtvar_y;
    // local accumulation on this rank
    double x_average_local = 0, x_squared_average_local = 0;
    double y_average_local = 0, y_squared_average_local = 0;

    const double dv = dx * dy * dz;
    //Note we using x*x rather than pow() as this should be a little quicker for integer powers

    #pragma omp parallel for \
            reduction(+:local_norm, x_average_local, \
            x_squared_average_local, y_average_local, y_squared_average_local)\
            private (x_curr, y_curr, j_x, j_y, j_z, index, mod_psi_curr)
    for (j_z = 0; j_z < num_z_pts; j_z++) {
        for (j_y = 0; j_y < num_y_pts; j_y++) {
            y_curr = y_left + dy * (1 + j_y);
            for (j_x = 0; j_x < num_x_pts_rank; j_x++) {
                x_curr = x_left_rank + dx * (1 + j_x);

                index = get_1d_index(j_x, j_y, j_z, num_x_pts_rank, num_y_pts);
                mod_psi_curr = cabs(z_psi_new[index]) * cabs(z_psi_new[index]);

                local_norm += dv * mod_psi_curr;
                x_average_local += dv * x_curr * mod_psi_curr;
                x_squared_average_local += dv * x_curr * x_curr * mod_psi_curr;
                y_average_local += dv * y_curr * mod_psi_curr;
                y_squared_average_local += dv * y_curr * y_curr * mod_psi_curr;
            }
        }
    }
    //Put all reduction quantities into a single array for less communication overhead
    reduction_array_local[0] = local_norm;
    reduction_array_local[1] = x_average_local;
    reduction_array_local[2] = x_squared_average_local;
    reduction_array_local[3] = y_average_local;
    reduction_array_local[4] = y_squared_average_local;

    MPI_Reduce(reduction_array_local, reduction_array_result, 5, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    //print output, only on the root rank (only rank to receive results)
    if (rank == ROOT) {
        norm = reduction_array_result[0];
        x_average = reduction_array_result[1];
        x_squared_average = reduction_array_result[2];
        y_average = reduction_array_result[3];
        y_squared_average = reduction_array_result[4];

        sqrtvar_x = sqrt(x_squared_average - (x_average * x_average));
        sqrtvar_y = sqrt(y_squared_average - (y_average *y_average));

        printf("%4d %6.4lf %12.8lf %12.8lf %12.8lf %12.8lf %12.8lf %12.8lf %12.8lf\n",
               n_t, t_curr, norm, x_average, x_squared_average, sqrtvar_x,
               y_average, y_squared_average, sqrtvar_y);
    }
}

int main(int argc, char **argv) {
//    omp_set_max_active_levels(2);

    // MPI init
    int result, rank, num_mpi_ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &result);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_mpi_ranks);

    int num_omp_threads = omp_get_max_threads();
    root_print(rank, "Code set to run with %d MPI nodes and %d openMP threads.\n",
            num_mpi_ranks, num_omp_threads);
    rank_print(rank, "DEBUG: Rank alive\n");

    //Variables
    int ii, j_x, j_y, j_z, n_t; // loop iterators
    // input params
    int num_x_pts_global, num_y_pts, num_z_pts, num_t_pts, n_iter, index;
    double x_left_global, x_right, sigma0_x, x_freq;
    double y_left, y_right, sigma0_y, y_freq;
    double z_left, z_right, sigma0_z, z_freq;

    double dt, g_coeff;

    //allocate small fixed size arrays for output reduction calls to avoid allocation in loop
    double output_reduction_array_local[5], output_reduction_array_result[5];

    // Read input from file redirect, echo as manual error checking
    int ierr = PARALLEL_MPI_read_inputs(rank, &num_x_pts_global, &num_y_pts, &num_z_pts, &num_t_pts, &n_iter,
            &x_left_global, &x_right, &sigma0_x, &x_freq, &y_left, &y_right, &sigma0_y, &y_freq,
                             &z_left, &z_right, &sigma0_z, &z_freq, &dt, &g_coeff);
    if (ierr <0){
        MPI_Finalize();
        return -1;
    }
    //Compute dx
    //     note +1 since we have n+1 in order to have n internal grid points
    const double dx = (x_right - x_left_global) / (num_x_pts_global + 1);
    const double dy = (y_right - y_left) / (num_y_pts + 1);
    const double dz = (z_right - z_left) / (num_z_pts + 1);
    const double dv = dx * dy * dz;

    // setup rank size slices:
    // Give ranks array lengths they need
    int num_x_pts_rank = num_x_pts_global / num_mpi_ranks;
    int standard_num_x_pts_rank = num_x_pts_rank; // doesn't include overflow points
    if (rank == num_mpi_ranks - 1) { // last rank gets overflow points
        num_x_pts_rank += num_x_pts_global % num_mpi_ranks;
    }
    double x_left_rank = x_left_global + standard_num_x_pts_rank * rank * dx;
    rank_print(rank, "has %d x_points associated\n", num_x_pts_rank);

    int num_pts_3d = num_x_pts_rank * num_y_pts * num_z_pts;


    // vector for old position time, new time
    // Note these are the only full 3D vectors, rest are 1D for local solves
    double complex * z_psi_old = malloc(num_pts_3d * sizeof(double complex));
    double complex * z_psi_new = malloc(num_pts_3d * sizeof(double complex));

    double z_curr, y_curr, x_curr;

    // Set up I.C. and  compute original norm
    double norm = 0.0, local_norm = 0.0;
    #pragma omp parallel for reduction(+:local_norm) private (z_curr, x_curr, y_curr, j_x, j_y, j_z)
    for (j_z = 0; j_z < num_z_pts; j_z++) {
        z_curr = z_left + dz * (1 + j_z); // +1 for interior point from the start
        for (j_y = 0; j_y < num_y_pts; j_y++) {
            y_curr = y_left + dy * (1 + j_y);
            for (j_x = 0; j_x < num_x_pts_rank; j_x++) {
                x_curr = x_left_rank + dx * (1 + j_x);
                //note use fabs not cabs as ic function is real
                local_norm += dv * pow(fabs(
                        ic_function(x_curr, y_curr, z_curr, sigma0_x)), 2.0);
            }
        }
    }
    MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    root_print(rank, "Orig norm: %.14lf\n", norm);

    local_norm = 0.0; // reset for fresh accumulation

//     re-normalise wave function so norm is 1
    #pragma omp parallel for reduction(+:local_norm) private (z_curr, x_curr, y_curr, j_x, j_y, j_z, index)
    for (j_z = 0; j_z < num_z_pts; j_z++) {
        z_curr = z_left + dz * (1 + j_z);
        for (j_y = 0; j_y < num_y_pts; j_y++) {
            y_curr = y_left + dy * (1 + j_y);
            for (j_x = 0; j_x < num_x_pts_rank; j_x++) {
                x_curr = x_left_rank + dx * (1 + j_x);
                index = get_1d_index(j_x, j_y, j_z, num_x_pts_rank, num_y_pts);
                z_psi_old[index] = ic_function(x_curr, y_curr, z_curr, sigma0_x) / sqrt(norm);
                local_norm += dv * pow(cabs(z_psi_old[index]), 2.0);
            }
        }
    }
    MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    root_print(rank, "Rescaled norm: %lf\n", norm);

    //avoid big memory issues of reallocating vectors - just have vector of largest 1d size
    // note this is rank dependent result (but will not differ by much per rank - just larger on final one
    int max_1d_pts = three_num_max(num_x_pts_rank, num_y_pts, num_z_pts);

    double complex *z_psi_implicit, *z_psi_explicit, *z_psi_old_local_1d;

    // Make 1D arrays with space for all threads to work indepdendently (using same array)
    // this means we can preallocate outside loops and not have to malloc and free all the time
    int space_for_threads = num_omp_threads * max_1d_pts;
    z_psi_old_local_1d = malloc(sizeof(double complex) * space_for_threads);
    z_psi_implicit     = malloc(sizeof(double complex) * space_for_threads);
    z_psi_explicit     = malloc(sizeof(double complex) * space_for_threads);
    //Implicit side matrix memory allocation
    double complex *za_x_off_diag, *za_y_off_diag, *za_z_off_diag, *za_main_diag, *z_workspace_vec;
    z_workspace_vec    = malloc(sizeof(double complex) * space_for_threads);
    za_x_off_diag      = malloc(sizeof(double complex) * space_for_threads);
    za_y_off_diag      = malloc(sizeof(double complex) * space_for_threads);
    za_z_off_diag      = malloc(sizeof(double complex) * space_for_threads);
    za_main_diag       = malloc(sizeof(double complex) * space_for_threads);
    // note x_lower and upper are the same so reduce to an off_diag vector

    // constant factors in Finite difference scheme (alpha reduced excludes /dv^2 since dv varies spatially
    double complex z_alpha_reduced = I * (double complex) (0.25 * dt);
    double complex z_beta = (double complex) (0.5 * dt) * I;
    double z_alpha_x_contrib = 1.0 / (dx * dx);
    double z_alpha_y_contrib = 1.0 / (dy * dy);
    double z_alpha_z_contrib = 1.0 / (dz * dz);

    //matrix vectors initialisation (note initialise values across memory for all threads
    for (ii = 0; ii < max_1d_pts * num_omp_threads; ii++) {
        za_x_off_diag[ii] = -z_alpha_reduced * z_alpha_x_contrib;
        za_y_off_diag[ii] = -z_alpha_reduced * z_alpha_y_contrib;
        za_z_off_diag[ii] = -z_alpha_reduced * z_alpha_z_contrib;
        // note main has to be iteratively updated as potential dependent
    }

    // t=0 output
    root_print(rank,"n    t_n      norm(t)     <x>(t)        <x^2>(t)  "
                    "   stddev_x(t)    <y>(t)        <y^2>(t)     stddev_y(t) \n");
    root_print(rank, "%4d %6.4lf %12.8lf %12.8lf %12.8lf\n", 0, 0.0, norm, 0.0, 0.0);
//    main loops
    for (n_t = 1; n_t < num_t_pts; n_t++) {
        double t_curr = n_t * dt;

        // X dir operator advancement
        for (j_z = 0; j_z < num_z_pts; j_z++) {
            for (j_y = 0; j_y < num_y_pts; j_y++) {
                // Build 1 D vector for solve in 1D for X
                #pragma omp parallel for
                for (j_x = 0; j_x < num_x_pts_rank; j_x++) {
                    z_psi_old_local_1d[j_x] = z_psi_old[get_1d_index(j_x, j_y, j_z, num_x_pts_rank, num_y_pts)];
                }
                //Declare recv values so that we can recv pointers from communication function
                double complex l_recv_buf = 0.0, r_recv_buf = 0.0; // note default init is required for edge ranks
                MPI_Request recv_reqs_lhs[2], send_reqs_lhs[2];
                PARALLEL_MPI_exchange_boundaries(rank, num_mpi_ranks, &l_recv_buf, &r_recv_buf,
                                                 &z_psi_old_local_1d[0], &z_psi_old_local_1d[num_x_pts_rank - 1],
                                                 1, recv_reqs_lhs, send_reqs_lhs);

                PARALLEL_MPI_check_waits(recv_reqs_lhs, rank, num_mpi_ranks);
                PARALLEL_MPI_check_waits(send_reqs_lhs, rank, num_mpi_ranks);


                // evolution solution contained in z_psi_implicit
                one_dim_evolution(rank, num_mpi_ranks, z_psi_old_local_1d, z_psi_explicit, z_psi_implicit,
                                  z_workspace_vec, za_main_diag, za_x_off_diag, za_x_off_diag,
                                  num_x_pts_rank, g_coeff, x_left_rank, dx, z_alpha_reduced * z_alpha_x_contrib,
                                  z_beta, 1.0 / 3.0, x_freq, l_recv_buf, r_recv_buf, n_iter);
                //contributes 1/3 of total potential for symmetry

                // copy operation to put matrix solution into right array indices -
                #pragma omp parallel for private(j_x, index)
                for (j_x = 0; j_x < num_x_pts_rank; j_x++) {
                    index = get_1d_index(j_x, j_y, j_z, num_x_pts_rank, num_y_pts);
                    z_psi_new[index] = z_psi_implicit[j_x];
                }
            }
        }

        // Wrap y and z evolution in same parallel region to reduce overhead of entering and leaving
        #pragma omp parallel
        {
            // Each OMP thread gets access to own partition of master arrays
            int thread_offset = omp_get_thread_num() * max_1d_pts;

            // Y Dir operator advancement
            #pragma  omp for private( j_x, j_y, j_z)
            for (j_z = 0; j_z < num_z_pts; j_z++) {
                for (j_x = 0; j_x < num_x_pts_rank; j_x++) {
                    // Build 1 D vector for solve in 1D
                    for (j_y = 0; j_y < num_y_pts; j_y++) {
                        z_psi_old_local_1d[j_y + thread_offset] = z_psi_new[get_1d_index(j_x, j_y, j_z, num_x_pts_rank,
                                                                                         num_y_pts)];
                    }
                    // evolution solution contained in z_psi_implicit
                    one_dim_evolution(SERIAL_NO_RANK, SERIAL_NO_RANK, &z_psi_old_local_1d[thread_offset],
                                      &z_psi_explicit[thread_offset], &z_psi_implicit[thread_offset],
                                      &z_workspace_vec[thread_offset], &za_main_diag[thread_offset],
                                      &za_y_off_diag[thread_offset], &za_y_off_diag[thread_offset],
                                      num_y_pts, g_coeff, y_left, dy,
                                      z_alpha_reduced * z_alpha_y_contrib, z_beta, 1.0 / 3.0, y_freq,
                                      0.0, 0.0, n_iter);

                    // copy operation to update global 3d storage of soln
                    for (j_y = 0; j_y < num_y_pts; j_y++) {
                        index = get_1d_index(j_x, j_y, j_z, num_x_pts_rank, num_y_pts);
                        z_psi_new[index] = z_psi_implicit[j_y + thread_offset];
                    }
                }
            }

            // z dir operator advancement
            #pragma omp for private(j_y, j_x, j_z)
            for (j_y = 0; j_y < num_y_pts; j_y++) {
                for (j_x = 0; j_x < num_x_pts_rank; j_x++) {
                    // Build 1 D vector for solve in 1D
                    for (j_z = 0; j_z < num_z_pts; j_z++) {
                        index = get_1d_index(j_x, j_y, j_z, num_x_pts_rank, num_y_pts);
                        z_psi_old_local_1d[thread_offset + j_z] = z_psi_new[index];
                    }
                    // evolution solution contained in z_psi_implicit
                    one_dim_evolution(SERIAL_NO_RANK, SERIAL_NO_RANK,
                                      &z_psi_old_local_1d[thread_offset], &z_psi_explicit[thread_offset],
                                      &z_psi_implicit[thread_offset], &z_workspace_vec[thread_offset],
                                      &za_main_diag[thread_offset], &za_z_off_diag[thread_offset],
                                      &za_z_off_diag[thread_offset],
                                      num_z_pts, g_coeff, z_left, dz, z_alpha_reduced * z_alpha_z_contrib,
                                      z_beta, 1.0 / 3.0, z_freq, 0.0, 0.0, n_iter);

                    // copy operation to update global 3d storage of soln
                    for (j_z = 0; j_z < num_z_pts; j_z++) {
                        index = get_1d_index(j_x, j_y, j_z, num_x_pts_rank, num_y_pts);
                        z_psi_new[index] = z_psi_implicit[j_z + thread_offset];
                    }
                }
            }

        }

        PARALLEL_MPI_produce_output(rank, n_t, t_curr, z_psi_new, num_x_pts_rank, num_y_pts, num_z_pts,
                                    x_left_rank, y_left, dx, dy, dz, output_reduction_array_local, output_reduction_array_result);
        // Pointer update step for next time step
        double complex *z_psi_temp = z_psi_old;
        z_psi_old = z_psi_new; // put new values into old for next time step
        z_psi_new = z_psi_temp;

    } //End time loop

    //Free memory
    free(z_psi_new);
    free(z_psi_old);

    free(z_psi_implicit);
    free(z_psi_explicit);
    free(z_psi_old_local_1d);
    free(z_workspace_vec);
    free(za_x_off_diag);
    free(za_y_off_diag);
    free(za_z_off_diag);
    free(za_main_diag);


    MPI_Finalize();
    exit(EXIT_SUCCESS);
}