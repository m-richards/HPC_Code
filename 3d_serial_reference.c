/* 3D solver for the gross-pitaveski equation in natural units.
Note that this code is conceptually similar to the hybrid code, but this
file is less well commented.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>

#define PI 3.1415926535

// General initial condition function called by script, delgates to a particular ic function
double ic_function(double x, double y, double z, double sigma0) {
    // return gross_pitaevski_ic(x, x_g, sigma0);
    return (sqrt(1.0 / (sqrt(PI) * sigma0)) * exp((-x * x - y * y - z * z) / (2.0 * pow(sigma0, 2.0))));
}


// external potential in one spatial dimension
double linear_potential_func(double x, double freq){
    return 0.5 * freq*freq*x*x;
}

/* Tridiagonal matrix solver. complex variable modification of code from Michael Bromley
 * under copyleft permissions given to students in PHYS3071. Originally based on code
 * From numerical recipies. gam is an array of length n to be used as memory workspace.
 * solution to system in xsoln.
 */
int tridag_cplx(double complex adiag[], double complex alower[], double complex aupper[],
                double complex rrhs[], double complex xsoln[], int n, double complex *gam) {
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

void build_rhs_explicit(const double complex *z_psi_old_l, double complex *z_psi_explicit_l,
                        int num_x_pts, double x_left, double dx, double complex z_alpha, double complex z_beta,
                        double g_coeff, double potential_scaling, double dimensional_frequency, double left_bc,
                        double right_bc) {
    double x_curr;
    double complex potential_current, potential_future, z_rho_left, z_rho_right;
    // Determine which direction we are solving in and setup variables so pointers
    // pass to potential function correctly for (x, y , z) ordering
    // Setup explicit side solution
    for (int j_x = 0; j_x < num_x_pts; j_x++) {
        x_curr = x_left + dx * (1 + j_x);

        //reference to point left and right of current [j_x, t] point
        if (j_x == 0) {
            z_rho_left = 0.0;
        } else {
            z_rho_left = z_psi_old_l[j_x - 1];
        }
        if (j_x == num_x_pts - 1) {
            z_rho_right = 0.0;
        } else {
            z_rho_right = z_psi_old_l[j_x + 1];
        }
        //Calculate current potential for t_curr and t_curr+dt (iterative estimate)
        potential_current = linear_potential_func(x_curr, dimensional_frequency) + potential_scaling * (
                g_coeff * pow(cabs(z_psi_old_l[j_x]), 2.0));

        z_psi_explicit_l[j_x] = ((double complex) 1.0
                                 - (double complex) 2.0 * z_alpha - z_beta * potential_current
                                ) * z_psi_old_l[j_x]
                                + z_alpha * (double complex) (z_rho_left + z_rho_right);

    }

}
// not z_psi_future_approx is the prev iter vec
void build_new_psi_implicit_soln(double complex *za_main_diag_l, double complex * za_lower_diag, double complex *za_upper_diag,
                double complex *z_psi_future_approx, double complex * z_psi_explicit, double complex *z_psi_implicit,
                double complex *z_workspace_vec,
                int  num_x_global_pts,  double x_left,  double dx, double g_coeff, double complex z_alpha, double complex z_beta,
                double potential_scaling, double dimensional_frequency){
    double x_curr;
    double complex potential_future;
    for (int j_x = 0; j_x < num_x_global_pts; j_x++) {
        x_curr = x_left + dx * (1 + j_x);

        potential_future = linear_potential_func(x_curr, dimensional_frequency) + potential_scaling * (
                g_coeff * pow(cabs(z_psi_future_approx[j_x]), 2.0));

        // setup main diag based on iterative approximation to future potential
        za_main_diag_l[j_x] = (double complex) 1.0
                              + (double complex) (2.0) * z_alpha
                              + z_beta * potential_future;

    }

    // implicit side solve
    int ierr = tridag_cplx(za_main_diag_l, za_lower_diag, za_upper_diag,
                           z_psi_explicit, z_psi_implicit, num_x_global_pts,
                           z_workspace_vec);
    if (ierr != 0) {
        printf("Matrix inversion failed. Exiting\n");
        exit(EXIT_FAILURE);
    }
}




/* Solver for a single time step of the 1D (nonlinear) schrodinger equation
 * Takes in vector of previous time step values, vector to store rhs matrix evaluation in,
 * vector to store result of matrix inversion in
 * vector containing iterated estimate of z_psi_new
 * memory workspace vector
 * main, upper and lower diagonals of lhs matrix
 * length of vectors, nonlinea GPE parameter g, leftmost_x_point, x step size,
 * matrix element collected coefficients alpha and beta,
 * struct containing the current position for the 2 fixed dimensions outside this routine.
 * parameter to inform which dir is being solved for (to support non symmetric potentials in future)
 * scaling factor applied to potential*/
void one_dim_evolution(double complex *z_psi_old_l, double complex *z_psi_explicit_l,
                       double complex *z_psi_implicit_l, double complex *z_workspace_vec_l,
                       double complex *za_main_diag_l, double complex *za_upper_diag_l, double complex *za_lower_diag_l,
                       int num_x_global_pts, double g_coeff,
                       double x_global_left, double dx, double complex z_alpha, double complex z_beta,
                        double potential_scaling, double dimensional_frequency, double left_bc, double right_bc, int n_iter) {

    // Setup explicit side solution
    build_rhs_explicit(z_psi_old_l, z_psi_explicit_l, num_x_global_pts, x_global_left, dx, z_alpha, z_beta, g_coeff,
            potential_scaling, dimensional_frequency, left_bc, right_bc);

    // iter vec starts as old time ve. note pointer is to reflect that this serves new purpose as
    double complex * z_psi_iter_l = z_psi_old_l; // we no longer need local old and its values coincide with the
    // initialisation for the iter.


    for (int k_iter = 0; k_iter < n_iter; k_iter++) {

        build_new_psi_implicit_soln(za_main_diag_l, za_lower_diag_l, za_upper_diag_l, z_psi_iter_l, z_psi_explicit_l,
                                    z_psi_implicit_l, z_workspace_vec_l, num_x_global_pts, x_global_left, dx, g_coeff,
                                    z_alpha, z_beta,
                                    potential_scaling, dimensional_frequency);
        double complex* z_psi_temp = z_psi_iter_l;
        z_psi_iter_l = z_psi_implicit_l;
        z_psi_implicit_l = z_psi_temp;
    }
}

int main() {

    //Variables
    int ii, j_x, j_y, j_z, n_t; // loop iterators
    int num_x_pts, num_y_pts, num_z_pts, num_t_pts, n_iter; //values to be read in
//    double x_global_left, x_right_global;
    double x_left, x_right, sigma0_x, x_freq;
    double y_left, y_right, sigma0_y, y_freq;
    double z_left, z_right, sigma0_z, z_freq;

    double dt, g_coeff;

    // Read input from file redirect, echo as manual error checking
    //X dir
    printf("X direction parameters\n");
    printf("Input the IC scaling factor sigma0_x \n");
    scanf("%lf", &sigma0_x);
    printf("sigma0_x = %lf\n", sigma0_x);
    printf("Input the number of x grid points num_x_pts\n");
    scanf("%d", &num_x_pts);
    printf("num_x_pts = %d\n", num_x_pts);
    printf("Input the left end x_coordinate x_left\n");
    scanf("%lf", &x_left);
    printf("x_left = %lf\n", x_left);
    printf("Input the right end x-coordinate x_right0\n");
    scanf("%lf", &x_right);
    printf("x_right = %lf\n", x_right);
    printf("input the x_direction frequency x_freq\n");
    scanf("%lf", &x_freq);
    printf("x_freq = %lf\n", x_freq);

    //y_dir
    printf("y direction parameters\n");
    printf("Input the IC scaling factor sigma0_y \n");
    scanf("%lf", &sigma0_y);
    printf("sigma0_y = %lf\n", sigma0_y);
    printf("Input the number of y grid points num_y_pts\n");
    scanf("%d", &num_y_pts);
    printf("num_y_pts = %d\n", num_y_pts);
    printf("Input the left end y_coordinate y_left\n");
    scanf("%lf", &y_left);
    printf("y_left = %lf\n", y_left);
    printf("Input the right end y-coordinate y_right0\n");
    scanf("%lf", &y_right);
    printf("y_right = %lf\n", y_right);
    printf("input the y_direction frequency y_freq\n");
    scanf("%lf", &y_freq);
    printf("y_freq = %lf\n", y_freq);

    //z_dir
    printf("z direction parameters\n");
    printf("Input the IC scaling factor sigma0_z \n");
    scanf("%lf", &sigma0_z);
    printf("sigma0_z = %lf\n", sigma0_z);
    printf("Input the number of z grid points num_z_pts\n");
    scanf("%d", &num_z_pts);
    printf("num_z_pts = %d\n", num_z_pts);
    printf("Input the left end z_coordinate z_left\n");
    scanf("%lf", &z_left);
    printf("z_left = %lf\n", z_left);
    printf("Input the right end z-coordinate z_right0\n");
    scanf("%lf", &z_right);
    printf("z_right = %lf\n", z_right);
    printf("input the z_direction frequencz z_freq\n");
    scanf("%lf", &z_freq);
    printf("z_freq = %lf\n", z_freq);


    //General params
    printf("Input the number of temporal steps num_t_pts\n");
    scanf("%d", &num_t_pts);
    printf("num_t_pts = %d\n", num_t_pts);
    printf("Input the temporal step size dt\n");
    scanf("%lf", &dt);
    printf("dt = %lf\n", dt);
    printf("Enter the number of iterations to use to approximate the potential\n");
    scanf("%d", &n_iter);
    printf("n_iter = %d\n", n_iter);
    if (n_iter ==0){
        printf("assuming n_iter=0 means no extra iterations, setting n_iter=1");
        n_iter =1;
    }
    if (n_iter < 0) {
        printf("n_iter < 1 illegal, exiting\n");
    }
    printf("enter the parameter value g\n");
    scanf("%lf", &g_coeff);
    printf("g = %lf\n", g_coeff);

    //     note +1 since we have n+1 in order to have n internal grid points
    
    const double dx = (x_right - x_left) / (num_x_pts + 1);
    const double dy = (y_right - y_left) / (num_y_pts + 1);
    const double dz = (z_right - z_left) / (num_z_pts + 1);

    const double dv = dx*dy*dz;

    double complex *z_psi_old, *z_psi_new;

    // Note these are the only full 3D vectors, rest are locally 1D for matrices
    int num_pts_3d = num_x_pts * num_y_pts * num_z_pts;
    // vector for old position time, new time, old iteration time if iterating
    z_psi_old = malloc(num_pts_3d * sizeof(double complex));
    z_psi_new = malloc(num_pts_3d * sizeof(double complex));

    double complex *z_psi_implicit, *z_psi_explicit, *z_psi_old_local_1d, *z_psi_iter_local_1d;
    // vectors to pass to the 1 dimensional problem
    int max_1d_pts; //save big memory issues of reallocating vectors - just have vector of largest 1d size
    if(num_x_pts >num_y_pts){ // low effort max
        max_1d_pts = num_x_pts;
    } else {
        max_1d_pts = num_y_pts;
    }
    if (max_1d_pts <num_z_pts){
        max_1d_pts = num_z_pts;
    }
    printf("max 1d pts = %d\n", max_1d_pts);
    
    z_psi_iter_local_1d = malloc(max_1d_pts * sizeof(double complex));
    z_psi_old_local_1d = malloc(max_1d_pts * sizeof(double complex));
    z_psi_implicit = malloc(max_1d_pts * sizeof(double complex));
    z_psi_explicit = malloc(max_1d_pts * sizeof(double complex));

    double z_curr, y_curr, x_curr;

    // Set up I.C. and  compute original norm
    double norm = 0.0;
    for (j_z = 0; j_z < num_z_pts; j_z++) {
        z_curr = z_left + dz * (1 + j_z);
        for (j_y = 0; j_y < num_y_pts; j_y++) {
            y_curr = y_left + dy * (1 + j_y);
            for (j_x = 0; j_x < num_x_pts; j_x++) {
                // +1 for ic not in interior
                x_curr = x_left + dx * (1 + j_x);
                norm += dx*dy*dz * pow(cabs(
                        ic_function(x_curr, y_curr, z_curr, sigma0_x)), 2.0);
            }
        }
    }
    printf("Orig norm: %.14lf\n", norm);

    double norm_new = 0.0;

//     re-normalise wave function so norm is 1
    for (j_z = 0; j_z < num_z_pts; j_z++) {
        z_curr = z_left + dz * (1 + j_z);
        for (j_y = 0; j_y < num_y_pts; j_y++) {
            y_curr = y_left + dy * (1 + j_y);
            for (j_x = 0; j_x < num_x_pts; j_x++) {
                x_curr = x_left + dx * (1 + j_x);
                int index = get_1d_index(j_x, j_y, j_z, num_x_pts, num_y_pts);
                z_psi_old[index] = ic_function(x_curr, y_curr, z_curr, sigma0_x) / sqrt(norm);
                norm_new += dx*dy*dz * pow(cabs(z_psi_old[index]), 2.0);
            }
        }
    }
    printf("Rescaled norm: %lf\n", norm_new);

    //Implicit side matrix memory allocation
    double complex *za_x_off_diag, *za_y_off_diag, *za_z_off_diag, *za_main_diag, *z_workspace_vec;
    z_workspace_vec = malloc(max_1d_pts * sizeof(double complex));
    // abuse the fact that lower and uppper diag are the same
    za_x_off_diag = malloc(max_1d_pts * sizeof(double complex));
    za_y_off_diag = malloc(max_1d_pts * sizeof(double complex));
    za_z_off_diag = malloc(max_1d_pts * sizeof(double complex));
    za_main_diag = malloc(max_1d_pts * sizeof(double complex));

    // constant factors (alpha reduced since it doesn't include the /dx^2 since dx varies
    double complex z_alpha_reduced = I * (double complex) (0.25 * dt);// / pow(dx_global, 2.0));
    double complex z_beta = (double complex) (0.5 * dt) * I;
    double z_alpha_x_contrib = 1.0/(dx * dx);
    double z_alpha_y_contrib = 1.0/(dy * dy);
    double z_alpha_z_contrib = 1.0/(dz * dz);

    //matrix initialisation
    for (ii = 0; ii < max_1d_pts; ii++) {
        za_x_off_diag[ii] = -z_alpha_reduced /pow(dx, 2.0);
        za_y_off_diag[ii] = -z_alpha_reduced / pow(dy, 2.0);
        za_z_off_diag[ii] = -z_alpha_reduced / pow(dz, 2.0);
        // note main has to be iteratively updated as potential dependent
    }

    // t=0 output
    printf("n    t_n      norm(t)     <x>(t)        <x^2>(t)     stddev_x(t) | error(t+1) |      <y>(t)        <y^2>(t)     stddev_y(t) \n");
    printf("%4d %6.4lf %12.8lf %12.8lf %12.8lf\n", 0,
           0.0, norm_new, 0.0, 0.0);
    const double BC_PLACEHOLDER = 0.0;
    //main loops
    for (n_t = 1; n_t < num_t_pts; n_t++) {
        double t_curr = n_t * dt;

            // X dir operator advancement
            for (j_z = 0; j_z < num_z_pts; j_z++) {
                for (j_y = 0; j_y < num_y_pts; j_y++) {

                    // Build 1 D vector for solve in 1D for X
                    for (j_x = 0; j_x < num_x_pts; j_x++) {
                        z_psi_old_local_1d[j_x] = z_psi_old[get_1d_index(j_x, j_y, j_z, num_x_pts, num_y_pts)];
                    }
                    one_dim_evolution(z_psi_old_local_1d, z_psi_explicit, z_psi_implicit, z_workspace_vec,
                                      za_main_diag, za_x_off_diag, za_x_off_diag, num_x_pts, g_coeff, x_left, dx,
                                      z_alpha_reduced * z_alpha_x_contrib, z_beta,
                                      1.0 / 3.0, x_freq, BC_PLACEHOLDER, BC_PLACEHOLDER, n_iter); //contributes 1/3 of total potential for symmetry
                    // solution contained in z_psi_implicit

                    // copy operation to put matrix solution into right array indices -
                    // values are only contiguous in x so we copy them explicitly
                    for (ii = 0; ii < num_x_pts; ii++) {
                        z_psi_new[get_1d_index(ii, j_y, j_z, num_x_pts, num_y_pts)] = z_psi_implicit[ii];
                    }
                }
            }

            // Y Dir operator advancement
            for (j_z = 0; j_z < num_z_pts; j_z++) {
                for (j_x = 0; j_x < num_x_pts; j_x++) {
                    // Build 1 D vector for solve in 1D
                    for (j_y = 0; j_y < num_y_pts; j_y++) {
                        z_psi_old_local_1d[j_y] = z_psi_new[get_1d_index(j_x, j_y, j_z, num_x_pts, num_y_pts)];

                    }
                    one_dim_evolution(z_psi_old_local_1d, z_psi_explicit, z_psi_implicit,
                                      z_workspace_vec, za_main_diag, za_y_off_diag, za_y_off_diag,
                                      num_y_pts, g_coeff, y_left, dy,
                                      z_alpha_reduced * z_alpha_y_contrib, z_beta, 1.0 / 3.0, y_freq,
                                      BC_PLACEHOLDER, BC_PLACEHOLDER, n_iter);
                    // solution contained in z_psi_implicit

                    // copy operation to put matrix solution into right array indices -
                    // values are only condtiguous in x so we copy them explicitly
                    for (ii = 0; ii < num_y_pts; ii++) {
                        z_psi_new[get_1d_index(j_x, ii, j_z, num_x_pts, num_y_pts)] = z_psi_implicit[ii];
                    }
                }
            }

            // z dir operator advancement
            for (j_y = 0; j_y < num_y_pts; j_y++) {
                for (j_x = 0; j_x < num_x_pts; j_x++) {
                    // do z implicitly
                    // Build 1 D vector for solve in 1D
                    for (j_z = 0; j_z < num_z_pts; j_z++) {
                        z_psi_old_local_1d[j_z] = z_psi_new[get_1d_index(j_x, j_y, j_z, num_x_pts, num_y_pts)];

                    }
                    one_dim_evolution(z_psi_old_local_1d, z_psi_explicit, z_psi_implicit, z_workspace_vec,
                                      za_main_diag, za_z_off_diag, za_z_off_diag, num_z_pts, g_coeff, z_left, dz,
                                      z_alpha_reduced * z_alpha_z_contrib,
                                      z_beta, 1.0 / 3.0, z_freq, BC_PLACEHOLDER, BC_PLACEHOLDER, n_iter);
                    // solution contained in z_psi_implicit

                    // copy operation to put matrix solution into right array indices -
                    // values are only condtiguous in x so we copy them explicitly
                    for (ii = 0; ii < num_z_pts; ii++) {
                        z_psi_new[get_1d_index(j_x, j_y, ii, num_x_pts, num_y_pts)] = z_psi_implicit[ii];
                    }
                }
            }

        // Computation of output quantities
        norm = 0;
        double x_average = 0, x_squared_average = 0, sqrtvar_x;
        double y_average = 0, y_squared_average = 0, sqrtvar_y;
        for (j_z = 0; j_z < num_z_pts; j_z++) {
            for (j_y = 0; j_y < num_y_pts; j_y++) {
                y_curr = y_left + dy * (1 + j_y);
                for (j_x = 0; j_x < num_x_pts; j_x++) {
                    x_curr = x_left + dx * (1 + j_x);
                    int index = get_1d_index(j_x, j_y, j_z, num_x_pts, num_y_pts);
                    double mod_psi_curr = pow(cabs(z_psi_new[index]), 2.0);
                    norm += dv * mod_psi_curr;
                    x_average += dv * x_curr * mod_psi_curr;
                    x_squared_average += dv * pow(x_curr, 2.0) * mod_psi_curr;
                    y_average += dv * y_curr * mod_psi_curr;
                    y_squared_average += dv * pow(y_curr, 2.0) * mod_psi_curr;
                    // iter error
                }
            }
        }

        sqrtvar_x = sqrt(x_squared_average - pow(x_average, 2.0));
        sqrtvar_y = sqrt(y_squared_average - pow(y_average, 2.0));
        //print output
        printf("%4d %6.4lf %12.8lf %12.8lf %12.8lf %12.8lf %12.8lf %12.8lf %12.8lf\n", n_t,
               t_curr, norm, x_average, x_squared_average, sqrtvar_x, y_average, y_squared_average,
               sqrtvar_y);

        double complex* z_psi_temp = z_psi_old;
        z_psi_old = z_psi_new; // put new values into old for next time step
        z_psi_new = z_psi_temp;

    } //End time loop

    //Free memory

    free(z_psi_new);
    free(z_psi_old);


    free(z_psi_implicit);
    free(z_psi_explicit);
    free(z_psi_old_local_1d);
    free(z_psi_iter_local_1d);

    free(z_workspace_vec);
    free(za_x_off_diag);
    free(za_y_off_diag);
    free(za_z_off_diag);
//
    free(za_main_diag);

    exit(EXIT_SUCCESS);
}