#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <ctime>
#define EPS 3.0e-14
#define MAXIT 10
#define   ZERO       1.0E-8
using namespace std;

//Defining functions for different calculations.
double integration_function(double x1, double y1, double z1, double x2, double y2, double z2, double alpha);
double integration_function_spherical(double r1, double theta_1, double phi_1, double r2, double theta_2, double phi_2);
double gauss_quadrature(int num_integration_points, double x1, double x2, double alpha);
double gauss_quadrature_spherical(int num_integration_points, double alpha);
double integration_function_monte_carlo(double r1, double theta_1, double phi_1, double r2, double theta_2, double phi_2);
void gauleg(double x1, double x2, double x[], double w[], int n);
void gauslag(double *x, double *w, int n, double alf);
double gammln( double xx);
double gaussian_deviate(long * idum);
double ran0(long *idum);
double* monte_carlo_spherical(int n);
double* monte_carlo(int n);
double* monte_carlo_parallel(int n);
bool test_gauslag();
bool test_gauleg_and_gauslag();

/*
 *Function gausleg, gauslag, gammln, gaussian_deviate and ran0 are taken from Morten Hjort-Jensen github. Theese
 * are written verisons of code from Numerical Recipes.
 */

//Defining functions for the different parts of the project.
void ex_a();
void ex_a_write_to_file();
void ex_b();
void ex_b_write_to_file();
void ex_c();
void ex_d();
void ex_e();

//Uncomment or comment the line below to do unit-testing.
//#define UNIT_TESTING
#ifndef UNIT_TESTING
//Uncomment the parts of the project that should be ran.
int main()
{
    //ex_a();
    //ex_b();
    //ex_c();
    //ex_d();
    //ex_e();
    return 0;
}
#else // UNIT_TESTING

int main()
{
    if (test_gauslag() && test_gauleg_and_gauslag())
    {
        cout << "Everything seems to be good";
    }
    else {
        cout << "Something is wrong";
    }
    return 0;
}
#endif //UNIT_TESTING

//Function for testing the gauslag function using a simple integral.
//Returns true if test is ok.
bool test_gauslag()
{

    int n = 6;

    double expected_value = 6;

    double *x = new double[n];
    double *w = new double[n];

    gauslag(x, w, n, 3);

    double integral_value = 0;

    for(int i = 0; i < n; i++)
    {
        integral_value += w[i];
    }

    //cout << integral_value << endl;
    //cout << expected_value << endl;

    if (abs(expected_value - integral_value) < 0.0001)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//Function for testing both the gauleg and gauslag function.
//Returns true if test is ok.
bool test_gauleg_and_gauslag()
{
    int n = 8;

    double expected_value = 12*acos(-1);

    double *x_r = new double[n];
    double *w_r = new double[n];

    gauslag(x_r,w_r,n,3);

    double *x_theta = new double[n];
    double *w_theta = new double[n];

    gauleg(0, 2*acos(-1), x_theta, w_theta, n);

    double integral_value = 0;

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            integral_value += w_r[i]*w_theta[j];
        }
    }

    if(abs(expected_value - integral_value) < 0.0001)
    {
        return true;
    }
    else
    {
        return false;
    }

}

/*
 * Function for solving part a of the project.
 * Uses Gauss-Legendre quadrature for solving the specified equation in cartesiean coordinates
 * and writes to file.
 */
void ex_a()
{
    double start_time = clock();
    double value = gauss_quadrature(35, -5, 5, 2);
    cout << "GaussQuad: " << value << "Exact value:"<< (5*3.14*3.14)/(16*16) <<endl;
    double end_time = clock();
    cout << "Time used: " << (end_time - start_time)/(CLOCKS_PER_SEC) << endl;
    cout << "Error: " << abs(value - (5*3.14*3.14)/(16*16)) << endl;
}

//Writes the values from ex_a to file.
void ex_a_write_to_file()
{
    ofstream myfile;
    myfile.open("Legendre.txt");
    int max_value = 20;
    myfile << "N, value, time";
    for(int i = 5; i <=max_value; i+=5)
    {
        double start_time = clock();
        double value = gauss_quadrature(i, -5, 5, 2);
        double end_time = clock();
        myfile << i << "," << value << "," << (end_time-start_time)/CLOCKS_PER_SEC << endl;
        //cout << "GaussQuad: " << value << "Exact value:"<< (5*3.14*3.14)/(16*16) <<endl;
    }
    myfile.close();

}

/*
 * Function for solving part b of the project. Uses Gauss-Laguerre quadrature for solving the radial part
 * and Gauss-Legendre for the angular parts. Solving the specified equation with spherical coordinates.
 */
void ex_b()
{
    double start_time = clock();
    double value = gauss_quadrature_spherical(30, 2);
    cout << "GaussQuad spherical: " << value << "Exact value: " << (5*3.14*3.14)/(16*16) << endl;
    double end_time = clock();
    cout << "Time used: " << (end_time - start_time)/(CLOCKS_PER_SEC) << endl;
    cout << "Error: " << abs(value - (5*3.14*3.14)/(16*16)) << endl;

}

//Writes the values from ex_b to file
void ex_b_write_to_file()
{
    ofstream myfile;
    myfile.open("Laguerre.txt");
    int max_value = 25;
    myfile << "N, value, time";
    for(int i = 5; i <=max_value; i+=5)
    {
        double start_time = clock();
        double value = gauss_quadrature_spherical(i, 2);
        double end_time = clock();
        myfile << i << "," << value << "," << (end_time-start_time)/CLOCKS_PER_SEC << endl;
        //cout << "GaussQuad: " << value << "Exact value:"<< (5*3.14*3.14)/(16*16) <<endl;
    }
    myfile.close();
}

/*
 * Function for solving part c of the project. Uses brute force monte carlo integration for
 * solving the specified equation with spherical coordinates for different values of N.
 * Writes the results to file.
 */
void ex_c()
{
    ofstream myfile;
    myfile.open("monte_carlo.txt");
    int N = 10;
    myfile << "N,value,variance,time" << endl;
    for(int i = 1; i < 9; i++)
    {
        double *values = monte_carlo(N);
        myfile << N << "," << values[0] << "," << values[1] << "," << values[2] << endl ;
        N = N*10;

    }
    myfile.close();
}

/*
 * Function for solving part d using monte carlo integration with importance sampling for
 * varying values of N.
 * Writes the results to file.
*/
void ex_d()
{
    ofstream myfile;
    myfile.open("monte_carlo_importance_sampling.txt");
    int N = 10;
    myfile << "N,value,variance,time" << endl;
    for(int i = 1; i < 9; i++)
    {
        double *values = monte_carlo_spherical(N);
        myfile << N << "," << values[0] << "," << values[1] << "," << values[2] << endl ;
        N = N*10;

    }
    myfile.close();
}

/*
 * Function for solving the integral using monte carlo integration with brute force.
 *
 * Input:
 * --------
 * n: Int, number of random points.
 *
 * Returns:
 * ---------
 * return_values: Vector [integral_value, variance, time]
*/
double *monte_carlo(int n)
{
    double x1,x2,y1,y2,z1,z2;

    //Integration interval. Can be choosen differently, but function seems to be zero further out.
    double a;
    double b;
    a = -5;
    b = 5;

    //
    double sum = 0; //The contribution from each random number.
    double sum_sigma = 0; //The contribution to the variance for each random number.
    double func_value;

    double *x = new double[n]; //Vector containing the function values for each loop.

    double start_time = clock();

    long idum = -1;

    //Loops over all the random numbers for each dimension.
    for(int i = 0; i < n; i++){
        x1 = a + (b-a)*ran0(&idum);
        y1 = a + (b-a)*ran0(&idum);
        z1 = a + (b-a)*ran0(&idum);
        x2 = a + (b-a)*ran0(&idum);
        y2 = a + (b-a)*ran0(&idum);
        z2 = a + (b-a)*ran0(&idum);
        //cout << x1 << y1 << z1 << x2 << y2 << z2 << endl;
        func_value = integration_function(x1,y1,z1,x2,y2, z2, 2);
        sum+=func_value;
        x[i] = func_value;

    }
    sum_sigma = 0;
    sum = sum/((double)n); //Calculates the mean of the function values. Will multiply by jacobi_det later.
    //Loops over each N and adds eachs points contribution to the variance sum.
    for(int i = 0; i < n; i++)
    {
        sum_sigma += (x[i] - sum)*(x[i] - sum);
    }
    double V = pow((b-a),6); //Jacobi determinant.
    double integral_value = V*sum;
    sum_sigma = V*sum_sigma/((double)n); //Variance
    double st_dev = sqrt(sum_sigma)/sqrt((double)n); //Standard deviation.

    double end_time = clock();

    double time_used = (end_time - start_time)/CLOCKS_PER_SEC;
    cout << "MC value brute force: " << integral_value << "    Exact value:" << (5*3.14*3.14)/(16*16) << endl;
    cout << "Variance MC brute force: " << sum_sigma << endl;
    cout << "Time used: " << time_used << " sec" << endl;
    cout << "Error: " << abs(integral_value - (5*3.14*3.14)/(16*16)) << endl;

    double *return_values = new double[3];
    return_values[0] = (integral_value);
    return_values[1] = sum_sigma;
    return_values[2] = time_used;

    return return_values;

}

/*
 * Function for solving the problem using monte carlo integration with importance sampling.
 *
 * Input:
 * --------
 * n: Int, number of random points.
 *
 * Returns:
 * ---------
 * return_values: Vector [integral_value, variance, time]
*/
double * monte_carlo_spherical(int n)
{
    double r1, r2, theta1, theta2, phi1, phi2;
    double jacobi_determinant = 4*pow(acos(-1), 4)/16.;

    double sum = 0;
    double sum_sigma = 0;
    double func_value;
    long idum = 3;
    double *x = new double[n];

    double start_time = clock();

    for(int i = 0; i < n; i++)
    {
        r1 = -0.25*log(1-ran0(&idum));
        r2 = -0.25*log(1-ran0(&idum));
        theta1 = acos(-1)*ran0(&idum);
        theta2 = acos(-1)*ran0(&idum);
        phi1 = 2*acos(-1)*ran0(&idum);
        phi2 = 2*acos(-1)*ran0(&idum);
        func_value = integration_function_monte_carlo(r1, theta1, phi1, r2, theta2, phi2);
        x[i] = func_value;
        sum += func_value;

    }
    sum_sigma = 0;
    sum = sum/((double)n);
    for(int i = 0; i < n; i++)
    {
        sum_sigma += (x[i] - sum)*(x[i] - sum);
    }
    sum_sigma = sum_sigma*jacobi_determinant/((double)n);
    double st_dev = sqrt(sum_sigma)/sqrt((double)n);

    double end_time = clock();

    cout << "Monte Carlo with importance sampling: " << (jacobi_determinant*sum) << "     Exact value: " << (5*3.14*3.14)/(16*16) << endl;
    cout << "Monte Carlo variance: " << sum_sigma << endl;
    cout << "Time used: " << (end_time - start_time)/CLOCKS_PER_SEC << endl;

    double *return_values = new double[3];
    return_values[0] = (jacobi_determinant*sum);
    return_values[1] = sum_sigma;
    return_values[2] = (end_time - start_time)/CLOCKS_PER_SEC;

    return return_values;

}

//Function for part e of the project. Writes the values to file, that is plotted using python.
void ex_e()
{
    ofstream myfile;
    myfile.open("monte_carlo_parallel.txt");
    int N = 10;
    myfile << "N,value,variance,time" << endl;
    for(int i = 1; i < 9; i++)
    {
        double *values = monte_carlo_parallel(N);
        myfile << N << "," << values[0] << "," << values[1] << "," << values[2] << endl ;
        N = N*10;

    }
    myfile.close();
}

/*
 * Function for solving the problem using monte carlo integration with importance sampling and parallel.
 * Input:
 * --------
 * n: Int, number of random points.
 *
 * Returns:
 * ---------
 * return_values: Vector, [integral_value, variance, time]
*/
double* monte_carlo_parallel(int n)
{
    double r1, r2, theta1, theta2, phi1, phi2;
    double jacobi_determinant = 4*pow(acos(-1), 4)/16.;

    double sum = 0;
    double sum_sigma = 0;
    double func_value;

    double *x = new double[n];

    double start_time = clock();
    int num_threads = omp_get_max_threads();
    cout << "Number of threads = " << num_threads << endl;
//Starts the threads. Default is to use every available thread, in my case 8.
#pragma omp parallel
    {
    long idum = omp_get_thread_num() + 100;
    cout << idum << endl;
//Have to use this line so that every loop is done exactly one time.
#pragma omp for reduction(+:sum)
    for(int i = 0; i < n; i++)
    {
        r1 = -0.25*log(1-ran0(&idum));
        r2 = -0.25*log(1-ran0(&idum));
        theta1 = acos(-1)*ran0(&idum);
        theta2 = acos(-1)*ran0(&idum);
        phi1 = 2*acos(-1)*ran0(&idum);
        phi2 = 2*acos(-1)*ran0(&idum);
        func_value = integration_function_monte_carlo(r1, theta1, phi1, r2, theta2, phi2);
        x[i] = func_value;
        sum += func_value;

    }
    }
    sum_sigma = 0;
    sum = sum/((double)n);
    for(int i = 0; i < n; i++)
    {
        sum_sigma += (x[i] - sum)*(x[i] - sum);
    }
    sum_sigma = sum_sigma*jacobi_determinant/((double)n);
    double st_dev = sqrt(sum_sigma)/sqrt(n);

    double end_time = clock();

    cout << "Monte Carlo with importance sampling and parallization: " << (jacobi_determinant*sum) << "     Exact value: " << (5*3.14*3.14)/(16*16) << endl;
    cout << "Monte Carlo variance: " << sum_sigma << endl;
    cout << "Time used: " << (end_time - start_time)/CLOCKS_PER_SEC << endl;

    double *return_values = new double[3];
    return_values[0] = (jacobi_determinant*sum);
    return_values[1] = sum_sigma;
    return_values[2] = (end_time - start_time)/CLOCKS_PER_SEC;

    return return_values;

}



/*
 * integration_function
 * Input:
 * ---------------
 * x1: Int, x-coordinate of first particle
 * y1: Int, y-coordinate of first particle
 * z1: Int, z-coordinate of first particle
 * x2: Int, x-coordinate of second particle
 * y2: Int, y-coordinate of second particle
 * z2: Int,  z-coordinate of second particle
 * alpha: Int.
 * Returns:
 * ----------------
 * double, function value at the point specified by input.
 *
 */
double integration_function(double x1, double y1, double z1, double x2, double y2, double z2, double alpha){
    double r1= sqrt(x1*x1 + y1*y1 + z1*z1);
    double r2= sqrt(x2*x2 + y2*y2 + z2*z2);
    double sep_dist = sqrt(pow((x1-x2),2) + pow((y1-y2), 2) + pow((z1-z2), 2));
    if(sep_dist < 0.000000000001){
        return 0;
    }
    else{
        return exp(-2*alpha*(r1+r2))/sep_dist;
    }
}

/*
 * Integration function with spherical coordinates for alpha = 2, where the weights are
 * accounted for.
 * If alpha changes, this function is no longer correct!!
 *
 * Returns:
 * ---------
 * double: f, is the function value at the given points, specified by r1, r2, theta_1, theta_2, phi_1 and phi_2
 */
double integration_function_spherical(double r1, double theta_1, double phi_1, double r2, double theta_2, double phi_2)
{
    double cos_beta = cos(theta_1)*cos(theta_2) + sin(theta_1)*sin(theta_2)*cos(phi_1-phi_2);
    double f = exp(-3*(r1+r2))*sin(theta_1)*sin(theta_2)/sqrt(r1*r1+r2*r2-2*r1*r2*cos_beta);
    if(abs(r1*r1+r2*r2-2*r1*r2*cos_beta) > ZERO)
        return f;
    else
        return 0;
}


/*
 * Function used in Monte Carlo methods.
 *
 * Returns:
 * --------
 * double: f, is the function value at the given points, specified by r1, r2, theta_1, theta_2, phi_1 and phi_2
*/
double integration_function_monte_carlo(double r1, double theta_1, double phi_1, double r2, double theta_2, double phi_2)
{
    double cos_beta = cos(theta_1)*cos(theta_2) + sin(theta_1)*sin(theta_2)*cos(phi_1-phi_2);
    double f = sin(theta_1)*r1*r1*r2*r2*sin(theta_2)/sqrt(r1*r1+r2*r2-2*r1*r2*cos_beta);
    if((r1*r1+r2*r2-2*r1*r2*cos_beta) > ZERO)
        return f;
    else
        return 0;
}

/*
* Function for solving the integral using gauss-legendre quadrature in cartesian coordinates.
*
* Input:
*-----------
* num_integration_points: Int, Number of integration points.
* x1: Int, Start of integration interval
* x2: Int, End of integration interval
*
* Output:
* -----------
* integral_gauss: Int, Resulting approximation of the integral.
*/
double gauss_quadrature(int num_integration_points, double x1, double x2, double alpha){

    int N = num_integration_points;

    //Will contain the abcissas and weights.
    double *x = new double[N];
    double *w = new double[N];

    //Calculates the weights and abcissas.
    gauleg(x1, x2, x, w, N);

    double integral_gauss = 0;

    //Iterating over all dimensions.
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                for(int l = 0; l < N; l++){
                    for(int m = 0; m < N; m++){
                        for(int n = 0; n < N; n++){
                            integral_gauss += w[i]*w[j]*w[k]*w[l]*w[m]*w[n]*integration_function(x[i],x[j],x[k],x[l],x[m],x[n], alpha);
                        }
                    }
                }
            }
        }
    }
    return integral_gauss;

}

/*
 * Function for computing the integral using Gauss-Laguerre quadrature in sperical coordinates.
 * Input:
 * --------
 * num_integration_points: int, Number of integration points to be used. Higher number typically equals better approximation.
 * alpha: int.
 *
 * Returns:
 * ----------
 * integral_gauss: double, The approximation of the integral.
 */
double gauss_quadrature_spherical(int num_integration_points, double alpha)
{
    int N = num_integration_points;

    //Contains the x and weights for the radial coordinates.
    double *x_lag = new double[N+1];
    double *w_lag = new double[N+1];

    //Contains the x and weights for theta.
    double *x_theta = new double[N];
    double *w_theta = new double[N];

    //Contains the x and weights for phi.
    double *x_phi = new double[N];
    double *w_phi = new double[N];

    gauslag(x_lag, w_lag, N, alpha);
    gauleg(0, acos(-1), x_theta, w_theta, N );
    gauleg(0, 2*acos(-1), x_phi, w_phi, N);

    double integral_gauss = 0;

    //Loops over all dimension and adds the contribution to the total integral.
    for(int i = 1; i < N+1 ; i++){
        for(int j = 1; j < N+1; j++){
            for(int k = 0; k < N; k++){
                for(int l = 0; l < N; l++){
                    for(int m = 0; m < N; m++){
                        for(int n = 0; n < N; n++){
                            integral_gauss += w_lag[i]*w_lag[j]*w_theta[k]*w_theta[l]*w_phi[m]*w_phi[n]*integration_function_spherical(x_lag[i],x_theta[k],x_phi[m],x_lag[j],x_theta[l],x_phi[n]);
                        }
                    }
                }
            }
        }
    }
    return integral_gauss;
}

/*
 * Function for finding the weights and abcissas with gauss legendre polynomials.
 * Input:
 * -------
 * x1: Int, start of integration interval
 * x2: Int, end of integration interval
 * x: Vector, will contain the abcissas.
 * w: Vector, will contain the weights.
 */
void gauleg(double x1, double x2, double x[], double w[], int n)
{
   int         m,j,i;
   double      z1,z,xm,xl,pp,p3,p2,p1;
   double      const  pi = 3.14159265359;
   double      *x_low, *x_high, *w_low, *w_high;

   m  = (n + 1)/2;                             // roots are symmetric in the interval
   xm = 0.5 * (x2 + x1);
   xl = 0.5 * (x2 - x1);

   x_low  = x;                                       // pointer initialization
   x_high = x + n - 1;
   w_low  = w;
   w_high = w + n - 1;

   for(i = 1; i <= m; i++) {                             // loops over desired roots
      z = cos(pi * (i - 0.25)/(n + 0.5));

           /*
       ** Starting with the above approximation to the ith root
           ** we enter the mani loop of refinement bt Newtons method.
           */

      do {
         p1 =1.0;
     p2 =0.0;

       /*
       ** loop up recurrence relation to get the
           ** Legendre polynomial evaluated at x
           */

     for(j = 1; j <= n; j++) {
        p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3)/j;
     }

       /*
       ** p1 is now the desired Legrendre polynomial. Next compute
           ** ppp its derivative by standard relation involving also p2,
           ** polynomial of one lower order.
           */

     pp = n * (z * p1 - p2)/(z * z - 1.0);
     z1 = z;
     z  = z1 - p1/pp;                   // Newton's method
      } while(fabs(z - z1) > ZERO);

          /*
      ** Scale the root to the desired interval and put in its symmetric
          ** counterpart. Compute the weight and its symmetric counterpart
          */

      *(x_low++)  = xm - xl * z;
      *(x_high--) = xm + xl * z;
      *w_low      = 2.0 * xl/((1.0 - z * z) * pp * pp);
      *(w_high--) = *(w_low++);
   }
}

/*
 * Function for finding the weights and abcissas with gauss laguerre polynomials.
 *
 * Input:
 * -------
 * x1: Int, start of integration interval
 * x2: Int, end of integration interval
 * x: Vector, will contain the abcissas.
 * w: Vector, will contain the weights.
 */
void gauslag(double *x, double *w, int n, double alf)
{
    int i,its,j;
    double ai;
    double p1,p2,p3,pp,z,z1;

    for (i=1;i<=n;i++) {
        if (i == 1) {
            z=(1.0+alf)*(3.0+0.92*alf)/(1.0+2.4*n+1.8*alf);
        } else if (i == 2) {
            z += (15.0+6.25*alf)/(1.0+0.9*alf+2.5*n);
        } else {
            ai=i-2;
            z += ((1.0+2.55*ai)/(1.9*ai)+1.26*ai*alf/
                (1.0+3.5*ai))*(z-x[i-2])/(1.0+0.3*alf);
        }
        for (its=1;its<=MAXIT;its++) {
            p1=1.0;
            p2=0.0;
            for (j=1;j<=n;j++) {
                p3=p2;
                p2=p1;
                p1=((2*j-1+alf-z)*p2-(j-1+alf)*p3)/j;
            }
            pp=(n*p1-(n+alf)*p2)/z;
            z1=z;
            z=z1-p1/pp;
            if (fabs(z-z1) <= EPS) break;
        }
        if (its > MAXIT) cout << "too many iterations in gaulag" << endl;
        x[i]=z;
        w[i] = -exp(gammln(alf+n)-gammln((double)n))/(pp*n*p2);
    }
}

double gammln( double xx)
{
    double x,y,tmp,ser;
    static double cof[6]={76.18009172947146,-86.50532032941677,
        24.01409824083091,-1.231739572450155,
        0.1208650973866179e-2,-0.5395239384953e-5};
    int j;

    y=x=xx;
    tmp=x+5.5;
    tmp -= (x+0.5)*log(tmp);
    ser=1.000000000190015;
    for (j=0;j<=5;j++) ser += cof[j]/++y;
    return -tmp+log(2.5066282746310005*ser/x);
}

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

/*
 * Function for generating random number in the interval [0,1]
 *
 * Input:
 * ----------
 * idum: long, random seed.
 */
double ran0(long *idum)
{
   long     k;
   double   ans;

   *idum ^= MASK;
   k = (*idum)/IQ;
   *idum = IA*(*idum - k*IQ) - IR*k;
   if(*idum < 0) *idum += IM;
   ans=AM*(*idum);
   *idum ^= MASK;
   return ans;
}

