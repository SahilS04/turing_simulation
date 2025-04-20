/*
  schnakenberg.cpp
  A 2D Schnakenberg reaction-diffusion simulation on a 256x256 grid,
  using an IMEX time scheme, Gauss-Seidel for diffusion, and OpenMP parallelization.
  Outputs CSV snapshots of A and B every 10 steps for Python analysis.

  References:
  - Turing, A.M. (1952). The Chemical Basis of Morphogenesis.
  - Schnakenberg, J. (1979). Simple chemical reaction systems with limit cycle behavior.
  - IMEX approach with Gauss-Seidel: typical in Turing pattern codes.

  Compile and run:
    g++ -fopenmp -O2 schnakenberg.cpp -o schnakenberg
    ./schnakenberg
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>   // for std::rand
#include <iomanip>   // for std::setprecision
#include <omp.h>     // for OpenMP

// Grid size
static const int NX = 512;
static const int NY = 512;

// Reaction-Diffusion parameters (example)
static const double DA = 100.0;   // Diffusion of A
static const double DB = 1.0;  // Diffusion of B
//static const double alpha = 0.1; // feed rate of activator
static const double beta  = 0.5; // feed rate of inhibitor
// Time stepping
static const double dt = 0.01;   // time step
static const int STEPS = 20000;   // total number of steps
static const int OUTPUT_INTERVAL = 20; // output every 20 steps

// Gauss-Seidel iteration settings
static const int GS_ITER = 40;   // Gauss-Seidel sweeps per time step

// Convert (i,j) to 1D index
inline int idx(int i, int j) {
    return i + j*NX;
}

// Helper to wrap indices for periodic BC
inline int wrap(int x, int max) {
    // for a negative or > max index, this ensures periodic
    if(x >= max) return x - max;
    if(x < 0)    return x + max;
    return x;
}

// Write a 2D field (arr) to CSV of size NX x NY
void writeCSV(const std::vector<double> &arr, const std::string &filename)
{
    std::ofstream fout(filename.c_str());
    fout << std::fixed << std::setprecision(6);
    for(int j=0; j<NY; j++) {
        for(int i=0; i<NX; i++) {
            fout << arr[idx(i,j)];
            if(i < NX-1) fout << ",";
        }
        fout << "\n";
    }
    fout.close();
}

// Red-Black Gauss-Seidel for (I - dtD Laplacian)X = RHS
//   using 2D periodic boundary
void diffuseGaussSeidel(std::vector<double> &X, const std::vector<double> &RHS,
                        double diffCoeff)
{
    // We solve (X - dt*diffCoeff*Lap(X)) = RHS
    // => X_new(i,j) = (RHS + dt*diffCoeff*(neighbors)) / (1 + 4*r)
    // We'll do multiple sweeps. Red-Black approach for parallelization.

    double r = dt * diffCoeff;
    for(int iter=0; iter<GS_ITER; iter++) {
        // 1) update Red cells
        #pragma omp parallel for collapse(2)
        for(int j=0; j<NY; j++) {
            for(int i=0; i<NX; i++) {
                // color check: (i+j) % 2 == 0 -> red
                if( (i+j)%2 == 0 ) {
                    int id = idx(i,j);
                    // periodic neighbors
                    int ip = wrap(i+1, NX), im = wrap(i-1, NX);
                    int jp = wrap(j+1, NY), jm = wrap(j-1, NY);

                    // sum of neighbors
                    double sumN = X[idx(ip,j)] + X[idx(im,j)]
                                + X[idx(i,jp)] + X[idx(i,jm)];
                    double numer = RHS[id] + r * sumN;
                    double denom = 1.0 + 4.0*r;
                    X[id] = numer/denom;
                }
            }
        }
        // 2) update Black cells
        #pragma omp parallel for collapse(2)
        for(int j=0; j<NY; j++) {
            for(int i=0; i<NX; i++) {
                if( (i+j)%2 == 1 ) {
                    int id = idx(i,j);
                    int ip = wrap(i+1, NX), im = wrap(i-1, NX);
                    int jp = wrap(j+1, NY), jm = wrap(j-1, NY);

                    double sumN = X[idx(ip,j)] + X[idx(im,j)]
                                + X[idx(i,jp)] + X[idx(i,jm)];
                    double numer = RHS[id] + r * sumN;
                    double denom = 1.0 + 4.0*r;
                    X[id] = numer/denom;
                }
            }
        }
    }
}

int main()
{
    // Allocate arrays for A, B
    std::vector<double> A(NX*NY), B(NX*NY);
    std::vector<double> Atemp(NX*NY), Btemp(NX*NY);

    // Initialize with near steady state + small random noise
    // Homogeneous steady state of Schnakenberg is roughly
    // A0 = alpha+beta,  B0 = beta / (alpha+beta)^2
    double alpha = 0.1;
    double A0 = alpha + beta; 
    double B0 = beta / ((alpha + beta)*(alpha + beta));

    srand(1234);  // fixed seed for reproducibility
    for(int j=0; j<NY; j++){
        for(int i=0; i<NX; i++){
            double noiseA = (((double)rand()/RAND_MAX) - 0.5);
            double noiseB = (((double)rand()/RAND_MAX) - 0.5);
            double perturb = sin(4.0 * M_PI * i / NX);
            A[idx(i,j)] = A0 + noiseA;
            B[idx(i,j)] = B0 + noiseB;
        }
    }

    // Start time stepping
    for(int step=0; step<=STEPS; step++){
        if (step == 10000){
            alpha = -0.2;
        }
        // ======= Reaction step (explicit) =======
        // A^* = A^n + dt*( alpha - A + A^2 * B )
        // B^* = B^n + dt*( beta  - A^2 * B )
        #pragma omp parallel for collapse(2)
        for(int j=0; j<NY; j++){
            for(int i=0; i<NX; i++){
                int id = idx(i,j);
                double a = A[id];
                double b = B[id];
                double fA = alpha - a + (a*a*b);
                double fB = beta  - (a*a*b);
                Atemp[id] = a + dt*fA;
                Btemp[id] = b + dt*fB;
            }
        }

        // ======= Diffusion step (implicit) =======
        // Solve: (I - dt*DA * Lap) A^{n+1} = Atemp
        //        (I - dt*DB * Lap) B^{n+1} = Btemp
        diffuseGaussSeidel(A, Atemp, DA);
        diffuseGaussSeidel(B, Btemp, DB);

        // Output snapshots every 10 steps
        if(step % OUTPUT_INTERVAL == 0){
            // create filenames
            char fnameA[64], fnameB[64];
            sprintf(fnameA, "A_t%05d.csv", step);
            sprintf(fnameB, "B_t%05d.csv", step);
            writeCSV(A, fnameA);
            //writeCSV(B, fnameB);
            std::cout << "Output step " << step << std::endl;
        }
    } // end time stepping

    std::cout << "Simulation finished.\n";
    return 0;
}
