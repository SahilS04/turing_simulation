/*
  sweep_solver.cpp
  ----------------
  A modified 2D Schnakenberg reaction-diffusion simulation that sweeps over both α and β.
  
  For each (α, β) combination:
    - The simulation is run from a homogeneous steady state (plus small noise),
      with maximum 10,000 time steps at dt = 0.01.
    - The simulation stops early if the spatial standard deviation (σ_A) of the activator
      crosses the threshold of 0.3.
    - The time (step · dt) at which this “spike” is first detected is recorded as t_pattern.
    - If no spike is detected, t_pattern is set to 10 (to indicate “no pattern formation”).
  
  The program opens a CSV file at the beginning and writes out a header followed by lines containing:
       alpha,beta,t_pattern
       
  Compile with:
    g++ -fopenmp -O2 sweep_solver.cpp -o sweep_solver
  Run with:
    ./sweep_solver
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>   // for rand() and srand()
#include <iomanip>   // for setprecision
#include <sstream>
#include <omp.h>

// Grid size
static const int NX = 128;
static const int NY = 128;

// Reaction-Diffusion parameters (Schnakenberg model)
static const double DA = 1.0;   // Diffusion coefficient for A
//static const double DB = 10.0;  // Diffusion coefficient for B
//static const double beta = 0.2;  // Default beta value
static const double alpha = 0.02; // Default alpha value

// Time stepping settings
static const double dt = 0.01;   // time step size
static const int MAX_STEPS = 5000;  // maximum number of steps

// Gauss-Seidel iteration settings
static const int GS_ITER = 20;   // Number of Gauss-Seidel sweeps per time step

// Threshold for pattern formation (activator standard deviation)
static const double THRESHOLD = 0.3;

// Convert (i,j) to 1D index
inline int idx(int i, int j) {
    return i + j * NX;
}

// Periodic boundary helper
inline int wrap(int x, int max) {
    if(x >= max) return x - max;
    if(x < 0)    return x + max;
    return x;
}

// Write CSV header and then later each line will have: alpha,beta,t_pattern
void openCSV(std::ofstream &fout, const std::string &filename) {
    fout.open(filename.c_str());
    if (!fout.is_open()) {
        std::cerr << "Error opening output file " << filename << std::endl;
        exit(1);
    }
    fout << "beta,ratio,t_pattern\n";
}

// Red-Black Gauss-Seidel solver for implicit diffusion.
// Solves: (I - dt * diffCoeff * Laplacian) * X = RHS.
void diffuseGaussSeidel(std::vector<double> &X, const std::vector<double> &RHS, double diffCoeff) {
    double r = dt * diffCoeff;
    for (int iter = 0; iter < GS_ITER; iter++) {
        // Red update
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                if ((i + j) % 2 == 0) {
                    int id = idx(i, j);
                    int ip = wrap(i + 1, NX), im = wrap(i - 1, NX);
                    int jp = wrap(j + 1, NY), jm = wrap(j - 1, NY);
                    double sumN = X[idx(ip, j)] + X[idx(im, j)]
                                + X[idx(i, jp)] + X[idx(i, jm)];
                    double numer = RHS[id] + r * sumN;
                    double denom = 1.0 + 4.0 * r;
                    X[id] = numer / denom;
                }
            }
        }
        // Black update
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                if ((i + j) % 2 == 1) {
                    int id = idx(i, j);
                    int ip = wrap(i + 1, NX), im = wrap(i - 1, NX);
                    int jp = wrap(j + 1, NY), jm = wrap(j - 1, NY);
                    double sumN = X[idx(ip, j)] + X[idx(im, j)]
                                + X[idx(i, jp)] + X[idx(i, jm)];
                    double numer = RHS[id] + r * sumN;
                    double denom = 1.0 + 4.0 * r;
                    X[id] = numer / denom;
                }
            }
        }
    }
}

// Compute standard deviation of a vector (spatial field)
double compute_std(const std::vector<double>& arr) {
    double sum = 0.0, mean = 0.0, sq_sum = 0.0;
    int n = arr.size();
    for (int i = 0; i < n; i++){
        sum += arr[i];
    }
    mean = sum / n;
    for (int i = 0; i < n; i++){
        sq_sum += (arr[i] - mean) * (arr[i] - mean);
    }
    return sqrt(sq_sum / n);
}

int main() {
    // Open CSV file to record (alpha, beta, t_pattern)
    std::ofstream fout;
    openCSV(fout, "pattern_formation_times.csv");

    // Parameter ranges:
    for (double beta = -0.25; beta < 0.26; beta += 0.01) {
        for (double DB = 1.0; DB < 18.0; DB += 1.0) {
            double ratio = DA/DB;
            std::cout << "Running simulation for beta = " << beta << ", DA/DB = " << ratio << std::endl;
            
            // Allocate fields for A and B
            std::vector<double> A(NX * NY), B(NX * NY);
            std::vector<double> Atemp(NX * NY), Btemp(NX * NY);

            // Compute homogeneous steady state:
            // A0 = alpha + beta,  B0 = beta / ((alpha + beta)^2)
            double sumAB = alpha + beta;
            // Avoid division by zero; if sumAB is zero, skip this run.
            if (fabs(sumAB) < 1e-8) {
                std::cerr << "Skipping alpha = " << alpha << ", beta = " << beta << " due to zero steady state value." << std::endl;
                continue;
            }
            if (fabs(beta) < 1e-8) {
                std::cerr << "Skipping beta = " << beta << " due to zero value." << std::endl;
                continue;
            }

            double A0 = sumAB;
            double B0 = beta / (sumAB * sumAB);

            // Initialize with small noise around steady state.
            srand(1234); // fixed seed for reproducibility
            for (int j = 0; j < NY; j++){
                for (int i = 0; i < NX; i++){
                    double noiseA = 0.1 * (((double)rand()/RAND_MAX) - 0.5);
                    double noiseB = 0.1 * (((double)rand()/RAND_MAX) - 0.5);
                    A[idx(i,j)] = A0 + noiseA;
                    B[idx(i,j)] = B0 + noiseB;
                }
            }

            // Variable to record time to pattern formation.
            // t_pattern will be the simulation time (step*dt) when sigma_A first exceeds THRESHOLD.
            // If no spike occurs, we record t_pattern = 10.
            double t_pattern = 50; // default flag for "no pattern"
            bool pattern_found = false;
            
            // Time-stepping loop
            for (int step = 0; step <= MAX_STEPS; step++) {
                // Reaction step (explicit):
                // A* = A + dt*(alpha - A + A^2 * B)
                // B* = B + dt*(beta - A^2 * B)
                #pragma omp parallel for collapse(2)
                for (int j = 0; j < NY; j++){
                    for (int i = 0; i < NX; i++){
                        int id = idx(i,j);
                        double a = A[id];
                        double b = B[id];
                        double fA = alpha - a + (a * a * b);
                        double fB = beta - (a * a * b);
                        Atemp[id] = a + dt * fA;
                        Btemp[id] = b + dt * fB;
                    }
                }
                
                // Diffusion step (implicit):
                diffuseGaussSeidel(A, Atemp, DA);
                diffuseGaussSeidel(B, Btemp, DB);
                double sA = compute_std(A);
                if (sA > THRESHOLD) {
                    t_pattern = step * dt;  // record time (in seconds)
                    pattern_found = true;
                    break;  // Stop simulation early if a spike is detected.
                }
            }  // end simulation time loop
            
            // Write out the result for this parameter set
            fout << std::fixed << std::setprecision(4) << beta << ","
                 << std::fixed << std::setprecision(4) << ratio << ","
                 << std::fixed << std::setprecision(4) << t_pattern << "\n";
        } // end beta loop
    } // end alpha loop

    fout.close();
    std::cout << "Parameter sweep simulation finished. Results written to pattern_formation_times.csv\n";
    return 0;
}
