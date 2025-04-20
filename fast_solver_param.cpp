// fast_solver_param.cpp
#include <bits/stdc++.h>
using namespace std;

// defaults (override via argv)
int NX = 384, NY = 384;
double DT = 0.01;
int STEPS = 8000, OUT_INT = 30;
int GS = 20;

inline int idx(int i,int j){ return i + j*NX; }
inline int wrap(int x,int m){ return x<0?x+m:x>=m?x-m:x; }

double compute_std(const vector<double>& A, double mean) {
    double s = 0;
    for (double v : A) s += (v - mean)*(v - mean);
    return sqrt(s / A.size());
}

void diffuse(vector<double>& X, const vector<double>& rhs, double D) {
    double r = DT*D, denom = 1.0 + 4*r;
    for (int it = 0; it < GS; ++it) {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                if ((i+j+it)&1) continue;  // red/black
                int I = idx(i,j);
                double nb = X[idx(wrap(i+1,NX),j)]
                          + X[idx(wrap(i-1,NX),j)]
                          + X[idx(i,wrap(j+1,NY))]
                          + X[idx(i,wrap(j-1,NY))];
                X[I] = (rhs[I] + r*nb) / denom;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        cerr<<"Usage: "<<argv[0]<<" alpha beta DA DB [NX NY]\n";
        return 1;
    }
    double alpha = atof(argv[1]),
           beta  = atof(argv[2]),
           DA    = atof(argv[3]),
           DB    = atof(argv[4]);
    if (argc > 6) { NX = atoi(argv[5]); NY = atoi(argv[6]); }

    // allocate fields
    vector<double> A(NX*NY), B(NX*NY),
                   At(NX*NY), Bt(NX*NY);
    double A0 = alpha + beta,
           B0 = beta / ((alpha+beta)*(alpha+beta));
    // small random noise
    mt19937_64 rng(1234);
    uniform_real_distribution<double> U(-1e-2,1e-2);
    for (int i = 0; i < NX*NY; ++i) {
        A[i] = A0 + U(rng);
        B[i] = B0 + U(rng);
    }

    // open timeseries CSV
    ofstream ts("timeseries.csv");
    ts<<"t,stdA,meanA\n";

    // time stepping
    for (int step = 0; step <= STEPS; ++step) {
        // reaction (explicit)
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int I = idx(i,j);
                double a = A[I], b = B[I];
                At[I] = a + DT*(alpha - a + a*a*b);
                Bt[I] = b + DT*(beta  - a*a*b);
            }
        }
        // diffusion (implicit Gauss–Seidel)
        diffuse(A, At, DA);
        diffuse(B, Bt, DB);

        // record std and mean at intervals
        if (step % OUT_INT == 0) {
            double meanA = accumulate(A.begin(), A.end(), 0.0) / A.size();
            double stdA  = compute_std(A, meanA);
            ts<< fixed<< setprecision(4)
              << step*DT <<","<< stdA <<","<< meanA <<"\n";
        }
    }

    // write final snapshot
    ofstream fout("A_final.csv");
    fout<< fixed<< setprecision(6);
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            fout<< A[idx(i,j)];
            if (i < NX-1) fout<< ",";
        }
        fout<<"\n";
    }

    return 0;
}
