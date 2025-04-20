/* analysis_solver.cpp  (compile with  g++ -O3 -fopenmp analysis_solver.cpp) */
#define NX 512
#define NY 512
#define DA 100.0
#define DB 1.0
#define ALPHA 0.1
#define BETA  0.5
#define DT    0.005
#define STEPS 15000
#define OUT_INT 50            // snapshot every 50 steps

#include <bits/stdc++.h>
using namespace std;

inline int id(int i,int j){return i+j*NX;}
inline int w(int x,int m){return (x<0)?x+m:(x>=m)?x-m:x;}

void write_field(const vector<double>& f,const string& name){
    ofstream o(name);
    o.setf(ios::fixed); o.precision(6);
    for(int j=0;j<NY;++j){for(int i=0;i<NX;++i){
        o<<f[id(i,j)]<<(i<NX-1? ',':'\n');
    }}
}

void write_line(ofstream& o,double t,double s,double mean){
    o<<t<<','<<s<<','<<mean<<'\n';
}

double stddev(const vector<double>& a,double mean){
    double s=0; for(double v:a) s+=(v-mean)*(v-mean);
    return sqrt(s/a.size());
}

void diffuse(vector<double>& X,const vector<double>& rhs,double D){
    const double r=DT*D, denom=1.0+4*r;
    for(int it=0;it<40;++it){ /* 40 Gauss-Seidel Iterations*/
        #pragma omp parallel for collapse(2)
        for(int j=0;j<NY;++j)for(int i=0;i<NX;++i){
            int I=id(i,j); if((i+j+it)%2) continue;            /* red/black */
            double nb = X[id(w(i+1,NX),j)]+X[id(w(i-1,NX),j)]
                      + X[id(i,w(j+1,NY))]+X[id(i,w(j-1,NY))];
            X[I]=(rhs[I]+r*nb)/denom;
        }
    }
}

int main(){
    vector<double> A(NX*NY),B(NX*NY),At(NX*NY),Bt(NX*NY);
    double A0=ALPHA+BETA, B0=BETA/((ALPHA+BETA)*(ALPHA+BETA));
    mt19937 rng(1234); uniform_real_distribution<> U(-0.5,0.5);
    for(auto& v:A) v=A0+U(rng)*1e-1;
    for(auto& v:B) v=B0+U(rng)*1e-1;

    ofstream log("timeseries.csv"); log<<"t,stdA,meanA\n";

    for(int s=0;s<=STEPS;++s){
        /* reaction */
        #pragma omp parallel for collapse(2)
        for(int j=0;j<NY;++j)for(int i=0;i<NX;++i){
            int I=id(i,j); double a=A[I],b=B[I];
            At[I]=a+DT*(ALPHA-a+a*a*b);
            Bt[I]=b+DT*(BETA -a*a*b);
        }
        diffuse(A,At,DA); diffuse(B,Bt,DB);

        if(s%OUT_INT==0){
            /* output field & stats */
            string fname="A_t"+to_string(s)+".csv";
            write_field(A,fname);
            double mean=accumulate(A.begin(),A.end(),0.0)/A.size();
            log.setf(ios::fixed); log.precision(4);
            write_line(log,s*DT,stddev(A,mean),mean);
            cerr<<"dump "<<s<<"\n";
        }
    }
    return 0;
}
