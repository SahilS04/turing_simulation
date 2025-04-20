/*****************************************************************
fast_solver_raw.cpp  –  Schnakenberg simulator
Writes spatial std‑dev and a single‑site value for regime analysis.
------------------------------------------------------------------
g++ -O3 -fopenmp fast_solver_raw.cpp -o fast_solver
./fast_solver  alpha  beta  DA  DB  [NX] [NY]
*****************************************************************/
#include <bits/stdc++.h>
using namespace std;

/* defaults --------------------------------------------------- */
int NX=384, NY=384;
double DT=0.01;
int STEPS=8000, OUT=30;
int GS=20;
/* ------------------------------------------------------------ */
inline int id(int i,int j){return i+j*NX;}
inline int w(int x,int m){return x<0?x+m:x>=m?x-m:x;}

double sd(const vector<double>& a,double mean){
    double s=0; for(double v:a) s+=(v-mean)*(v-mean);
    return sqrt(s/a.size());
}
void diffuse(vector<double>& X,const vector<double>& rhs,double D){
    double r=DT*D, den=1+4*r;
    for(int it=0;it<GS;++it){
        #pragma omp parallel for collapse(2)
        for(int j=0;j<NY;++j)for(int i=0;i<NX;++i){
            if((i+j+it)&1) continue;
            double nb=X[id(w(i+1,NX),j)]+X[id(w(i-1,NX),j)]
                     +X[id(i,w(j+1,NY))]+X[id(i,w(j-1,NY))];
            X[id(i,j)] = (rhs[id(i,j)] + r*nb)/den;
        }
    }
}

int main(int c,char** v){
    if(c<5){cerr<<"usage: a b DA DB\n";return 1;}
    double A0p = atof(v[1]),  B0p = atof(v[2]);
    double DA = atof(v[3]),   DB  = atof(v[4]);
    if(c>6){ NX=atoi(v[5]); NY=atoi(v[6]); }

    vector<double>A(NX*NY),B(NX*NY),At(NX*NY),Bt(NX*NY);
    double A0=A0p+B0p, B0=B0p/((A0p+B0p)*(A0p+B0p));
    mt19937 g(1234); uniform_real_distribution<> U(-1e-2,1e-2);
    for(size_t i=0;i<A.size();++i){A[i]=A0+U(g); B[i]=B0+U(g);}

    int is = NX/3, js = NY/3;              // probe site
    ofstream ts("timeseries.csv"); ts<<"t,stdA,A_site\n";

    for(int s=0;s<=STEPS;++s){
        #pragma omp parallel for collapse(2)
        for(int j=0;j<NY;++j)for(int i=0;i<NX;++i){
            int I=id(i,j); double a=A[I],b=B[I];
            At[I]=a+DT*( A0p - a + a*a*b );
            Bt[I]=b+DT*( B0p - a*a*b );
        }
        diffuse(A,At,DA); diffuse(B,Bt,DB);

        if(s%OUT==0){
            double mean=accumulate(A.begin(),A.end(),0.0)/A.size();
            ts<<s*DT<<','<<sd(A,mean)<<','<<A[id(is,js)]<<'\n';
        }
    }
    return 0;
}
