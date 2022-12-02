

functions {
  real sparse_Leroux_lpdf(vector phi, real alpha, 
    int[,] W_sparse, vector W_weight, vector ID_sparse, 
    real sumlogdet, 
    int n, int W_n) {
      row_vector[n] phit_ID; // phi' * D
      row_vector[n] phit_W; // phi' * W

      phit_ID = (phi .* ID_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + W_weight[i]*phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + W_weight[i]*phi[W_sparse[i, 1]];
      }
    
      return 0.5 * (sumlogdet
                    - (phi' * phi - alpha * (phit_ID * phi + phit_W * phi)));
  }
}


data {
    // Data block
        int<lower=0> N; // Number of obs
        int Y[N]; // Vector of cases
        int<lower=0> tau; // Number of obs
        matrix[N, tau] Yprev; // Vector of cases of previous week
        vector[N] lOff; // Vector of cases of previous week
        
        int<lower=0> Nin;   // N In sample
        int<lower=0> Nout;  // N Out of sample
        int idIn[Nin];        // Vector of cases
        int idOut[Nout];       // Vector of cases
        
        int<lower=0> Ndis;  // Number of districts
        int<lower=0> Ntimes;  // Number of times
        int tdId[Ntimes, Ndis]; // Associate obs to district

        matrix<lower = 0>[Ndis, Ndis] W; // Adjacency
        int W_n;                         // Number of adjacent region pairs
        
        int<lower = 1> k;    // n RtCovariates
        matrix[N, k] X;      // Rt covariate
        int<lower = 1> l;    // n base covariates
        matrix[N, l] V;      // Base covariates 
        vector[N] lsN;       // Susceptible

    }

transformed data {
        // Vector of zeros
        vector[Ndis] zeros;
        vector[tau] ones;
        // Data for sparse car
        int W_sparse[W_n, 2];   // adjacency pairs
        vector[W_n] W_weight;     // Connection weights
        vector[Ndis] ID_sparse;     // diagonal of D (number of neigbors for each site)
        vector[Ndis] lambda;       // eigenvalues of invsqrtD * W * invsqrtD
        
        zeros = rep_vector(0, Ndis);
        ones = rep_vector(1, tau);
  
        { // generate sparse representation for W
          int counter;
          counter = 1;
          // loop over upper triangular part of W to identify neighbor pairs
          for (i in 1:(Ndis - 1)) {
            for (j in (i + 1):Ndis) {
              if (W[i, j] > 0) {
                W_sparse[counter, 1] = i;
                W_sparse[counter, 2] = j;
                W_weight[counter] = W[i, j];
                counter = counter + 1;
              }
            }
          }
        }
        
        for (i in 1:Ndis) ID_sparse[i] = 1-sum(W[i]);
        
        lambda = eigenvalues_sym(add_diag(W, ID_sparse));
    }
      
parameters {  
    // Parameters block

        // Linear term coefficients
        real lr0;
        vector[k] beta;
        vector[l] eta;
        simplex[tau] w;
        
        // CAR parameters 1
        vector[Ndis] phi1s[Ntimes];
        real<lower=0> sigmac1;
        real<lower = 0, upper = 1> alpha1;
        // AR parameters
        real<lower=0, upper=1> rho1;
        
	      // CAR parameters 2
        vector[Ndis] phi2s[Ntimes];
        real<lower=0> sigmac2;
        real<lower = 0, upper = 1> alpha2;        
        // AR parameters
        real<lower=0, upper=1> rho2;

    }

transformed parameters {
        // Transformed parameters block
        vector[N] phi1;
        vector[N] phi2;
        vector[N] lm;

        // CAR phi
        phi1[tdId[1]] = sigmac1*phi1s[1];
        phi2[tdId[1]] = sigmac2*phi2s[1];
        for (i in 2:Ntimes){
          phi1[tdId[i]] = rho1*phi1[tdId[i-1]] + sigmac1*phi1s[i];
          phi2[tdId[i]] = rho2*phi2[tdId[i-1]] + sigmac2*phi2s[i];
        }


        lm = log((Yprev[,1]*w[1]+Yprev[,2]*w[2]+Yprev[,3]*w[3]) .* exp(lr0 + X*beta + phi1) + exp(lOff + V*eta + phi2)) + lsN;
    } 


model {
        // Model block
        
        // Linear coefficients prior
        lr0 ~ normal(-0.5, 1);	
        beta ~ normal(0, 1);
        eta ~ normal(0, 1);
        w ~ dirichlet(ones);
        
        // CAR priors
        {
          vector[Ndis] ldet_terms1;
          vector[Ndis] ldet_terms2;
          real sumlogdet1;
          real sumlogdet2;
          
          // CAR determinant
          for (i in 1:Ndis){
            ldet_terms1[i] = log1m(alpha1 * lambda[i]);
            ldet_terms2[i] = log1m(alpha2 * lambda[i]);
          }  
          sumlogdet1 = sum(ldet_terms1);
          sumlogdet2 = sum(ldet_terms2);
          // CAR prior
          for (i in 1:Ntimes){
                phi1s[i] ~ sparse_Leroux(alpha1, 
                                      W_sparse, W_weight, ID_sparse, sumlogdet1, 
                                      Ndis, W_n);
                phi2s[i] ~ sparse_Leroux(alpha2, 
                                      W_sparse, W_weight, ID_sparse, sumlogdet2, 
                                      Ndis, W_n);
          }
        }

        //alpha1 ~ beta(2, 1);
        //rho1 ~ beta(2, 1);
        sigmac1 ~ normal(0, 0.1);
        
        //alpha2 ~ beta(2, 1);
        //rho2 ~ beta(2, 1);
        sigmac2 ~ normal(0, 0.5);

        // Identifiability
        sum(phi1) ~ normal(0, 0.001*N);
        sum(phi2) ~ normal(0, 0.001*N);

        // likelihood
        Y[idIn] ~ poisson_log(lm[idIn]); 
    }

    // generated quantities{
    //     // Output block
    //     real Y_pred[N];
    //     vector[Nin] log_lik;
    //     
    //     for (i in 1:N){
    //       Y_pred[i] = poisson_log_safe_rng(lm[i]);  // Posterior predictive distribution
    //       if (i<=Nin){
    //         log_lik[i] = poisson_log_lpmf(Y[idIn[i]] | lm[idIn[i]]);
    //       }
    //     }
    // }
