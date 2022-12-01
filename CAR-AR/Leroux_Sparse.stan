

functions {
  real sparse_Leroux_lpdf(vector phi, real alpha, 
    int[,] W_sparse, vector W_weight, vector ID_sparse, 
    real sumlogdet, int n, int W_n) {
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
        int Y[N];       // Vector of cases
        vector[N] lOff; // log-Offset
        
        matrix<lower = 0>[N, N] W;       // Adjacency
        int W_n;                         // Number of adjacent region pairs
        
        int<lower = 1> k;    // num of covariates
        matrix[N, k] X;      // Covariate
    }

transformed data {
        // Vector of zeros
        vector[N] zeros;
        // Data for sparse car
        int W_sparse[W_n, 2];   // adjacency pairs
        vector[W_n] W_weight;     // Connection weights
        vector[N] ID_sparse;     // diagonal of D (number of neigbors for each site)
        vector[N] lambda;       // eigenvalues of invsqrtD * W * invsqrtD
        
        zeros = rep_vector(0, N);

        { // generate sparse representation for W
          int counter;
          counter = 1;
          // loop over upper triangular part of W to identify neighbor pairs
          for (i in 1:(N - 1)) {
            for (j in (i + 1):N) {
              if (W[i, j] > 0) {
                W_sparse[counter, 1] = i;
                W_sparse[counter, 2] = j;
                W_weight[counter] = W[i, j];
                counter = counter + 1;
              }
            }
          }
        }
        
        for (i in 1:N) ID_sparse[i] = 1-sum(W[i]);
        
        lambda = eigenvalues_sym(add_diag(W, ID_sparse));
    }
      
parameters {  
    // Parameters block

        // Linear term coefficients
        vector[k] beta;

        // CAR parameters
        vector[N] phi;
        real<lower=0> sigmac;
        real<lower = 0, upper = 1> alpha;
    }

transformed parameters {
        // Transformed parameters block
        vector[N] lm;

        // CAR phi
        lm = lOff + X*beta + sigmac*phi;
    } 


model {
        // Model block
        
        // Linear coefficients prior
        beta ~ normal(0, 1);

        // CAR priors
        {
          vector[N] ldet_terms;
          real sumlogdet;

          // CAR determinant
          for (i in 1:N){
            ldet_terms[i] = log1m(alpha * lambda[i]);
          }  
          sumlogdet = sum(ldet_terms);
          // CAR prior
          phi ~ sparse_Leroux(alpha, 
                              W_sparse, W_weight, ID_sparse, 
                              sumlogdet, N, W_n);
        }

        alpha ~ beta(1, 1);
        sigmac ~ normal(0, 1);
        
        // Sum-to-zero constrain
        sum(phi) ~ normal(0, 0.001*N);
        
        // log-Likelihood
        Y ~ poisson_log(lm); 
    }

