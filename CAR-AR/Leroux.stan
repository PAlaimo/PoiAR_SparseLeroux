data {
        // Data block
        int<lower=0> N; // Number of obs
        int Y[N];       // Vector of cases
        vector[N] lOff; // log-Offset
        
        matrix<lower = 0>[N, N] W;       // Adjacency

        int<lower = 1> k;    // num of covariates
        matrix[N, k] X;      // Covariate
    }

transformed data {  
        // Building up the ingredients for the spatial precision matrix
      
        // Declaration
        vector[N] Dvec;                     // D: diagonal matrix with number of neigbors for each site
        vector[N] zeros = rep_vector(0, N); // Vector of zeros
      
        // Construction
        for (i in 1:N) Dvec[i] = sum(W[i, ]);
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
        vector[N] lm;
        matrix[N, N] P;
      
        // Precision matrix
        P = add_diag( alpha * add_diag(-W, Dvec), (1-alpha) );
        
        // Log mean
        lm = lOff + X*beta + sigmac*phi;        
    }

model {
        
        // Covariates coefficents
        beta ~ normal(0, 1);
        
        // Conditional variance
        sigmac ~ normal(0, 1);
        
        // Spatial random effects
        phi ~ multi_normal_prec(zeros, P);
        alpha ~ beta(1, 1);
        
        // Sum-to-zero constrain
        sum(phi) ~ normal(0, 0.001*N);
        
        // log-Likelihood
        Y ~ poisson_log(lm);
    }
