Sparse spatio-temporal Leroux-AR in STAN
================

<style type="text/css">
body{ /* Normal  */
      font-size: 14px;
  }
h1.title {
  font-size: 30px;
  color: black;
  font-weight: bold;
}
h1 { /* Header 1 */
    font-size: 25px;
  color: black;
  font-weight: bold;
}
h2 { /* Header 2 */
    font-size: 20px;
  color: black;
  font-weight: bold;
}
h3 { /* Header 3 */
    font-size: 15px;
  color: black;
  font-weight: bold;
}
code.r{ /* Code block */
    font-size: 14px;
}
pre, code {
    color:  #1B0F0E;
  }
</style>
There is a rich literature on *Gaussian Markov Random Fields* and
*Conditional Auto Regressive* (CAR) priors that suite the modeling of
dependent random effects over a network. They have seen wide
applicability in the disease mapping context, and much interest has been
recently devoted to their extension in the space-time setting. We here
consider the space-time extension of the Leroux model originally
proposed by and its sparse and efficient implementation in STAN.

## The Leroux model

Let
![\phi\_{s}, \\;s=1,\dots,l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cphi_%7Bs%7D%2C%20%5C%3Bs%3D1%2C%5Cdots%2Cl "\phi_{s}, \;s=1,\dots,l")
be the set of Gaussian spatial random effects over a discrete domain
![\mathcal{S}=\left\lbrace \boldsymbol{s}\_1, \dots, \boldsymbol{s}\_l\right\rbrace](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmathcal%7BS%7D%3D%5Cleft%5Clbrace%20%5Cboldsymbol%7Bs%7D_1%2C%20%5Cdots%2C%20%5Cboldsymbol%7Bs%7D_l%5Cright%5Crbrace "\mathcal{S}=\left\lbrace \boldsymbol{s}_1, \dots, \boldsymbol{s}_l\right\rbrace").
Let
![\boldsymbol{W}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BW%7D "\boldsymbol{W}")
be a
![l\times l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l%5Ctimes%20l "l\times l")
adjacency matrix for the locations in
![\mathcal{S}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmathcal%7BS%7D "\mathcal{S}"),
whose elements
![w\_{ij}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_%7Bij%7D "w_{ij}")
are
![\>0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%3E0 ">0")
if and only if
![i\neq j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i%5Cneq%20j "i\neq j")
and
![i\sim j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i%5Csim%20j "i\sim j")
(location
![\boldsymbol{s}\_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Bs%7D_i "\boldsymbol{s}_i")
is a neighbor of location
![\boldsymbol{s}\_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7Bs%7D_j "\boldsymbol{s}_j")),
and
![0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;0 "0")
otherwise. Notice that each location is not a neighbor to itself.

The Leroux model extends the typical ICAR specification through a convex
combination of its precision matrix with that of a Multivariate Gaussian
with independent components. This is achieved through the inclusion of a
coefficient
![\alpha\in (0, 1)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Calpha%5Cin%20%280%2C%201%29 "\alpha\in (0, 1)").
It regulates the extent of the spatial dependence and induces the
following proper joint prior:

![\boldsymbol{\phi} \sim \mathcal{N}\_l\left(\boldsymbol{0},\\; \sigma^2\cdot\boldsymbol{Q}(\alpha,\\,\boldsymbol{W})^{-1}\right),](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7B%5Cphi%7D%20%5Csim%20%5Cmathcal%7BN%7D_l%5Cleft%28%5Cboldsymbol%7B0%7D%2C%5C%3B%20%5Csigma%5E2%5Ccdot%5Cboldsymbol%7BQ%7D%28%5Calpha%2C%5C%2C%5Cboldsymbol%7BW%7D%29%5E%7B-1%7D%5Cright%29%2C "\boldsymbol{\phi} \sim \mathcal{N}_l\left(\boldsymbol{0},\; \sigma^2\cdot\boldsymbol{Q}(\alpha,\,\boldsymbol{W})^{-1}\right),")

where
![\sigma^2\>0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csigma%5E2%3E0 "\sigma^2>0")
and

![\boldsymbol{Q}(\alpha,\\,\boldsymbol{W})=\left( \alpha\cdot(\boldsymbol{D}-\boldsymbol{W}) + (1-\alpha)\boldsymbol{I}\_l\right).](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BQ%7D%28%5Calpha%2C%5C%2C%5Cboldsymbol%7BW%7D%29%3D%5Cleft%28%20%5Calpha%5Ccdot%28%5Cboldsymbol%7BD%7D-%5Cboldsymbol%7BW%7D%29%20%2B%20%281-%5Calpha%29%5Cboldsymbol%7BI%7D_l%5Cright%29. "\boldsymbol{Q}(\alpha,\,\boldsymbol{W})=\left( \alpha\cdot(\boldsymbol{D}-\boldsymbol{W}) + (1-\alpha)\boldsymbol{I}_l\right).")

This yields the conditional mean

![\mathbb{E}\left\[ \phi\_{i}\|\boldsymbol{\phi}\_{-i}\right\]=\frac{\alpha}{N_i+1-\alpha}\sum\_{j\sim i}\phi\_{j}.](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmathbb%7BE%7D%5Cleft%5B%20%5Cphi_%7Bi%7D%7C%5Cboldsymbol%7B%5Cphi%7D_%7B-i%7D%5Cright%5D%3D%5Cfrac%7B%5Calpha%7D%7BN_i%2B1-%5Calpha%7D%5Csum_%7Bj%5Csim%20i%7D%5Cphi_%7Bj%7D. "\mathbb{E}\left[ \phi_{i}|\boldsymbol{\phi}_{-i}\right]=\frac{\alpha}{N_i+1-\alpha}\sum_{j\sim i}\phi_{j}.")

The Leroux model recovers the *independent* and *ICAR* settings for
![\alpha=0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Calpha%3D0 "\alpha=0")
and
![\alpha=1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Calpha%3D1 "\alpha=1"),
respectively.

### Sparse implementation of the Leroux model

We implement the model in via . Implementation of this model require
computing the determinant and quadratic form of the
![l\times l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l%5Ctimes%20l "l\times l")
matrix
![\boldsymbol{Q}(\alpha, \boldsymbol{W})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BQ%7D%28%5Calpha%2C%20%5Cboldsymbol%7BW%7D%29 "\boldsymbol{Q}(\alpha, \boldsymbol{W})").
This can become computationally expensive in many modrn applications,
that often deal with
![l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l "l")
fairly large (at least
![l\>100](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l%3E100 "l>100")).

This bottleneck is typical of many others CAR specification (proper CAR,
etc.) and it requires the adoption of ad-hoc strategies to estimate the
model in a reasonable amount of time. We can do so exploiting the
inherent sparsity pattern in
![\boldsymbol{Q}(\alpha, \boldsymbol{W})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BQ%7D%28%5Calpha%2C%20%5Cboldsymbol%7BW%7D%29 "\boldsymbol{Q}(\alpha, \boldsymbol{W})"),
deriving from the sparsity in
![\boldsymbol{W}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BW%7D "\boldsymbol{W}").

-   Only few entries of
    ![\boldsymbol{Q}(\alpha, \boldsymbol{W})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BQ%7D%28%5Calpha%2C%20%5Cboldsymbol%7BW%7D%29 "\boldsymbol{Q}(\alpha, \boldsymbol{W})")
    are different from zero and we can significantly speed up the
    evaluation by computing only the non-zero entries
-   We can draw exploit a basic matrix manipulation to derive an
    efficient computational strategy to evaluate the determinant of
    ![\boldsymbol{Q}(\alpha, \boldsymbol{W})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BQ%7D%28%5Calpha%2C%20%5Cboldsymbol%7BW%7D%29 "\boldsymbol{Q}(\alpha, \boldsymbol{W})").
    Indeed, we have that

    ![\boldsymbol{Q}(\alpha,\boldsymbol{W}) = \left(\boldsymbol{I}\_l-\alpha\left(\boldsymbol{W}+\boldsymbol{I}\_l-\boldsymbol{D}\right)\right)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BQ%7D%28%5Calpha%2C%5Cboldsymbol%7BW%7D%29%20%3D%20%5Cleft%28%5Cboldsymbol%7BI%7D_l-%5Calpha%5Cleft%28%5Cboldsymbol%7BW%7D%2B%5Cboldsymbol%7BI%7D_l-%5Cboldsymbol%7BD%7D%5Cright%29%5Cright%29 "\boldsymbol{Q}(\alpha,\boldsymbol{W}) = \left(\boldsymbol{I}_l-\alpha\left(\boldsymbol{W}+\boldsymbol{I}_l-\boldsymbol{D}\right)\right)")

    and, by basic linear algebra properties:

    ![\|\boldsymbol{Q}(\alpha,\boldsymbol{W})\|\propto \prod\_{i=1}^l(1-\alpha\lambda_i),](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%7C%5Cboldsymbol%7BQ%7D%28%5Calpha%2C%5Cboldsymbol%7BW%7D%29%7C%5Cpropto%20%5Cprod_%7Bi%3D1%7D%5El%281-%5Calpha%5Clambda_i%29%2C "|\boldsymbol{Q}(\alpha,\boldsymbol{W})|\propto \prod_{i=1}^l(1-\alpha\lambda_i),")

    where
    ![\lambda_i=1,\dots, l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Clambda_i%3D1%2C%5Cdots%2C%20l "\lambda_i=1,\dots, l")
    are the eigenvalues of
    ![\left(\boldsymbol{W}+\boldsymbol{I}\_l-\boldsymbol{D}\right)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cleft%28%5Cboldsymbol%7BW%7D%2B%5Cboldsymbol%7BI%7D_l-%5Cboldsymbol%7BD%7D%5Cright%29 "\left(\boldsymbol{W}+\boldsymbol{I}_l-\boldsymbol{D}\right)").
    These do not vary from iteration to iteration and can be computed
    only once at the beginning.

### Example

We consider the well-known *lip-cancer* purely spatial data. They
summarise the overall lip cancer cases and possible covariate risk
factors for the period 1975 to 1986, for
![l=56](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l%3D56 "l=56")
districts in Scotland.

We will use the following information:

-   `observed`: number of recorded lip cancer cases
    ![y_s, s=1,\dots,l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;y_s%2C%20s%3D1%2C%5Cdots%2Cl "y_s, s=1,\dots,l").
-   `expected`: expected number of lip cancer cases according to the
    indirect standardisation using Scotland-wide disease rates
    ![o_s, s=1,\dots,l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;o_s%2C%20s%3D1%2C%5Cdots%2Cl "o_s, s=1,\dots,l").
-   `pcaff`: the percentage of the workforce employed in agriculture,
    fishing and forestry. We use it under the `log1p` transform and
    scaling to get
    ![x_s, s=1,\dots,l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_s%2C%20s%3D1%2C%5Cdots%2Cl "x_s, s=1,\dots,l").

We consider a typical Poisson regression with spatial random effects:

![y_s \sim Poi(\lambda_s),\quad \log(\lambda_s)= \log(o_s) + \beta_0 + \beta_1\cdot x_s + \phi_s](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;y_s%20%5Csim%20Poi%28%5Clambda_s%29%2C%5Cquad%20%5Clog%28%5Clambda_s%29%3D%20%5Clog%28o_s%29%20%2B%20%5Cbeta_0%20%2B%20%5Cbeta_1%5Ccdot%20x_s%20%2B%20%5Cphi_s "y_s \sim Poi(\lambda_s),\quad \log(\lambda_s)= \log(o_s) + \beta_0 + \beta_1\cdot x_s + \phi_s")

where
![\phi_s](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cphi_s "\phi_s")
is modeled according to the Leroux model.

The data can be loaded from the `CARBayes` and `CARBayesData` packages.
We will make use also of the other following packages.

We load the data and combine them.

``` r
# Loading raw data
data("lipdata")
data("lipshp")
data("lipdbf")

# Combining data
lipdbf$dbf <- lipdbf$dbf[ ,c(2,1)]
data.combined <- combine.data.shapefile(data=lipdata, shp=lipshp, dbf=lipdbf) %>% st_as_sf()
```

We derive the neighboring structure.

``` r
# Neighboring matrix
Wnb <- poly2nb(data.combined)
W <- nb2mat(Wnb, zero.policy = TRUE, style = "B")
```

Let us visualize the log-relative risks and the log-pcaff covariate.

<img src="VignetteSparseLeroux_files/figure-gfm/unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

We will fit to those data the Leroux and its sparse version, available
in the folder as *Leroux.stan* and *Leroux_Sparse.stan*. The core of the
sparse version is the following likelihood function and the
corresponding determinant computation. The eigen-values
![\lambda_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Clambda_i "\lambda_i")
and the sparse versions `W_sparse` and `ID_sparse` of
![\boldsymbol{W}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BW%7D "\boldsymbol{W}")
and
![(\boldsymbol{I}-\boldsymbol{D})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%28%5Cboldsymbol%7BI%7D-%5Cboldsymbol%7BD%7D%29 "(\boldsymbol{I}-\boldsymbol{D})")
can be computed in advance and given in input or evaluated in the
transformed data section.

``` r
functions {
  real sparse_Leroux_lpdf(vector phi, real alpha, int[,] W_sparse, vector W_weight, 
                          vector ID_sparse, real sumlogdet, int n, int W_n) {
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

data {  ...  }

parameters {  ...  }


model {
        ...
  
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
          phi ~ sparse_Leroux(alpha, W_sparse, W_weight, ID_sparse, sumlogdet, N, W_n);
        }
        
        ...

    }
```

In my implementation the sparse representations are computed in the
`transformed data` section of the STAN code. We then prepare the data
for input in STAN.

``` r
# Data
N <- nrow(data.combined)
O <- data.combined$observed
E <- data.combined$expected
X <- model.matrix(~., 
                  data=lipdata %>% dplyr::select(pcaff) %>% 
                    mutate(across(everything(), function(x) scale(log1p(x)))))

# Data for the Leroux
lipList <- list(N = N,
                Y = O,
                lOff = log(E),
                W = W,
                k = ncol(X),
                X = X)

# Data for the Sparse Leroux
lipListSp <- list(N = N,
                Y = O,
                lOff = log(E),
                W = W,
                W_n = sum(W[upper.tri(W)]>0),
                k = ncol(X),
                X = X)
```

We can then setup the STAN options and compile the STAN code.

``` r
# Setting up STAN options
rstan_options(auto_write = TRUE)
n_iter <- 3000
n_chains <- 1
n_cores <- max(n_chains, parallel::detectCores()-1)

# Stan compilation
stan_CodeLer <- stan_model("Leroux.stan")
stan_CodeLerSp <- stan_model("Leroux_Sparse.stan")
```

Finally, we fit the two versions of the model on the data and compare
the execution times.

``` r
fit_Ler <- sampling(stan_CodeLer, data = lipList, 
                    chains = n_chains, iter = n_iter, 
                    cores = n_cores,
                    pars = c("lm", "P"), include=F, seed=111)
```

    ## 
    ## SAMPLING FOR MODEL 'Leroux' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 0 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 3000 [  0%]  (Warmup)
    ## Chain 1: Iteration:  300 / 3000 [ 10%]  (Warmup)
    ## Chain 1: Iteration:  600 / 3000 [ 20%]  (Warmup)
    ## Chain 1: Iteration:  900 / 3000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 1200 / 3000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 1500 / 3000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 1501 / 3000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 1800 / 3000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 2100 / 3000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 2400 / 3000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 2700 / 3000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 3000 / 3000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 62.174 seconds (Warm-up)
    ## Chain 1:                85.39 seconds (Sampling)
    ## Chain 1:                147.564 seconds (Total)
    ## Chain 1:

``` r
fit_LerSp <- sampling(stan_CodeLerSp, data = lipListSp, 
                      chains = n_chains, iter = n_iter, 
                      cores = n_cores,
                      pars = c("lm"), include=F, seed=111)
```

    ## 
    ## SAMPLING FOR MODEL 'Leroux_Sparse' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 0 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 3000 [  0%]  (Warmup)
    ## Chain 1: Iteration:  300 / 3000 [ 10%]  (Warmup)
    ## Chain 1: Iteration:  600 / 3000 [ 20%]  (Warmup)
    ## Chain 1: Iteration:  900 / 3000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 1200 / 3000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 1500 / 3000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 1501 / 3000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 1800 / 3000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 2100 / 3000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 2400 / 3000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 2700 / 3000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 3000 / 3000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 15.161 seconds (Warm-up)
    ## Chain 1:                22.001 seconds (Sampling)
    ## Chain 1:                37.162 seconds (Total)
    ## Chain 1:

The second version provided the results in way less time
(![\approx 1/5](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Capprox%201%2F5 "\approx 1/5")),
yielding almost identical inferences as shown here below.

``` r
betas <- paste0("beta[", 1:(ncol(X)), "]")
cars <- c("alpha", "sigmac")

# Check divergences
np_Ler <- bayesplot::nuts_params(fit_Ler)
np_LerSp <- bayesplot::nuts_params(fit_LerSp)

# The summary
sumLer <- summary(fit_Ler, pars=c(betas, cars))
sumLerSp <- summary(fit_LerSp, pars=c(betas, cars))
sumLer$summary
```

    ##               mean     se_mean         sd         2.5%        25%        50%
    ## beta[1] 0.09530696 0.001060840 0.05298303 -0.007236078 0.06094519 0.09557272
    ## beta[2] 0.30344290 0.003908092 0.09741796  0.113041659 0.23651718 0.30055624
    ## alpha   0.80894289 0.004316776 0.14345344  0.455623904 0.73392167 0.83878359
    ## sigmac  0.73136963 0.005043477 0.11630023  0.536966086 0.64520088 0.72106682
    ##               75%     97.5%     n_eff      Rhat
    ## beta[1] 0.1310216 0.1999381 2494.4427 0.9993332
    ## beta[2] 0.3690426 0.5011618  621.3675 0.9999277
    ## alpha   0.9213116 0.9881245 1104.3401 1.0004918
    ## sigmac  0.8030810 0.9953625  531.7422 0.9996481

``` r
sumLerSp$summary
```

    ##               mean     se_mean         sd        2.5%        25%        50%
    ## beta[1] 0.09559722 0.001245541 0.05143149 -0.00430493 0.06001818 0.09734034
    ## beta[2] 0.30883579 0.003672934 0.09694188  0.10672862 0.24567725 0.31086608
    ## alpha   0.80381976 0.004590962 0.14272265  0.43924097 0.72384462 0.83680963
    ## sigmac  0.73775894 0.005864718 0.12212468  0.53115625 0.65033980 0.72708606
    ##               75%     97.5%     n_eff      Rhat
    ## beta[1] 0.1317480 0.1938428 1705.0686 0.9993339
    ## beta[2] 0.3729942 0.4948299  696.6210 1.0002986
    ## alpha   0.9121018 0.9846082  966.4477 0.9994187
    ## sigmac  0.8153081 0.9952680  433.6232 1.0006684

``` r
# Extracting pars
pPars <- rstan::extract(fit_Ler, permuted = FALSE, pars=c(betas, cars))
pParsSp <- rstan::extract(fit_LerSp, permuted = FALSE, pars=c(betas, cars))


p1 <- bayesplot::mcmc_combo(pPars,
                      np=np_Ler,
                      off_diag_fun = c("hex"))
p2 <- bayesplot::mcmc_combo(pParsSp,
                      np=np_LerSp,
                      off_diag_fun = c("hex"))

gridExtra::grid.arrange(p1, p2, nrow=1)
```

<img src="VignetteSparseLeroux_files/figure-gfm/unnamed-chunk-9-1.png" style="display: block; margin: auto;" />

## The Leroux-AR model

The Leroux-AR space-time extension connects
![T](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T "T")
temporal slices of spatial random effects $=through a first-order
auto-regressive structure:

![\boldsymbol{\phi}\_1\sim \mathcal{N}\_l \left( \boldsymbol{0},\\; \sigma^2 \cdot \boldsymbol{Q}\left(\alpha, \boldsymbol{W}\right)^{-1}\right)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7B%5Cphi%7D_1%5Csim%20%5Cmathcal%7BN%7D_l%20%5Cleft%28%20%5Cboldsymbol%7B0%7D%2C%5C%3B%20%5Csigma%5E2%20%5Ccdot%20%5Cboldsymbol%7BQ%7D%5Cleft%28%5Calpha%2C%20%5Cboldsymbol%7BW%7D%5Cright%29%5E%7B-1%7D%5Cright%29 "\boldsymbol{\phi}_1\sim \mathcal{N}_l \left( \boldsymbol{0},\; \sigma^2 \cdot \boldsymbol{Q}\left(\alpha, \boldsymbol{W}\right)^{-1}\right)")

![\boldsymbol{\phi}\_t\\,\|\\,\boldsymbol{\phi}\_{t-1},\dots,\boldsymbol{\phi}\_1\sim \mathcal{N}\_l\left( \rho\cdot\boldsymbol{\phi}\_{t-1},\\, \sigma^2\cdot\boldsymbol{Q}\left(\alpha, \boldsymbol{W}\right)^{-1}\right),\\; t=2,\dots,T.](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7B%5Cphi%7D_t%5C%2C%7C%5C%2C%5Cboldsymbol%7B%5Cphi%7D_%7Bt-1%7D%2C%5Cdots%2C%5Cboldsymbol%7B%5Cphi%7D_1%5Csim%20%5Cmathcal%7BN%7D_l%5Cleft%28%20%5Crho%5Ccdot%5Cboldsymbol%7B%5Cphi%7D_%7Bt-1%7D%2C%5C%2C%20%5Csigma%5E2%5Ccdot%5Cboldsymbol%7BQ%7D%5Cleft%28%5Calpha%2C%20%5Cboldsymbol%7BW%7D%5Cright%29%5E%7B-1%7D%5Cright%29%2C%5C%3B%20t%3D2%2C%5Cdots%2CT. "\boldsymbol{\phi}_t\,|\,\boldsymbol{\phi}_{t-1},\dots,\boldsymbol{\phi}_1\sim \mathcal{N}_l\left( \rho\cdot\boldsymbol{\phi}_{t-1},\, \sigma^2\cdot\boldsymbol{Q}\left(\alpha, \boldsymbol{W}\right)^{-1}\right),\; t=2,\dots,T.")

The parameter
![0\<\rho\<1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;0%3C%5Crho%3C1 "0<\rho<1")
is the temporal auto-regressive coefficient, controlling for temporal
dependence. We denote with
![\boldsymbol{\theta}=(\alpha, \rho, \sigma)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7B%5Ctheta%7D%3D%28%5Calpha%2C%20%5Crho%2C%20%5Csigma%29 "\boldsymbol{\theta}=(\alpha, \rho, \sigma)")
the vector of coefficients on which the Leroux-AR specification depends
on.

### The non-centered parametrization

STAN benefits from non-centered parametrization of parameters and random
effects. It improves the geometry of the posterior and facilitates its
exploration. Therefore, when coding the Leroux-AR in STAN it can be
extremely beneficial sampling independent spatial vectors in the
`model block` and then induce the temporal dependence across the spatial
vectors at different times in the `transformed parameters block`. In
practice, we must create some auxiliary vectors
![\boldsymbol{\phi}\_t^\*](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7B%5Cphi%7D_t%5E%2A "\boldsymbol{\phi}_t^*")
such that:

![\boldsymbol{\phi}^\*\_t\\, \sim \\, \mathcal{N}\left(\boldsymbol{0}\\,,\\,\boldsymbol{Q}\left(\alpha, \boldsymbol{W}\right)^{-1}\right), \quad t=1,\dots,T](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7B%5Cphi%7D%5E%2A_t%5C%2C%20%5Csim%20%5C%2C%20%5Cmathcal%7BN%7D%5Cleft%28%5Cboldsymbol%7B0%7D%5C%2C%2C%5C%2C%5Cboldsymbol%7BQ%7D%5Cleft%28%5Calpha%2C%20%5Cboldsymbol%7BW%7D%5Cright%29%5E%7B-1%7D%5Cright%29%2C%20%5Cquad%20t%3D1%2C%5Cdots%2CT "\boldsymbol{\phi}^*_t\, \sim \, \mathcal{N}\left(\boldsymbol{0}\,,\,\boldsymbol{Q}\left(\alpha, \boldsymbol{W}\right)^{-1}\right), \quad t=1,\dots,T")

and then evaluate
![\boldsymbol{\phi}\_1\\, = \\, \boldsymbol{\phi}\_1^{\*}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7B%5Cphi%7D_1%5C%2C%20%3D%20%5C%2C%20%5Cboldsymbol%7B%5Cphi%7D_1%5E%7B%2A%7D "\boldsymbol{\phi}_1\, = \, \boldsymbol{\phi}_1^{*}")
and:

![\quad \boldsymbol{\phi}\_t\\, = \\, \rho\cdot\boldsymbol{\phi}\_{t-1} + \boldsymbol{\phi}\_t^{\*}, \quad t=2,\dots, T](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cquad%20%5Cboldsymbol%7B%5Cphi%7D_t%5C%2C%20%3D%20%5C%2C%20%5Crho%5Ccdot%5Cboldsymbol%7B%5Cphi%7D_%7Bt-1%7D%20%2B%20%5Cboldsymbol%7B%5Cphi%7D_t%5E%7B%2A%7D%2C%20%5Cquad%20t%3D2%2C%5Cdots%2C%20T "\quad \boldsymbol{\phi}_t\, = \, \rho\cdot\boldsymbol{\phi}_{t-1} + \boldsymbol{\phi}_t^{*}, \quad t=2,\dots, T")

### Example

In this application we consider the data about the weekly number of
reported campylobacteriosis cases in Germany in 2011 included in the
`surveillance` package.

We will use the following information:

-   `observed`: number of weekly campylobacteriosis infections in each
    district
    ![y\_{st},\\; s=1,\dots,l,\\; t=1,\dots,T](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;y_%7Bst%7D%2C%5C%3B%20s%3D1%2C%5Cdots%2Cl%2C%5C%3B%20t%3D1%2C%5Cdots%2CT "y_{st},\; s=1,\dots,l,\; t=1,\dots,T").
-   `population`: population size of each district
    ![p_s,\\; s=1,\dots,l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;p_s%2C%5C%3B%20s%3D1%2C%5Cdots%2Cl "p_s,\; s=1,\dots,l").
-   `humidity`: scaled average weekly humidity
    ![x\_{st},\\; s=1,\dots,l,\\; t=1,\dots,T](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_%7Bst%7D%2C%5C%3B%20s%3D1%2C%5Cdots%2Cl%2C%5C%3B%20t%3D1%2C%5Cdots%2CT "x_{st},\; s=1,\dots,l,\; t=1,\dots,T").

We consider a space-time Poisson regression with space-time random
effects

![y\_{st} \sim Poi(\lambda\_{st}),\quad \log(\lambda\_{st})= \log(o_s) + \beta_0 + \beta_1\cdot x\_{st} + \phi\_{st}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;y_%7Bst%7D%20%5Csim%20Poi%28%5Clambda_%7Bst%7D%29%2C%5Cquad%20%5Clog%28%5Clambda_%7Bst%7D%29%3D%20%5Clog%28o_s%29%20%2B%20%5Cbeta_0%20%2B%20%5Cbeta_1%5Ccdot%20x_%7Bst%7D%20%2B%20%5Cphi_%7Bst%7D "y_{st} \sim Poi(\lambda_{st}),\quad \log(\lambda_{st})= \log(o_s) + \beta_0 + \beta_1\cdot x_{st} + \phi_{st}")

where the offset
![o_s](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;o_s "o_s")
is the population size divided by
![10^6](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;10%5E6 "10^6")
and
![\phi\_{st}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cphi_%7Bst%7D "\phi_{st}")
is modeled according to the Leroux-AR model.
