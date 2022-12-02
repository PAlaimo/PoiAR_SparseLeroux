
# Packages ----------------------------------------------------------------

require(tidyverse)
require(magrittr)
require(lubridate)
require(zeallot)
require(rstan)
require(bayesplot)
require(loo)
set.seed(130494)

source("AuxFuns/RollSum.R")

# Reading Data ------------------------------------------------------------

# Loading the simulated data: short window 2020/09/25-202/12/11
load("Data/London_SimDataShort.RData")

# They have been simulated with memory tau=3
print(tau)
# Non-reinfection window ttilde=30 weeks
print(slen)
# The log-offset is contained in the object lOff: it is log(popn/10000)
head(lOff)
all(log(dat$popn/10000)==lOff)
# The adjacency matrix is in the data WS
W[1:10, 1:10]

# Manipulating data -------------------------------------------------------

# We need to keep:
dat1 <- dat %>% select(Y,                        # The Ys
                       Ylag1, Ylag2, Ylag3,      # Their lags up to tau=3
                       t, mapid,                 # The times and locations
                       popn,                     # Population sizes
                       popden, jsa, houseprice,  # The time-constant covariates in X
                       LP) %>%                   # Local policies in X
  arrange(t, as.numeric(mapid))                  # We arrange everything by time and location

# Data preparation --------------------------------------------------------

# Cases
Y <- as.integer(dat1$Y)                             # The outcome
Yprev <- dat1 %>% select(Ylag1:Ylag3) %>% as.matrix # The available lags
tau <- ncol(Yprev)                                 # The size of the lags (even if we did not know tau)

# Sizes and indices
N <- as.integer(length(Y))                         # Total number of observations
Ntimes <- as.integer(length(unique(dat1$t)))        # Total number of times (weeks)
Ndis <- as.integer(length(unique(dat1$mapid)))      # Total number of areas
tdId <- matrix(as.integer(1:(Ndis*Ntimes)), 
               ncol=Ndis, nrow=Ntimes, byrow = T)  # Each row t has the indices of the full data corresponding at time t

# Split train and test
NOut <- floor(0.2*N)                      # Number of out-of-sample points
idOut <- sort(sample(1:N, size = NOut))   # Sampling the out-of-sample indices
idIn <- (1:N)[-idOut]                     # Corresponding in-sample indices

Nin <- length(idIn)                       # Size in sample
Nout <- length(idOut)                     # Size out-of-sample

tr <- (1:N)%in%idIn                       # Train T or F

# Covariates on the auto-regressive coefficient (as they were used in the simulation)
XtoScale <- dat1 %>% 
  mutate(logpopden=log(popden), logjsa=log(jsa), loghprice=log(houseprice)) %>% 
  select(logpopden, logjsa, loghprice) %>%     # Log and scaled
  scale()
otherX <- dat1 %>% select(LP) %>% 
  mutate(LP=ifelse(LP%in%c("0", "T1"), "T01", LP)) %>%
  mutate(across(everything(), as.factor))      # Local policies (T1, T2, T3, T4)
X <- model.matrix(~., data=bind_cols(XtoScale, otherX))[,-1] # Design matrix
k <- ncol(X)

# Covariates on baseline: no covariates
V <- model.matrix(~1, data=dat1)
l <- ncol(V)

# Reverse-engineer the log-offset
lOff <- dat1 %>% mutate(popn=log(popn/10000)) %>% pull(popn)

# Reverse-engineer the factor to control for the explosion
# Estimated susceptibles at each location and time, 
# given non-reinfection window of slen=30 
lsN <- dat1 %>% group_by(mapid) %>% 
  mutate(inf=(Ylag1+Ylag2+rollsum(Ylag3, n=slen-2)), 
         sus=(popn-inf), 
         lsN=log(sus/popn)) %>% pull(lsN)

# Remove useless data structures
rm(XtoScale, otherX)

# Stan model compilation and setup -----------------------------------------

# Stan options
rstan_options(auto_write = TRUE)
# Stan compilation
# No worries about the warnings: the constrain to zero has jacobian=1
stan_Code <- stan_model("STANCode/ING2LerouxWC.stan")

# Detecting available cores on the machine
mc.cores <- parallel::detectCores(logical = T)

# Stan set-up
n_chains <- 1   # Number of chains
M <- 4000       # Number of iterations
thin <- 2       # Thinning (just to save memory)

n_cores <- min(n_chains, mc.cores-1)


# Data prep for STAN and fit ----------------------------------------------

# Convert data to STAN input format
stanDat <- list(
  "N" = N, 
  "Y" = Y, 
  "tau" = tau,
  "Yprev" = Yprev,
  "W" = as.matrix(W),
  "W_n" = sum(W[upper.tri(W)]>0),
  "Ndis" = Ndis,
  "Ntimes" = Ntimes,
  "Nin" = Nin,
  "Nout" = Nout,
  "idIn" = idIn,
  "idOut" = idOut,
  "tdId" = tdId,
  "k" = k,
  "X" = X,
  "l" = l,
  "V" = V,
  "lsN" = lsN,
  "lOff" = lOff)

# Function to generate reasonable initial values
init <- function(chain_id = 1)
{
  list(# Coefficients
    beta = array(rnorm(k, 0, .1), dim=k), 
    eta = array(rnorm(l, 0, .5), dim=l), 
    # AR
    rho1 = runif(1, .1, .9),
    rho2 = runif(1, .1, .9),
    # CAR
    alpha1 = runif(1, .1, .9),
    alpha2 = runif(1, .1, .9),
    # Random effects
    sigmac1 = abs(rnorm(1, 0, .1)),
    sigmac2 = abs(rnorm(1, 0, .5)),
    phi1s = matrix(rnorm(N, 0, .1), nrow=Ntimes, ncol=Ndis),
    phi2s = matrix(rnorm(N, 0, .5), nrow=Ntimes, ncol=Ndis))
}

# Fit
fit_Stan <- sampling(stan_Code, 
                     data = stanDat, init=init, 
                     chains = n_chains, iter = M, thin=2, 
                     cores = n_cores,
                     pars = c("phi1s", "phi2s", "lm"), include=F,
                     control = list(adapt_delta=0.9)
)

# Results -----------------------------------------------------------------

# Load the truth
load("Data/London_SimDataShort_Truth.RData")

betas <- paste0("beta[", 1:k, "]")
etas <- paste0("eta[", 1:l, "]")
ws <- paste0("w[", 1:tau, "]")
thetas <- c("alpha1", "alpha2", "rho1", "rho2", "sigmac1", "sigmac2")

sumfit <- summary(fit_Stan, pars=c("lr0", betas, etas, ws, thetas))$summary


# Check params ------------------------------------------------------------

# Chains
postPars <- extract(fit_Stan, pars=c("lr0", betas, etas, ws, thetas),
                    permuted=F)

# Visualization
mcmc_combo(x=postPars, pars=betas)
mcmc_combo(x=postPars, pars=etas)
mcmc_combo(x=postPars, pars=thetas)
mcmc_combo(x=postPars, pars=ws)

mcmc_intervals(postPars, pars="lr0", prob_outer = 0.95) + 
  geom_vline(xintercept = lr0star)
mcmc_intervals(postPars, pars=betas, prob_outer = 0.95) + 
  geom_vline(xintercept = betastar)
mcmc_intervals(postPars, pars=etas, prob_outer = 0.95) + 
  geom_vline(xintercept = etastar)
mcmc_intervals(postPars, pars=thetas, prob_outer = 0.95) + 
  geom_vline(xintercept = thetastar)
mcmc_intervals(postPars, pars=ws, prob_outer = 0.95) + 
  geom_vline(xintercept = wstar)


# Check predictions -------------------------------------------------------

# Chains
postPars2 <- extract(fit_Stan, pars=c("lr0", betas, etas, ws, thetas))

# Extracting random effects
out_Stan <- rstan::extract(fit_Stan, permuted = TRUE, pars=c("phi1", "phi2"))

indices <- tibble(s=dat1$mapid, tStan=dat1$t, tr=tr)
postphi1 <- out_Stan$phi1 %>% 
  as_tibble() %>% 
  summarise(across(everything(), quantile, probs=c(0.025, 0.5, 0.975))) %>% 
  t() %>% as_tibble() %>% set_colnames(c("q025", "q50", "q975")) %>% 
  bind_cols(phihat = colMeans(out_Stan$phi1))

postphi2 <- out_Stan$phi2 %>% 
  as_tibble() %>% 
  summarise(across(everything(), quantile, probs=c(0.025, 0.5, 0.975))) %>% 
  t() %>% as_tibble() %>% set_colnames(c("q025", "q50", "q975")) %>% 
  bind_cols(phihat = colMeans(out_Stan$phi2))

rst <- exp(c(postPars2$lr0) + reduce(postPars2[betas], cbind) %*% t(X) + out_Stan$phi1)
postrst <-  rst %>% 
  as_tibble() %>% 
  summarise(across(everything(), quantile, probs=c(0.025, 0.5, 0.975))) %>% 
  t() %>% as_tibble() %>% set_colnames(c("q025", "q50", "q975")) %>% 
  bind_cols(rsthat = colMeans(rst))

bst <- exp(t(lOff + t(reduce(postPars2[etas], cbind) %*% t(V) + 
                        out_Stan$phi2)))
rm(out_Stan)
postbst <-  bst %>% 
  as_tibble() %>% 
  summarise(across(everything(), quantile, probs=c(0.025, 0.5, 0.975))) %>% 
  t() %>% as_tibble() %>% set_colnames(c("q025", "q50", "q975")) %>% 
  bind_cols(bsthat = colMeans(bst))

ast <- (reduce(postPars2[ws], cbind) %*% t(Yprev)) * rst
rm(rst, postPars2)
postast <-  ast %>% 
  as_tibble() %>% 
  summarise(across(everything(), quantile, probs=c(0.025, 0.5, 0.975))) %>% 
  t() %>% as_tibble() %>% set_colnames(c("q025", "q50", "q975")) %>% 
  bind_cols(asthat = colMeans(ast))

pst <- ast/(ast+bst)
postpst <-  pst %>% 
  as_tibble() %>% 
  summarise(across(everything(), quantile, probs=c(0.025, 0.5, 0.975))) %>% 
  t() %>% as_tibble() %>% set_colnames(c("q025", "q50", "q975")) %>% 
  bind_cols(psthat = colMeans(pst))

lm <- t(log(t(ast + bst)) + lsN)
rm(ast, bst)

postPredsMat <- matrix(rpois(length(exp(lm)), lambda = exp(lm)),
                       nrow=nrow(lm), ncol=ncol(lm))
rm(lm)
postPreds <- postPredsMat %>% 
  as_tibble() %>% 
  summarise(across(everything(), quantile, probs=c(0.025, 0.5, 0.975))) %>% 
  t() %>% as_tibble() %>% set_colnames(c("q025", "q50", "q975")) %>% 
  bind_cols(Yhat = colMeans(postPredsMat))
rm(postPredsMat)

# Visualization
js <- sample(unique(dat1$mapid), size = 6)

coeff <- 10000
postPreds %>% bind_cols(dat, tr=tr) %>% 
  filter(mapid%in%js) %>% ggplot() + 
  geom_ribbon(aes(x=wdate, ymin=q025/popn*coeff, ymax=q975/popn*coeff), 
              alpha=0.5, fill="lightblue", col="grey") +
  geom_line(aes(x=wdate, y=q50/popn*coeff)) + 
  geom_point(aes(x=wdate, y=Y/popn*coeff, fill=tr), shape=21,
             size=1.5) + 
  scale_fill_viridis_d(direction = -1, option="plasma") +
  labs(x="Date", y="Case rate X 10000", fill="Training set") + 
  facet_wrap(~areaName) +
  theme_bw()
