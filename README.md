# A spatio-temporal Poisson Auto-Regressive model for epidemic processes

The Covid-19 pandemic has provided many modeling challenges to investigate, evaluate, and  understand various novel unknown aspects of disease dynamics and public health intervention strategies. Of main interest is the modeling development of the disease infection rate at a (quite) granular spatial resolution when we want to describe the disease dynamics in a large area (such as the whole of England). The model must be flexible, realistic, and general enough to describe multiple processes areal process during a time of rapid ininterventions and government policy changes, both in space and time. 

The empirical model we develope is a joint (over multiple areas) Poisson Auto-Regression that incorporates both temporal and spatial dependence to describe the disease dynamics observed in different areas. The spacetime dependence is captured by two set of space-time random effects governing the growth/reproduction rate of the detected cases and in the baseline/endemic/hidden process. Furthermore, our specification is general enough to include additional covariates that can explain changes in both terms. This provides a framework for evaluating local policy changes over the whole spatial andtemporal domain of the study. 

The model is adopted in a full Bayesian framework and implemented in Stan (Carpenter, 2017). The random effects are modeled through the CAR-AR Leroux model by Rushworth et Al (2014). We provide an efficient implementation of the CAR-AR Leroux prior distribution bases on the sparse representation of the corresponding precision matrix and determinant. We consider the non-centered parametrization to favor the convergence of the No-U-Turn-Sampler (NUTS) implemented in STAN. The model and its estimation has been validated using a substantial simulation study.

Our main application focuses on the weekly COVID-19 cases observed between April 2020 and March 2021 at the local authority district level in England (yet to be published).
We consider two alternative sets of covariates: the level of local restriction in place in terms of *Tier levels* and the value of the *Google Mobility Indices*. They are used in a novel way to determine the best model for epidemic growth by estimating the reported and hidden cases across space and time.  The model detects substantial spatial and temporal heterogeneity due to policy changes and change in disease dynamics and is able to capture various interesting aspects of the disease epidemiology. More details about the applications are available in the main paper.

## The model

Let $\boldsymbol{Y}=\left[ \boldsymbol{y}\_1,\dots,\boldsymbol{y}\_T\right]$ be the matrix of the $n=(T\times L)$ observed cases, with ${\boldsymbol{y}\_{t}}= y\_{\ell 1}, \dots, y\_{\ell T}$, $t=1,\dots,T$, and $\ell\in\mathcal{S}=\left\lbrace \boldsymbol{\ell}\_1,\dots,\boldsymbol{\ell}\_L\right\rbrace$. We consider an extended version of the Poisson auto-regression to account for the epidemiological and spatial nature of the data.

We assume that the counts at each $\ell\in\mathcal{S}$ and time $t$ depends directly on the counts of the previous time through:

$$
\begin{align}
&Y\_{\ell t}\,|\,\boldsymbol{y}\_{\ell (1:t-1)}\sim Poi(\lambda\_{\ell t}),\\
&\lambda\_{\ell t} = \left(\sum\limits\_{i=1}^{\tau} \left( w_i\cdot y\_{\ell (t-i)}\right) \cdot \tilde{r}\_{\ell t} + b\_{\ell t}\right)\cdot d\_{\ell t},
\end{align}
$$

where:
- $\sum\_{i=1}^{\tau} \left( w_i\cdot y\_{\ell (t-i)}\right)$ is a weighted average of the counts observed at the previous $\tau$ times in the same region $\ell$.
- $\tilde{r}\_{\ell t}>0$ and $b\_{\ell t}>0$ are the time-location specific auto-regressive coefficient and baseline, respectively. The former determines the memory of the process, regulating the impact of the previous count on the current one.
- $d\_{\ell t}$ is a factor that discounts the overall rate for the proportion of susceptible individuals at each time and in each location. It is key to temper the growth in smaller regions, where full capacity is reached in shorter rimes. Taken a non-reinfection window of $\tilde{\tau}$ times (i.e. the same individual cannot be reinfected within $\tilde{\tau}$ times) it amounts to:
$$d\_{\ell t} = \frac{\sum\limits\_{j=1}^{\tilde{t}} y\_{\ell (t-j)}}{pop\_{\ell}},\quad \forall \\, \ell\in\mathcal{S}, \quad t=1,\dots, T,$$
where $pop\_{\ell}$ is the population size of region $\ell\in\mathcal{S}$.

Our main interest lies in modeling the auto-regressive coefficient, which could be expressed as a function of covariates and random effects through the *log-link* function
$$\log\left(\tilde{r}\_{\ell t}\right)= \boldsymbol{x}\_{\ell t}^\top\cdot\boldsymbol{\beta} + \phi\_{\ell t},$$
where $\boldsymbol{x}\_{\ell t}$ is a $k\times 1$ vector of (potentially space-time varying) covariates, $\boldsymbol{\beta}$ is a vector of $k$ coefficients, and $\left\lbrace\phi_{\ell t}\right\rbrace_{\ell t}^{LT}$ is a set of space-time correlated random effects. 

Very similarly, we can model the baseline on the log-scale. To make it scale-invariant it shall depend on a location specific offset $\text{off}\_\ell,\\, \ell\in\mathcal{S}$:
$$\log\left( b\_{\ell t}\right)=\log\left(\text{off}\_{\ell }\right)+\boldsymbol{v}\_{\ell t}^\top\cdot\boldsymbol{\eta}+\psi\_{\ell t},$$
where $\boldsymbol{v}\_{\ell t}$ is a $\nu\times 1$ vector of (potentially space-time varying) covariates, $\boldsymbol{\eta}$ is the corresponding vector of coefficients, and $\psi\_{\ell t}$ is a second set of space-time correlated random effects. The latter allows explaining any extra-variability in the process and account for over-dispersion with respect to the Poisson assumption.

Let $\boldsymbol{W}$ be the adjacency matrix of the $L$ regions, such that $w\_{ij}>0$ if and only if $i\sim j$. Let $\boldsymbol{D}$ be the diagonal matrix such that $d\_{ii}=\sum\limits\_{j}w\_{ij}$.
The two sets of space-time random effects $\phi\_{\ell t}, \psi\_{\ell t}$ are modeled through the space-time extension of the Leroux model by Rushworth et Al (2014). It connects the $T$ time slices through a first-order auto-regressive structure:

$$
\begin{align}
        &\boldsymbol{\phi}\_1\sim \mathcal{N}_l\left( \boldsymbol{0}, \sigma^2\cdot\boldsymbol{Q}\left(\alpha, \boldsymbol{W}\right)^{-1}\right)\\
        &\boldsymbol{\phi}\_t \| \boldsymbol{\phi}\_{t-1},\dots,\boldsymbol{\phi}\_1\sim \mathcal{N}\_L\left( \rho\cdot\boldsymbol{\phi}\_{t-1}, \sigma^2\cdot\boldsymbol{Q}\left(\alpha, \boldsymbol{W}\right)^{-1}\right), t=2,\dots,T,      
\end{align}
$$

where $\boldsymbol{Q}(\alpha, \boldsymbol{W})=\left(\alpha(\boldsymbol{D}-\boldsymbol{W})+(1-\alpha)\boldsymbol{I}\_L\right)$ is the precision matrix of the Leroux prior, $0<\alpha<1$ is a spatial smoothing parameter and $0<\rho<1$ is the temporal auto-regressive coefficient.
More details on this spatial and spatio-temporal specification, with the corresponding sparse and efficient implementation in STAN, are provided in the dedicated folder of this project.

## Content of the Github project

In this project we provide further details and examples in the following folders.
- **CAR-AR**
  - Details about the sparse implementation of the CAR Leroux model
  - Details about the implementation of the space-time CAR-AR Leroux model by Rushworth et Al (2014) in STAN through the non-centered parametrization
- **STPoiAR**
  - Application of the proposed space-time Poisson Auto-Regressive model on a set of simulated data
  - Application on the real data (will be available at the time of the paper publication)

## Biblio

- Carpenter, Bob, Gelman, Andrew, Hoffman, Matthew D., Lee, Daniel, Goodrich, Ben, Betancourt, Michael, Brubaker, Marcus, Guo, Jiqiang, Li, Peter, and Riddell, Allen. Stan : A Probabilistic Programming Language. United States: N. p., 2017. Web. doi:10.18637/jss.v076.i01. 
- Rushworth, Alastair, Duncan Lee, and Richard Mitchell. "A spatio-temporal model for estimating the long-term effects of air pollution on respiratory hospital admissions in Greater London." Spatial and spatio-temporal epidemiology 10 (2014): 29-38.
