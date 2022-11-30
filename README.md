# A spatio-temporal Poisson Auto-Regressive model for epidemic processes

The Covid-19 pandemic has provided many modeling challenges to investigate, evaluate, and  understand various novel unknown aspects of disease dynamics and public health intervention strategies. Of main interest is the modeling development of the disease infection rate at a (quite) granular spatial resolution when we want to describe the disease dynamics in a large area (such as the whole of England). The model must be flexible, realistic, and general enough to describe multiple processes areal process during a time of rapid ininterventions and government policy changes, both in space and time. 

The empirical model we develope is a joint (over multiple areas) Poisson Auto-Regression that incorporates both temporal and spatial dependence to describe the disease dynamics observed in different areas. The spacetime dependence is captured by two set of space-time random effects governing the growth/reproduction rate of the detected cases and in the baseline/endemic/hidden process. Furthermore, our specification is general enough to include additional covariates that can explain changes in both terms. This provides a framework for evaluating local policy changes over the whole spatial andtemporal domain of the study. 

The model is adopted in a full Bayesian framework and implemented in Stan (Carpenter, 20). The random effects are modeled through the CAR-AR Leroux model by Rushworth et Al (2014). We provide an efficient implementation of the CAR-AR Leroux prior distribution bases on the sparse representation of the corresponding precision matrix and determinant. We consider the non-centered parametrization to favor the convergence of the No-U-Turn-Sampler (NUTS) implemented in STAN. The model and its estimation has been validated using a substantial simulation study.

Our main application focuses on the weekly COVID-19 cases observed between April 2020 and March 2021 at the local authority district level in England (yet to be published).
We consider two alternative sets of covariates: the level of local restriction in place in terms of *Tier levels* and the value of the *Google Mobility Indices*. They are used in a novel way to determine the best model for epidemic growth by estimating the reported and hidden cases across space and time.  The model detects substantial spatial and temporal heterogeneity due to policy changes and change in disease dynamics and is able to capture various interesting aspects of the disease epidemiology. More details about the applications are available in the main paper.

In this project we provide: 
- Details about the sparse implementation of the CAR Leroux model
- Details about the implementation of the space-time CAR-AR Leroux model by Rushworth et Al (2014) in STAN through the non-centered parametrization
- Application of the proposed space-time Poisson Auto-Regressive model on a set of simulated data


## Biblio

- Carpenter, Bob, Gelman, Andrew, Hoffman, Matthew D., Lee, Daniel, Goodrich, Ben, Betancourt, Michael, Brubaker, Marcus, Guo, Jiqiang, Li, Peter, and Riddell, Allen. Stan : A Probabilistic Programming Language. United States: N. p., 2017. Web. doi:10.18637/jss.v076.i01. 
- Rushworth, Alastair, Duncan Lee, and Richard Mitchell. "A spatio-temporal model for estimating the long-term effects of air pollution on respiratory hospital admissions in Greater London." Spatial and spatio-temporal epidemiology 10 (2014): 29-38.
