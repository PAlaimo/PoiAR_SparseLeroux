rollsum <- function(x, n = 3L){ 
  (cs <- cumsum(x)) - c(-cs[1]*((n-1):0), cs[1:(length(x)-n)])
}
