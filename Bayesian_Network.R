# Mitchel Carson
# CS 4440: Artificial Intelligence
# Dr. Mohammad Javidian
# Appalachian State University Computer Science


#                 Programming Assignment: Bayesian Networks


# Packages
library(bnlearn)  # For structure learning, adjacency matrix, d-separation
library(gRain)    # For exact inference
library(gRbase)   # Utility for building CPTs




# PART 1: D-Separation using the HailFinder Dataset

# Load the HailFinder dataset
data("hailfinder")

# Learn the Bayesian Network structure from data (Hill-Climbing)
dag_hailfinder <- hc(hailfinder)

# Extract the adjacency matrix from the learned DAG
adjMat_hailfinder <- amat(dag_hailfinder)

# Print the adjacency matrix
cat("Adjacency Matrix of the Learned HailFinder DAG:\n")
print(adjMat_hailfinder)
cat("\n")

# Visualize the dag with graphviz
graphviz.plot(dag_hailfinder)

# Perform a d-separation test
dsep_result <- dsep(dag_hailfinder, "TempDis", "WindFieldPln", "MeanRH")
cat("D-separation test (TempDis _|_ WindFieldPln | MeanRH):", dsep_result, "\n\n")







# PART 2: Exact Inference on the Burglary-Earthquake Network (Variable Elimination)

# Define variable levels
levels_var <- c("TRUE", "FALSE")

# Define Conditional Probability Tables (CPTs) using cptable
cptB <- cptable(~B, values = c(0.001, 0.999), levels = levels_var)
cptE <- cptable(~E, values = c(0.002, 0.998), levels = levels_var)
# CPT for Alarm given Burglary (B) and Earthquake (E)
# Probabilities for Alarm = TRUE are given by:
#   P(A=TRUE|B=TRUE,E=TRUE) = 0.95, P(A=TRUE|B=TRUE,E=FALSE) = 0.94,
#   P(A=TRUE|B=FALSE,E=TRUE) = 0.29, P(A=TRUE|B=FALSE,E=FALSE) = 0.001.
cptA <- cptable(~A|B:E, values = c(95, 5, 94, 6, 29, 71, 1, 999), levels = levels_var)
# CPT for JohnCalls given Alarm
cptJ <- cptable(~J|A, values = c(90, 10, 5, 95), levels = levels_var)
# CPT for MaryCalls given Alarm
cptM <- cptable(~M|A, values = c(70, 30, 1, 99), levels = levels_var)

# Compile CPTs and create a gRain Bayesian network
plist <- compileCPT(list(cptB, cptE, cptA, cptJ, cptM))
bn_grain <- grain(plist)

# Set evidence: JohnCalls = TRUE, MaryCalls = TRUE
bn_grain_ev <- setEvidence(bn_grain, evidence = list(J = "TRUE", M = "TRUE"))

# Query the probability of Earthquake given the evidence
query_exact <- querygrain(bn_grain_ev, nodes = "E")$E
cat("Exact Inference: P(Earthquake | JohnCalls, MaryCalls):\n")
# Print the solution
print(query_exact)
cat("\n")





# 3. Approximate Inference via Gibbs Sampling

# Probability helper functions
prob_B <- function(b) {
  if (b == "TRUE") 0.001 else 0.999
}

prob_E <- function(e) {
  if (e == "TRUE") 0.002 else 0.998
}

prob_A <- function(a, b, e) {
  # A depends on B and E
  if (b == "TRUE" && e == "TRUE") {
    p_true <- 0.95
  } else if (b == "TRUE" && e == "FALSE") {
    p_true <- 0.94
  } else if (b == "FALSE" && e == "TRUE") {
    p_true <- 0.29
  } else { # b == "FALSE" && e == "FALSE"
    p_true <- 0.001
  }
  if (a == "TRUE") p_true else (1 - p_true)
}

prob_J <- function(j, a) {
  # J depends on A
  p_true <- if (a == "TRUE") 0.90 else 0.05
  if (j == "TRUE") p_true else (1 - p_true)
}

prob_M <- function(m, a) {
  # M depends on A
  p_true <- if (a == "TRUE") 0.70 else 0.01
  if (m == "TRUE") p_true else (1 - p_true)
}

gibbs_sampling <- function(n_iter = 10000, burn_in = 1000) {
  # Non-evidence variables: B, E, A
  # Evidence: J="TRUE", M="TRUE"
  
  # Initialize the Markov chain randomly
  state <- list(
    B = sample(c("TRUE", "FALSE"), 1),
    E = sample(c("TRUE", "FALSE"), 1),
    A = sample(c("TRUE", "FALSE"), 1),
    J = "TRUE",  # evidence
    M = "TRUE"   # evidence
  )
  
  samples <- vector("list", n_iter)
  
  for (iter in seq_len(n_iter)) {
    # Update B (Markov blanket for B is {B, A, E})
    probs_B <- sapply(c("TRUE", "FALSE"), function(b) {
      prob_B(b) * prob_A(state$A, b, state$E)
    })
    probs_B <- probs_B / sum(probs_B)
    state$B <- sample(c("TRUE", "FALSE"), 1, prob = probs_B)
    
    # Update E (Markov blanket for E is {E, A, B})
    probs_E <- sapply(c("TRUE", "FALSE"), function(e) {
      prob_E(e) * prob_A(state$A, state$B, e)
    })
    probs_E <- probs_E / sum(probs_E)
    state$E <- sample(c("TRUE", "FALSE"), 1, prob = probs_E)
    
    # Update A (Markov blanket for A is {A, B, E, J, M})
    probs_A <- sapply(c("TRUE", "FALSE"), function(a) {
      prob_A(a, state$B, state$E) *
        prob_J("TRUE", a) *
        prob_M("TRUE", a)
    })
    probs_A <- probs_A / sum(probs_A)
    state$A <- sample(c("TRUE", "FALSE"), 1, prob = probs_A)
    
    samples[[iter]] <- state
  }
  
  # Discard burn-in
  valid_samples <- samples[(burn_in + 1):n_iter]
  
  # Estimate P(E=TRUE)
  e_values <- sapply(valid_samples, function(s) s$E)
  mean(e_values == "TRUE")
}

set.seed(469)
gibbs_estimate <- gibbs_sampling(n_iter = 10000, burn_in = 1000)
cat("Gibbs Sampling (10k samples, 10% burn-in) estimate for P(E=TRUE | J=TRUE, M=TRUE) =", 
    gibbs_estimate, "\n")

# PART 4: Validation (Comparing Exact vs. Approximate)

# Compare the approximate result (Gibbs) with the exact result (gRain)
exact_value <- query_exact["TRUE"]
approx_value <- gibbs_estimate
difference <- abs(exact_value - approx_value)

cat("Validation:\n")
cat("  Exact P(E=TRUE|J=TRUE,M=TRUE)     =", exact_value, "\n")
cat("  Approx. (Gibbs) P(E=TRUE|J=TRUE,M=TRUE) =", approx_value, "\n")