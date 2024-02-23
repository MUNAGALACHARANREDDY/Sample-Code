```python
# Sample-Code
# A simple Python code that uses the MCMC technique to estimate the Tensor-to-Scalar ratio  'r' as it is significant in detecting the primordial gravitational waves.
# The values I used for the parameters in this code are general values for a simplified explanation.
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import numpy as np

# Generate a simple B-mode power spectrum.
def generate_bmode_spectrum(ell, r, A_BB=1.0, alpha=-2.3):
 
C_ell_BB = A_BB * (l / 80.0) ** alpha * r
    
 # Adding some Gaussian noise to simulate measurement errors

noise = np.random.normal(0, 0.1 * C_ell_BB, len(ell)) #len(ell) specifies number of samples generated.
    
return C_ell_BB + noise

# Set the seed for reproducibility

np.random.seed(100)

# Example multipole moment array (ell values)

l = np.arange(2, 301)

# Generate B-mode spectra for a specific value of r

r_example = 0.1      # Here I have given a hypothetical value meant to stimulate a scenario where the effects of these waves are significant enough to be potentially observed in the data. 
spectrum = generate_bmode_spectrum(ell, r_example)

# spectrum now holds the generated data for the given r value
# now defining log-likelihood function.
def
 
log_likelihood(theta,ell, observed_spectrum):
# Unpack the parameters from theta 
r, A_BB, alpha = theta  
 model_spectrum = A_BB * (ell / 80.0) ** alpha * r  
# Predicted B-mode power spectrum without noise
# Assuming that the variance of the observed data is known and constant for simplicity
# In real scenarios, this could be estimated from the data or a more complex model

    sigma2 = 0.1 ** 2
  
# Assuming the noise level used in the data generation
# Compute the log-likelihood

log_likelihood = -0.5* np.sum(((observed_spectrum - model_spectrum) ** 2) / sigma2 + np.log(2 * np.pi * sigma2))
    
return
 log_likelihood

# Example usage

# Note: You should replace 'spectrum' with the actual observed data in practice

theta_example = [0.1, 1.0, -2.3]  
# Example parameter values [r, A_BB, alpha]

log_likelihood_value = log_likelihood(theta_example, ell, synthetic_spectrum)
print
("Log-likelihood:", log_likelihood_value)

