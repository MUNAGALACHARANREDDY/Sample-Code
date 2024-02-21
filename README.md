```python
# Sample-Code
# A simple Python code that uses the MCMC technique to estimate the Tensor-to-Scalar ratio  'r' as it is significant in detecting the primordial gravitational waves.
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import numpy as np

def generate_bmode_spectrum(ell, r, A_BB=1.0, alpha=-2.3):
 # Generate a B-mode power spectrum.
# Basic B-mode power spectrum model (simplified)

C_ell_BB = A_BB * (ell / 80.0) ** alpha * r
    
 # Add some Gaussian noise to simulate measurement errors

noise = np.random.normal(0, 0.1 * C_ell_BB, len(ell))
    
return C_ell_BB + noise

# Set the seed for reproducibility

np.random.seed(100)

# Example multipole moment array (ell values)

ell = np.arange(2, 301)

# Generate B-mode spectra for a specific value of r

r_example = 0.1
spectrum = generate_bmode_spectrum(ell, r_example)

# spectrum now holds the generated data for the given r value

