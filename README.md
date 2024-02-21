```python
# Sample-Code
# A simple Python code that uses the MCMC technique to estimate the Tensor-to-Scalar ratio  'r' as it is significant in detecting the primordial gravitational waves.
# The values I used for the parameters in this code are general values for a simplified explanation.
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import numpy as np

def generate_bmode_spectrum(l, r, A_BB=1.0, alpha=-2.3):
 # Generate a B-mode power spectrum.

C_l_BB = A_BB * (l / 80.0) ** alpha * r
    
 # Add some Gaussian noise to simulate measurement errors

noise = np.random.normal(0, 0.1 * C_l_BB, len(l)) #len(l) specifies number of samples generated.
    
return C_l_BB + noise

# Set the seed for reproducibility

np.random.seed(100)

# Example multipole moment array (l values)

l = np.arange(2, 301)

# Generate B-mode spectra for a specific value of r

r_example = 0.1
spectrum = generate_bmode_spectrum(l, r_example)

# spectrum now holds the generated data for the given r value

