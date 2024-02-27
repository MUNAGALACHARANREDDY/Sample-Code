'''python'''
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# Generate a simple B-mode power spectrum.
def generate_bmode_spectrum(ell, r, A_BB=1.0, alpha=-2.3):
    C_ell_BB = A_BB * (ell / 80.0) ** alpha * r
    # Adding some Gaussian noise to simulate measurement errors
    noise = np.random.normal(0, 0.1 * C_ell_BB, len(ell))
    return C_ell_BB + noise

# Set the seed for reproducibility
np.random.seed(100)

# Example multipole moment array (ell values)
ell = np.arange(2, 301)

# Generate B-mode spectra for a specific value of r
r_example = 0.1
spectrum = generate_bmode_spectrum(ell, r_example)

# Now defining log-likelihood function.
def log_likelihood(theta, ell, observed_spectrum):
    r, A_BB, alpha = theta
    model_spectrum = generate_bmode_spectrum(ell, r, A_BB, alpha)
    sigma2 = 0.1 ** 2
    return -0.5 * np.sum(((observed_spectrum - model_spectrum) ** 2) / sigma2 + np.log(2 * np.pi * sigma2))

# Defining log-prior distribution
def log_prior(theta):
    r, A_BB, alpha = theta
    if 0.0 <= r <= 1.0 and 0.0 <= A_BB <= 10.0 and -4.0 <= alpha <= 0.0:
        return 0.0  # log(1) = 0, uniform prior
    return -np.inf  # log(0) = -inf, outside the prior range

# Defining Probability Function.
def log_probability(theta, ell, observed_spectrum):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf  # Prior is 0, so posterior is also 0
    return lp + log_likelihood(theta, ell, observed_spectrum)

# Example usage:
theta_initial = [0.1, 1.0, -2.3]  # Initial guesses for r, A_BB, alpha
log_prob_initial = log_probability(theta_initial, ell, spectrum)
print("Initial log probability:", log_prob_initial)

# Initiate and run the MCMC sampler.
ndim = 3  # Number of dimensions (parameters) we're fitting.
nwalkers = 24
initial_guesses = [0.1, 1.0, -2.3]
pos = [initial_guesses + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
nsteps = 5000

# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(ell, spectrum))

# Run the MCMC algorithm
sampler.run_mcmc(pos, nsteps, progress=True)

# Plotting the results
samples = sampler.get_chain(discard=100, thin=15, flat=True)
fig = corner.corner(samples, labels=["r", "A_BB", "alpha"])
plt.show()





