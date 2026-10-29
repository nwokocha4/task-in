# Dispersion specification



The forward model is a Gaussian plume superposition by stability class with a Briggs-style plume rise that depends on source exit velocity and emission rate. Pasquill-Gifford sigma curves sigma_y(x), sigma_z(x) follow the standard stability-class dependence; coefficients are supplied in docs/sigma_table.csv. Plume rise follows the Briggs form as referenced in EPA guidance, with the unstable/neutral and stable branches distinguished. The solver package dispersion_model is not shipped; construct the operator from this specification. Exposure averaging period is 1 hour. Receptor list and output ordering are as in network.npz receptor_xy. Short-term exposure limit is 50 ug/m3. Admissible emissions are within the prior support (prior.json defines log-emission mean and sigma; theta_j is mean sigma 3 sigma).

