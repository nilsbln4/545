"""Settings you fill in yourself. The scripts do not choose these for you."""

# Problem 3: de-mean each return series before fitting? (The exam text does not
# say to; set True if you decide to.)
DEMEAN_Q3 = False

# Problem 3(b) and 3(c): the distribution used for each stock's margin.
# Use "normal", "t" or "nig". Leave None until you have decided.
DISTRIBUTION_CHOICE = {
    "S1": "normal",
    "S2": "t",
    "S3": "nig",
    "S4": "normal",
    "S5": "t",
    "S6": "nig",
    "S7": "normal",
    "S8": "t",
    "S9": "nig",
    "S10": "t",
}

# Problem 3(c): the copula used for the simulation: "gaussian" or "t".
COPULA_CHOICE = "gaussian"

# Problem 3(c): number of simulated days (the exam requires at least 5,000).
N_SIM = 20_000
SEED = 545

# Problem 2: number of simulated days for the simulation variants.
N_SIM_Q2 = 200_000
