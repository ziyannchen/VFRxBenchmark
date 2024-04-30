from scipy.stats import spearmanr, pearsonr


def calculate_srcc(targets, refs):
    rho_s, _ = spearmanr(targets, refs)
    return rho_s

def calculate_plcc(targets, refs):
    rho_p, _ = pearsonr(targets, refs)
    return rho_p


def calcultae_correlations(targets, refs):
    srcc = calculate_srcc(targets, refs)
    plcc = calculate_plcc(targets, refs)

    print(f'Correlation score SRCC: {srcc:.4}; PLCC: {plcc:.4}')
    return srcc, plcc