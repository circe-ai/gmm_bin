import numpy as np
import argparse
import csv
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt

def csvreader(inputfile):
    """Reads a CSV file into a NumPy array."""
    out = []
    with open(inputfile, newline = '') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            out.append(row)
    return np.array(out)

def csvwrite(file,data,labels=None):
    """Writes a 2D array to a CSV file, optionally including a header row."""
    with open(file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if labels is not None:
            csv_writer.writerow(labels)
        for row in data:
            csv_writer.writerow(row)

def select_gmm_components(data, min_comp=3, max_components=10,model='bic'):
    """
    Iterates through a range of component counts to find the optimal GMM fit.
    Returns the best model and its component count based on BIC (or AIC).
    """
    data = data.reshape(-1, 1)

    evals = []
    a_evals = []    
    models = []

    for k in range(min_comp, max_components + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=33)
        gmm.fit(data)
        evals.append(gmm.bic(data))
        if model=='aic':
            a_evals.append(gmm.aic(data))
        models.append(gmm)

    best_idx = np.argmin(evals)
    return models[best_idx], best_idx + min_comp#, evals

def fit_gmm(data,n_comp=3):
    """Fits a simple GMM with a specified number of components."""
    gmm=GaussianMixture(n_components=n_comp)
    gmm.fit(data.reshape(-1,1))
    return gmm

def assign_bins_from_gmm(model, data):
    """
    Assigns data to GMM components and sorts the labels based on component means,
    ensuring labels logically progress from lowest to highest value distributions.
    """
    data = data.reshape(-1, 1)
    #return model.predict(data)
    raw_labels=model.predict(data)
    means = model.means_.flatten()

    # sort components by mean (low → high energy)
    order = np.argsort(means)

    # mapping: old_label → new_ordered_label
    label_map = {old: new for new, old in enumerate(order)}
    return np.vectorize(label_map.get)(raw_labels)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Automated multi-Gaussian data binning utility.")

    parser.add_argument(
        "-data_path", type=str, required=True, 
        help="Path to data (rows-samples; columns-features)"
    )

    parser.add_argument(
        "--labeled", type=bool, default=False, 
        help="Pass True if the data has the first row as feature labels"
    )
    
    parser.add_argument(
        "--output", type=str, default='./binned_output.csv', 
        help="Path to output file to write binned data"
    )

    parser.add_argument(
        "--min_comp", type=int, default=3,
        help="Minimum components for binning"
    )

    parser.add_argument(
        "--max_comp", type=int, default=8,
        help="Maximum components for binning"
    )        
    args=parser.parse_args()

    ### Begin Binning Procdure ###

    # 1. Read datafile and handle headers ##
    out=csvreader(args.data_path)

    if args.labeled==True:
        labels=out[0]
        data=out[1:].astype(float)
    
    if args.labeled==False:
        data=out.astype(float)

    # 2. Run binning for all features (columns) and determine optimal bin number via BIC 
    B=[select_gmm_components(data[:,i],min_comp=args.min_comp,max_components=args.max_comp) for i in range(data.shape[1])]

    # 3. Assign sorted bin labels to the data using the optimal GMM models
    L=np.array([assign_bins_from_gmm(B[i][0],data[:,i]) for i in range(data.shape[1])]).T

    # 4. Write final binned dataset to output file
    if args.labeled==True:
        csvwrite(args.output,np.vstack((labels,L)))

    else:
        csvwrite(args.output,L)

    ### END Binning Proceudre ###




### Below here is utility functions for plotting/testing etc ###

def get_pdf(gmm,x):
    """Calculates the combined Probability Density Function for a fitted GMM."""
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_
    pdf = np.zeros_like(x)
    for mean, var, weight in zip(means, variances, weights):
        pdf += weight * norm.pdf(x, mean, np.sqrt(var)) 
    return pdf

def plot_bins(data,n_comps=3,ax=None):
    """Plots a histogram of the data overlaid with the GMM PDF and shaded bin regions."""
    dt=np.copy(data)
    
    gmm=fit_gmm(data,n_comps)

    x = np.linspace(dt.min(), dt.max(), 1000)
    pdf=get_pdf(gmm,x)
#    x_labels = gmm.predict(x.reshape(-1, 1))
    x_labels=assign_bins_from_gmm(gmm, x)
    if ax is None:
        fig,ax=plt.subplots(1,1)
    ax.hist(dt, bins='auto', density=True, color='black',alpha=0.7)

# shade regions
    for k in range(gmm.n_components):
        mask = x_labels == k
        ax.fill_between(x[mask], 0, 1, alpha=0.5)
    
    ax.plot(x, pdf, 'r-')
    ax.set_ylim(0,max(pdf)+np.std(pdf))
    plt.show()


def labels_to_edges(data, labels):
    """Converts discrete label assignments into continuous bin edges."""
    edges = []
    for k in np.unique(labels):
        subset = data[labels == k]
        edges.append((subset.min(), subset.max()))
    
    edges = sorted(edges)
    
    # convert to continuous edges
    bin_edges = [edges[0][0]]
    for _, right in edges:
        bin_edges.append(right)
    
    return np.array(bin_edges)


def plot_bins_maxent(data, max_ent_labels):
    """Plots histogram and shaded regions for externally provided maximum entropy labels."""
    dt = np.copy(data)

    x = np.linspace(dt.min(), dt.max(), 1000)

    from scipy.stats import gaussian_kde
    pdf = gaussian_kde(dt)(x)

    plt.hist(dt, bins='auto', density=True, color='black',alpha=0.7)

    bin_edges=labels_to_edges(dt,max_ent_labels)

    for i in range(len(bin_edges) - 1):
        mask = (x >= bin_edges[i]) & (x <= bin_edges[i+1])
        plt.fill_between(x[mask], 0, 1, alpha=0.5)

    plt.plot(x, pdf, 'r-')
    plt.ylim(0, max(pdf) + np.std(pdf))
    plt.show()


def gmm_log_likelihood_vectorized(X, weights, means, covariances):
    """Vectorized calculation of GMM log-likelihood using Mahalanobis distance."""
    N, D = X.shape
    K = len(weights)
    
    log_probs = np.zeros((N, K))
    
    for k in range(K):
        diff = X - means[k]
        inv_cov = np.linalg.inv(covariances[k])
        log_det = np.linalg.slogdet(covariances[k])[1]
        
        # Mahalanobis distance
        mahal = np.sum(diff @ inv_cov * diff, axis=1)
        
        log_gauss = -0.5 * (
            D * np.log(2 * np.pi)
            + log_det
            + mahal
        )
        
        log_probs[:, k] = np.log(weights[k]) + log_gauss
    
    return np.sum(logsumexp(log_probs, axis=1))

## Max entropy binning function for comparison ##

def maxEntropyBin(x, nbins=3):
    u, cc = np.unique(x, return_counts=1)
    y = x.searchsorted(u, sorter=np.argsort(x))[1:]
    lab = np.zeros_like(x, dtype=int)
    xsize = x.size
    y = np.concatenate((y[1:], [x.size])) / x.size
    cp = np.cumsum(cc) / cc.sum()
    bins = np.empty(nbins - 1)
    ii = cp.searchsorted(1 / nbins, 'left')
    pii = cp[ii]
    bins[0] = u[ii]
    ibins = x <= bins[0]
    lab[ibins] = 0
    for i in range(1, nbins - 1):
        n = nbins - i
        ii = cp.searchsorted((1 + (n - 1) * pii) / n, 'left')
        pii = cp[ii]
        bins[i] = u[ii]
        ibins = (x > bins[i - 1]) & (x <= bins[i])
        lab[ibins] = i
    lab[x > bins[-1]] = nbins - 1
    return lab 
