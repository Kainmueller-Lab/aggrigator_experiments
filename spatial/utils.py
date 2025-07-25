from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd
import anndata
from sklearn.metrics import roc_auc_score


##########################
#### DATA PREPARATION ####
##########################
def setup_anndata_object(fingerprints):
    """
    Setup an AnnData object from the fingerprints DataFrame.
    """
    # put all id_fingerprints in anndata object
    adata = anndata.AnnData(fingerprints)
    adata.obs['sample_type'] = ['ood' if idx.startswith('ood_') else 'id' for idx in adata.obs_names]
    adata.obsm['X_transformed'] = adata.X.copy()
    return adata


def create_fitting_split(adata, seed=42, train_fraction=0.5):
    """
    Create a fitting split for the AnnData object for the id set
    """
    np.random.seed(seed)
    ids = adata.obs_names[adata.obs['sample_type'] == 'id']
    ids = np.array(ids)  # Convert to numpy array for shuffling
    np.random.shuffle(ids)
    n_total = len(ids)
    n_train = int(train_fraction * n_total)
    train_ids = ids[:n_train]
    adata.obs['fitted'] = False
    adata.obs.loc[train_ids, 'fitted'] = True
    adata.obs['fitted'] = adata.obs['fitted'].astype(bool)
    return adata


def apply_transformation(adata, transformation='identity', fitted_only=True):
    """
    Apply a transformation to the data in the AnnData object, by fitting a Normalizer
    on only points marked as 'fitted' and applies on all points
    Args:
        adata (AnnData): The AnnData object containing the data.
        transformation (str): The type of transformation to apply ('identity', 'logit', 'arcsin-sqrt', 'quantile').
        fitted_only (bool): If True, only apply the transformation on points marked as 'fitted'.
    Returns:
        adata (AnnData): The AnnData object with 
            a) the transformed data in adata.obsm['X_transformed'].
            b) a dictionary of Normalizer objects in adata.uns['normalizer'].
    """
    normalizer_dict = {}

    if fitted_only:
        data = adata[adata.obs['fitted']].X
    else:
        data = adata.X

    for var in adata.var_names:
        d = data[:, adata.var_names == var].flatten()
        normalizer = Normalizer(transformation=transformation)
        normalizer.fit(d)
        normalizer_dict[var] = normalizer
    
    # apply the transformation to all data
    transformed_data = np.array(
        [normalizer_dict[var].apply(adata.X[:, adata.var_names == var].flatten())
        for var in adata.var_names]).T
    adata.obsm['X_transformed'] = transformed_data
    adata.uns['normalizer'] = normalizer_dict
    return adata


class Normalizer:
    '''
    normalizes a flat array of data, use: 
    1) init with your favorite transformation function: n = Normalizer(transformation='logit')
    2) fit with your data: n.fit(data)
    3) apply to new data: n.apply(new_data) 
    '''
    def __init__(self, mu=0, sigma=1, transformation='logit', normalize=False):
        """
        Initializes the Normalizer with mean and standard deviation.
        Args:
            mu (float): Mean of the fitted data
            sigma (float): Standard deviation of the fitted data
            transformer (str): Transformation function to use, e.g., 'probit'
        """
        self.transformation = transformation
        self.mu = mu
        self.sigma = sigma
        self.eps = 1e-6
        self.normalize = normalize
        self.fitted = False

    def _data_transform(self, X):
        """
        transforms the data from [0, 1] to [-inf, inf] using selected function
        """
        if self.transformation == 'logit':
            X = np.clip(X, a_min=self.eps, a_max=1 - self.eps)
            return np.log(X / (1 - X))
        if self.transformation == 'probit':
            X = np.clip(X, a_min=self.eps, a_max=1 - self.eps)
            return norm.ppf(X)
        if self.transformation == 'arcsin-sqrt':
            X = np.clip(X, a_min=self.eps, a_max=1 - self.eps)
            return np.arcsin(np.sqrt(X))
        if self.transformation == 'quantile':
            X = np.clip(X, a_min=self.eps, a_max=1 - self.eps).reshape(-1, 1)
            if self.fitted:
                return self.qt.transform(X)
            else:
                return X
        else:
            return X

    def fit(self, X):
        X = self._data_transform(X)
        if self.transformation == 'quantile':
            self.qt = QuantileTransformer(output_distribution="normal")
            self.qt.fit(X)
        else:
            if self.normalize:
                self.mu = X.mean()
                self.sigma = X.std()
        self.fitted = True

    def apply(self, X):
        X = self._data_transform(X)
        return (X - self.mu) / (self.sigma + self.eps)
    

############################
#### EVALUATION METRICS ####
############################
def compute_auroc(adata, score_name='gmm_score'):
    """Compute AUROC for non-fitted points in the adata object."""
    non_fitted_ids = adata.obs_names[~adata.obs['fitted'].astype(bool)].tolist()
    non_fitted_scores = adata.obs[score_name][adata.obs_names.isin(non_fitted_ids)]
    non_fitted_labels = adata.obs['sample_type'][adata.obs_names.isin(non_fitted_ids)]
    auroc = roc_auc_score(non_fitted_labels == 'ood', non_fitted_scores)
    return auroc