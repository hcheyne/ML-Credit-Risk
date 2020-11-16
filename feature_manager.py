import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from itertools import permutations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class feature_manager(object):
    """
    This class manages the features needed for various tests based on passed parameters
    Input:
        method   : str
                defines the method by which the features will be extracted and restricted
                'intuitive', 'ohlson' - follow predefined models
                'all_ratios' - all pairs from data input in form a/(b+eps)
                'pca', 'lasso', 'pca_lasso' or 'lasso_pca' - start with all ratios then restrict based on one or a combination of feature selection methods
        tolerance: float
                lower bound of explained variance for PCA
        ratio_num_range : tuple (int, int)
                acceptable range of features after restricting
    Attributes:
        method : str
                defines the method by which the features will be extracted and restricted, from input
        TOL : float
                lower bound of explained variance for PCA, from input
        ratio_num_range : tuple (int, int)
                acceptable range of features after restricting, from input
        observations : dict
                Dictionary of the observation indices for each dataset trained/tested
        features : Index
                Names of the features used before any PCA
        flag : String
                Name of the default flag
        inf_na_replacement : List
                List of tuples (FeatureName, PositiveInf, NegativeInf, NaN) used to replace infinite/NA values
        trfm :  Dict
                Dictionary of data transforms used
        method_aux = List
                List of tuples (MethodName, VariableName, Variable) to store details about methods used
        epsilon = Float
                Value used as epsilon
        dropped_observations = List
                List to track any dropped observations
        Methods:
            manage_features : Takes a dataset and extracts, cleans, and restricts features based on parameters when class was initialized
            extract : Takes a dataset and calls the appropriate data extraction method based on self.method
            extract_all : Takes a dataset and creates all possible ratios of form a/(b+eps) where a and b are original features and adds them to the existing features and data
            extract_intuitive : Takes a dataset and creates features to match historic, proprietary model, raw data is removed. (removed to comply with privacy requirements)
            extract_ohlson : Takes a dataset and creates features to match literature model, raw data is removed. (feature names removed to comply with privacy requirements)
            clean_data : Takes a dataset and cleans the data by removing features and observations that have too many NAs, then replacing remaining NAs/infs with proxy values.
            find_infs_nas : Takes a dataset and creates proxy values for NAs/inf value replacement
            replace_infs_nas : Takes a dataset and replaces NAs/infs with proxy values
            restrict : Takes a dataset and calls the appropriate data restriction method(s) based on self.method
            pca_restrict : Takes a dataset and replaces features with features from Principle Component Analysis (PCA) with explained variance decided by parameters and curvature
            lasso_restrict : Takes a dataset and removes features based on Lasso selection
            lasso_smooth : Takes vectors x and y and smooths y wrt x using weighted nearest neighbours
            transform : Takes a dataset and transforms the features to match the current attributes of the instance of the class, usually for building a test set. Extracts, cleans, restricts.
            pca_transform : Takes a dataset and applies PCA as previously trained.
    """

    def __init__(self, method="intuitive", tolerance=0.75, ratio_num_range=(15, 50)):
        """
        Initializes feature_manager class

        Parameters:
            method (string): "intuitive", "ohlson", "all_ratios", "pca", "lasso","pca_lasso", "lasso_pca"
            tolerance (float): tolerance for minimum explained variance for PCA feature reduction
            ratio_num_range (tuple): acceptable range of features after restricting
        """
        self.method = method  ### the method name used: pca, pca_lasso, lasso, lasso_pca, ohlson, intuitive, raw, all_ratios, clustering(implement this)
        self.TOL = tolerance  ### Minimum variance for PCA
        self.ratio_num_range = ratio_num_range  ### (MIN, MAX) numbers of selected ratios
        if ratio_num_range[0] > ratio_num_range[1]:  # check min<=max
            self.ratio_num_range = (ratio_num_range[1], ratio_num_range[0])
        self.observations = {
            "train": None
        }  ### Dictionary of the observation indices for each dataset trained/tested
        self.features = None  ### names of the features used before any PCA
        self.flag = None  ### name of the default flag
        self.inf_na_replacement = []  ### list of tuples (RatioName, PositiveInf, NegativeInf, NaN)
        self.trfm = {
            "whiten": None,
            "pca": None,
            "lasso": None,
        }  ### Whitening & PCA & LASSO object used in transformation
        self.method_aux = []  ### list of tuples (MehtodName, VariableName, Variable)
        self.epsilon = 0.1  # float_info.epsilon
        self.dropped_observations = []  ### for interest's sake

    def manage_features(self, data):
        """
        Takes a dataset and extracts, cleans, and restricts features based on parameters when class was initialized

        Parameters:
            data (pandas.DataFrame): Input dataset to manage

        Returns:
            data (pandas.DataFrame): resulting dataset after feature management applied
        """
        row_ix = data.index
        col_ix = data.columns[1:]
        self.flag = data.columns[0]
        self.features = col_ix
        self.observations["train"] = row_ix
        data = self.extract(data)
        data = self.clean_data(data)
        data = self.restrict(data)
        return data

    def extract(self, data):
        """
        Takes a dataset and calls the appropriate data extraction method based on self.method

        Parameters:
            data (pandas.DataFrame): Input dataset to manage

        Returns:
            data (pandas.DataFrame): Resulting dataset after feature extraction
        """
        if self.method == "intuitive":
            data = self.extract_intuitive(data)
        elif self.method == "ohlson":
            data = self.extract_ohlson(data)
        else:
            data = self.extract_all(data)
        return data

    def extract_all(self, data):
        """
        Takes a dataset and creates all possible ratios of form a/(b+eps) where a and b are original features and adds them to the existing features and data

        Parameters:
            data (pandas.DataFrame): Input dataset to manage

        Returns:
            data (pandas.DataFrame): resulting dataset after feature extraction
        """
        ##### Note: After calculating all ratios, final data may contains NAs

        var_names = self.features
        for var1, var2 in permutations(var_names, 2):
            cache = data[var1] / (data[var2] + self.epsilon)
            ratio_name = var1 + "/" + var2
            data[ratio_name] = cache
        del cache
        self.features = data.columns[1:]
        return data

    def extract_intuitive(self, data):
        """
        Takes a dataset and creates features to match historic, proprietary model, raw data is removed. (removed to comply with privacy requirements)

        Parameters:
            data (pandas.DataFrame): Input dataset to manage

        Returns:
            data (pandas.DataFrame): resulting dataset after feature extraction
        """
        ##### Note: After calculating ratios, final data may contain +/- inf
        ### Delete variables that are not used to calculating ratios

        # var_names = data.columns
        # drop_var = var_names[~var_names.isin(keep)]
        # data.drop(drop_var, axis=1, inplace=True)

        ### Compute Accounting Ratios - removed for confidentiality
        """

        var_names = data.columns
        drop_var = var_names[~var_names.isin(keep)]
        data.drop(drop_var, axis=1, inplace=True)
        self.features = data.columns[1:]
        """
        return data

    def extract_ohlson(self, data):
        """
        Takes a dataset and creates features to match literature model, raw data is removed. (feature names removed to comply with privacy requirements)

        Parameters:
            data (pandas.DataFrame): Input dataset to manage

        Returns:
            data (pandas.DataFrame): resulting dataset after feature extraction
        """
        ##### Note: After calculating ratios, final data contains +/- inf
        ### Delete variables that are not used to calculating ratios

        # var_names = data.columns
        # drop_var = var_names[~var_names.isin(keep)]
        # data.drop(drop_var, axis=1, inplace=True)

        ### Compute Accounting Ratios - removed for confidentiality

        keep = [self.flag, "A", "B", "C", "D", "E", "F", "G", "H", "I"]  #'E','H'
        var_names = data.columns
        drop_var = var_names[~var_names.isin(keep)]
        data.drop(drop_var, axis=1, inplace=True)
        self.features = data.columns[1:]
        return data

    def clean_data(self, data):
        """
        Takes a dataset and cleans the data by removing features and observations that have too many NAs, then replacing remaining NAs/infs with proxy values.

        Parameters:
            data (pandas.DataFrame): Input dataset to manage

        Returns:
            data (pandas.DataFrame): resulting dataset after cleaning
        """
        # features that are more than 20% NA are dropped first
        ft_na_lim = np.floor(data.shape[0] * 0.2)
        na_ct = np.isnan(data).sum(axis=0)
        drop_cols = data.columns[(na_ct > ft_na_lim)]
        data.drop(drop_cols.values, axis=1, inplace=True)
        # data = data[data.columns[~data.columns.isin(drop_cols)]]
        self.features = data.columns[1:]

        # observations that are more than 20% NA are dropped next
        obs_na_lim = np.floor(data.shape[1] * 0.2)
        na_ct = np.isnan(data).sum(axis=1)
        drop_rows = data.index[(na_ct > obs_na_lim)]
        data.drop(drop_rows.values, axis=0, inplace=True)
        # data = data[~data.index.isin(drop_rows)]
        self.observations["train"] = data.index
        self.dropped_observations.append(drop_rows.values)

        # replace +/-inf with large numbers, and NA with mean
        self.find_infs_nas(data)
        data = self.replace_infs_nas(data)
        return data

    def find_infs_nas(self, data):
        """
        Takes a dataset and creates proxy values for NAs/inf value replacement

        Parameters:
            data (pandas.DataFrame): Input dataset to manage
        """
        for f in self.features:
            mask = np.isfinite(data[f])
            MAX, MIN = data[f][mask].max(), data[f][mask].min()
            PositiveInf = 3 * MAX * (MAX > 0) - 3 * MIN * (MAX <= 0)  # 3 * MAX
            NegativeInf = 3 * MIN * (MIN < 0) - 3 * MAX * (MIN >= 0)  # 3 * MIN
            NAmean = data[f][mask].mean()
            self.inf_na_replacement.append((f, PositiveInf, NegativeInf, NAmean))

    def replace_infs_nas(self, data):
        """
        Takes a dataset and replaces NAs/infs with proxy values

        Parameters:
            data (pandas.DataFrame): Input dataset

        Returns:
            data (pandas.DataFrame): resulting dataset after replacement
        """
        for inf in self.inf_na_replacement:
            if inf[0] in data.columns:
                data[inf[0]].replace(+np.inf, inf[1], inplace=True)
                data[inf[0]].replace(-np.inf, inf[2], inplace=True)
                data[inf[0]].replace(np.nan, inf[3], inplace=True)
        return data

    def restrict(self, data):
        """
        Takes a dataset and calls the appropriate data restriction method(s) based on self.method

        Parameters:
            data (pandas.DataFrame): Input dataset to manage

        Returns:
            data (pandas.DataFrame): Resulting dataset after feature restriction
        """
        ### Fit PCA and/or Lasso with data and transform data in-place

        if self.method not in ("ohlson", "intuitive", "all_ratios"):
            self.ratio_num_range = (
                max(self.ratio_num_range[0], 1),
                min(self.ratio_num_range[1], data.shape[1] - 2),
            )
            if self.method == "pca":
                data = self.pca_restrict(data, min_var=-1)  # No INF & NAN
            elif self.method == "lasso":
                data = self.lasso_restrict(data)  # No INF & NAN

            elif self.method == "lasso_pca":  ### Do Lasso first, then PCA
                data = self.lasso_restrict(
                    data, a=1e-1, rnr=(self.ratio_num_range[1], data.shape[1] - 2)
                )  # No INF & NAN
                data = self.pca_restrict(data, min_var=-1)

            elif self.method == "pca_lasso":  ### Do PCA first, then Lasso
                data = self.pca_restrict(
                    data, min_var=0.95, rnr=(self.ratio_num_range[1], data.shape[1] - 2)
                )  # No INF & NAN
                data = self.lasso_restrict(data, a=1e-1)
            else:
                print("Feature selection method name not known.\n")
        return data

    def pca_restrict(self, data, curvature=True, rnr=None, show_plots=False):
        """
        Takes a dataset and replaces features with features from Principle Component Analysis (PCA) with explained variance decided by parameters and curvature

        Parameters:
            data (pandas.DataFrame): Input dataset to manage
            curvature (bool): True to use curvature to decide number of included components, False to use least components to have self.TOL explained variance included
            rnr (tuple (int, int)): input value to change the allowed range of ratios remaining
            show_plots (bool): True to show plots, otherwise False

        Returns:
            data (pandas.DataFrame): Resulting dataset after feature restriction
        """
        if rnr is None:
            rnr = self.ratio_num_range
        ### standardization & PCA
        whitening = StandardScaler(with_mean=True, with_std=True)
        pca = PCA(copy=False).fit(whitening.fit_transform(data[data.columns[1:]]))

        ##################################
        ### calculate no. of components
        v = pca.explained_variance_ratio_
        c = v.cumsum()
        #  Method 1: Based on the maximum curvature of cumulative variance
        if curvature:
            ### fourth order approximation of 2nd derivative of cumulative explained variance (c) wrt number of included components
            d2c_h4 = (-c[:-4] + 16 * c[1:-3] - 30 * c[2:-2] + 16 * c[3:-1] - c[4:]) / 12
            ### fourth order approximation of 1st derivative of explained variance (v) wrt number of included components (note v = c', v' = c'')
            d1v_h4 = (v[:-4] - 8 * v[1:-3] + 8 * v[3:-1] - v[4:]) / 12
            # print((np.argmin(d2c_h4[3:])+7, np.argmin(d1v_h4[3:])+7, sum(c<self.TOL)+1, rnr[0]))
            n_components = max(
                np.argmin(d2c_h4[3:]) + 7,
                np.argmin(d1v_h4[3:]) + 7,
                sum(c < self.TOL) + 1,
                rnr[0],
            )

            if show_plots:
                plot_len = min(100, c.shape[0])
                fig = plt.figure(figsize=(16, 4))

                ax1 = fig.add_subplot(1, 2, 1)
                ax1.set_title("Explained varaince vs included components")
                ax1.plot(np.arange(1, plot_len + 1), c[:plot_len])
                ax1.plot(n_components, c[n_components - 1], ".")

                ax1 = fig.add_subplot(1, 2, 2)
                ax1.set_title(
                    "Second derivative of explained varaince wrt included components"
                )
                ax1.plot(np.arange(3, plot_len - 1), d2c_h4[: (plot_len - 4)])
                ax1.plot(np.arange(3, plot_len - 1), d1v_h4[: (plot_len - 4)])
                # outputBuilder.printPlot(plt)
        #  Method 2: Given minimum explained variance
        else:
            n_components = sum(c < self.TOL) + 1
        ##################################
        # ensure n_components is within required range
        if n_components < rnr[0]:
            print(
                "PCA chose too few components, used range minimum number of variables(%d) instead of %d"
                % (rnr[0], n_components)
            )
            n_components = rnr[1]
        if n_components > rnr[1]:
            print(
                "PCA chose too many components, used range maximum number of variables(%d) instead of %d"
                % (rnr[1], n_components)
            )
            n_components = rnr[1]

        self.method_aux.append(("pca", "explained_variance", c[n_components - 1]))

        ### Take first n_components only
        pca.components_ = pca.components_[:n_components]
        pca.n_components_ = n_components
        self.trfm["whiten"], self.trfm["pca"] = whitening, pca

        ### Keep PCA results only
        X_pca = pca.transform(whitening.transform(data[data.columns[1:]]))
        data = data.loc[
            :, self.flag
        ]  # data.drop(self.features.values, axis=1, inplace=True)  # No INF & NAN
        temp = pd.DataFrame(X_pca, index=data.index)
        data = pd.DataFrame(data).join(temp)
        del X_pca
        return data

    def lasso_restrict(self, data, a=0.01, rnr=None):
        """
        Takes a dataset and removes features based on Lasso selection

        Parameters:
            data (pandas.DataFrame): Input dataset to manage
            a (float): alpha for LASSO regression
            rnr (tuple (int, int)): input value to change the allowed range of ratios remaining

        Returns:
            data (pandas.DataFrame): Resulting dataset after feature restriction
        """
        if rnr is None:
            rnr = self.ratio_num_range
        ### NOTE: Input data can contain NaNs, output data contain no NaN
        # lasso = linear_model.LogisticRegression(C=c0, penalty='l1', tol=0.01, random_state=56, n_jobs=2)
        smoothed = pd.DataFrame(
            [self.lasso_smooth(data[i], data[self.flag]) for i in data.columns[1:]]
        ).T

        lasso = linear_model.Lasso(alpha=a)
        lasso.fit(smoothed, data[self.flag])

        ### Adjust the numbers of selected ratios
        n_ratios = sum(lasso.coef_ != 0)
        if (n_ratios > rnr[1]) | (n_ratios < rnr[0]):
            times = 2
            a = a * (1 / times, times)[n_ratios > rnr[1]]
            lasso = linear_model.Lasso(alpha=a)
            lasso.fit(smoothed, data[self.flag])
            n_ratios_new = sum(lasso.coef_ != 0)
            stop_lasso = (n_ratios_new <= rnr[1]) & (n_ratios_new >= rnr[0])

            while not stop_lasso:
                if (n_ratios_new > rnr[1]) & (n_ratios > rnr[1]):
                    a = a * times
                elif (n_ratios_new < rnr[0]) & (n_ratios < rnr[0]):
                    a = a / times
                else:
                    a = a * np.sqrt((times, 1 / times)[n_ratios > rnr[1]])
                lasso = linear_model.Lasso(alpha=a)
                lasso.fit(smoothed, data[self.flag])
                n_ratios_new, n_ratios = sum(lasso.coef_ != 0), n_ratios_new
                stop_lasso = (n_ratios_new <= rnr[1]) & (n_ratios_new >= rnr[0])
        # print('# of variables: %d, cost parameter: %e' % (sum(lasso.coef_!=0), lasso.alpha))

        ### only keep selected ratios / components
        drop_var = data.columns[1:][lasso.coef_ == 0]
        data.drop(drop_var.values, axis=1, inplace=True)
        self.trfm["lasso"] = lasso
        # data.dropna(axis=0, how='any', inplace=True)   # No INF & NAN
        if self.method == "pca_lasso":
            keep_var = lasso.coef_ != 0
            self.trfm["pca"].components_ = self.trfm["pca"].components_[keep_var, :]
            self.trfm["pca"].n_components_ = len(self.trfm["pca"].components_)
        else:
            self.features = data.columns[1:]
            # self.inf_replacement = [infs for infs in self.inf_replacement if infs[0] not in drop_var]
        return data

    def lasso_smooth(self, x, y, MX=25):
        """
        Takes vectors x and y and smooths y wrt x using weighted nearest neighbours

        Parameters:
            x (list): independent variable
            y (list): dependent variable
            MX (int): maximum neighbours to have non-zero weight

        Returns:
            dat['s'] (list): smoothed values
        """
        # kernel smoother
        dat = pd.DataFrame([x, y]).T
        dat.columns = ["x", "y"]
        dat.sort_values(by="x", inplace=True)
        sm = np.diag((3 / 4) * np.ones(len(x)))
        r = 1
        while r < MX and r < len(x) / 5:
            sm = sm + np.diag((1 / 2) ** r * np.ones(len(x) - r), r)
            sm = sm + np.diag((1 / 2) ** r * np.ones(len(x) - r), -r)
            r = r + 1
        sm = sm / sm.sum(axis=1)[:, None]

        dat["s"] = np.dot(sm, dat["y"])
        dat.reset_index(inplace=True)
        dat.sort_values(by="index", inplace=True)
        dat.reset_index(inplace=True)
        return dat["s"]

    def transform(self, data):
        """
        Takes a dataset and transforms the features to match the current attributes of the instance of the class, usually for building a test set. Extracts, cleans, restricts.

        Parameters:
            data (pandas.DataFrame): Input dataset to transform

        Returns:
            data (pandas.DataFrame): Transformed dataset
        """
        ### Note: output data contain no NaN
        for f in self.features:
            if "/" in f:
                a, b = f.split("/")
                data[f] = data[a] / (data[b] + self.epsilon)

        var_names = data.columns
        drop_var = var_names[~var_names.isin(self.features)]
        # drop_var = drop_var[drop_var!=self.flag]
        data.drop(drop_var, axis=1, inplace=True)

        # observations that are more than 20% NA are dropped next
        obs_na_lim = np.floor(data.shape[1] * 0.2)
        na_ct = np.isnan(data).sum(axis=1)
        drop_rows = data.index[(na_ct > obs_na_lim)]
        data.drop(drop_rows.values, axis=0, inplace=True)
        self.dropped_observations.append(drop_rows.values)
        self.observations["test" + str(len(self.observations))] = data.index
        self.replace_infs_nas(data)  # No INF

        if self.method in ("pca", "lasso_pca", "pca_lasso"):  # self.method=='pca':    #
            data = self.pca_transform(data)
        return data

    def pca_transform(self, data):
        """
        Takes a dataset and applies PCA as previously trained.

        Parameters:
            data (pandas.DataFrame): Input dataset to transform

        Returns:
            data (pandas.DataFrame): Transformed dataset
        """
        X_pca = self.trfm["pca"].transform(self.trfm["whiten"].transform(data))
        data.drop(self.features.values, axis=1, inplace=True)
        data[list(range(X_pca.shape[1]))] = pd.DataFrame(X_pca, index=data.index)
        del X_pca
        return data
