
from utilities import *
from scipy.linalg import solve_triangular
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator


class BayesianLogisticRegression(LinearClassifierMixin, BaseEstimator):
    def __init__(self, epochs):
        self.itr = epochs

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        X = self.add_bias(X)

        self.coef_ = [0] * 10
        self.sigma_ = [0] * 10
        self.intercept_ = [0] * 10

        for i in range(10):
            curr_class = self.classes_[i]
            mask = (y == curr_class)
            y_binary = np.ones(y.shape, dtype=np.float64)
            y_binary[~mask] = 0
            coef_eb, sigma_eb = self.vb_fit(X, y_binary)
            self.intercept_[i], self.coef_[i] = self._get_intercept(coef_eb)

        self.coef_ = np.asarray(self.coef_)
        return self







class VBLogisticRegression(BayesianLogisticRegression):
    '''
    Variational Bayesian Logistic Regression with local variational approximation.


    Parameters:
    -----------
    n_iter: int, optional (DEFAULT = 50 )
       Maximum number of iterations

    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold, if cange in coefficients is less than threshold
       algorithm is terminated

    fit_intercept: bool, optinal ( DEFAULT = True )
       If True uses bias term in model fitting

    a: float, optional (DEFAULT = 1e-6)
       Rate parameter for Gamma prior on precision parameter of coefficients

    b: float, optional (DEFAULT = 1e-6)
       Shape parameter for Gamma prior on precision parameter of coefficients

    verbose: bool, optional (DEFAULT = False)
       Verbose mode


    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients

    intercept_: array, shape = (n_features)
        intercepts


    References:
    -----------
   [1] Bishop 2006, Pattern Recognition and Machine Learning ( Chapter 10 )
   [2] Murphy 2012, Machine Learning A Probabilistic Perspective ( Chapter 21 )
    '''

    def __init__(self, n_iter=50, a=1e-4, b=1e-4 ):
        super(VBLogisticRegression, self).__init__(n_iter)
        self.a = a
        self.b = b
        self._mask_val = 0.

    def vb_fit(self, X, y):
        eps = 1
        n_samples, n_features = X.shape
        XY = np.dot(X.T, (y - 0.5))
        w0 = np.zeros(n_features)

        # hyperparameters of q(alpha) (approximate distribution of precision parameter of weights)
        a = self.a + 0.5 * n_features
        b = self.b

        for i in range(50):
            # In the E-step we update approximation of
            # posterior distribution q(w,alpha) = q(w)*q(alpha)

            # --------- update q(w) ------------------
            l = lam(eps)

            w, Ri = self.posterior_distribution(X, l, a, b, XY)
            # -------- update q(alpha) ---------------
            b = self.b + 0.5 * (np.sum(w[1:] ** 2) + np.sum(Ri[1:, :] ** 2))


            # -------- update eps  ------------
            # In the M-step update parameter eps which controls accuracy of local variational approximation to lower bound
            XMX = np.dot(X, w) ** 2
            XSX = np.sum(np.dot(X, Ri.T) ** 2, axis=1)
            eps = np.sqrt(XMX + XSX)


            # convergence
            if np.sum(abs(w - w0) > 1e-3) == 0 : break
            w0 = w

        l = lam(eps)
        coef_, sigma_ = self.posterior_distribution(X, l, a, b, XY, True)
        return coef_, sigma_


    def  posterior_distribution(self, X, l, a, b, XY, full_covar=False):
        ''' Finds gaussian approximation to posterior of coefficients using local variational approximation'''
        sigma_inv = 2 * np.dot(X.T * l, X)
        alpha_vec = np.ones(X.shape[1]) * float(a) / b

        alpha_vec[0] = np.finfo(np.float16).eps
        np.fill_diagonal(sigma_inv, np.diag(sigma_inv) + alpha_vec)
        R = np.linalg.cholesky(sigma_inv)
        Z = solve_triangular(R, XY, lower=True)

        ''' is there any specific function in scipy that efficently inverts low triangular matrix XD '''
        mean = solve_triangular(R.T, Z, lower=False)
        Ri = solve_triangular(R, np.eye(X.shape[1]), lower=True)

        # if full_covar:
        #     sigma = np.dot(Ri.T, Ri)
        #     return mean, sigma
        return mean, Ri
    def add_bias(self, X):
        '''Adds intercept to data matrix'''
        return np.hstack((np.ones([X.shape[0], 1]), X))
    def _get_intercept(self, coef):
        return coef[0], coef[1:]
    def get_variance(self, X):
        return np.asarray([np.sum(np.dot(X, s) * X, axis=1) for s in self.sigma_])
    def try_predict(self, X):

        prediction = self.predict_proba(X);
        ret = np.argmax(prediction, 1)
        return ret
    def predict_proba(self, X):
        scores = self.decision_function(X)

        X = self.add_bias(X)

        # probit approximation to predictive distribution
        sigma = self.get_variance(X)

        ks = 1.0 / (1.0 + np.pi * sigma / 8) ** 0.5
        probs = sigmoid(scores.T * ks).T

        probs /= np.reshape(np.sum(probs, axis=1), (probs.shape[0], 1))
        return probs

def lam(eps):
    ''' Calculates lambda eps (used for Jaakola & Jordan local bound) '''
    eps = -abs(eps)
    return 0.25 * exprel(eps) / (np.exp(eps) + 1)
def exprel(eps):
    return (np.exp(eps)-1)/eps