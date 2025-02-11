"""Solution."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, DotProduct, WhiteKernel
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm

# TODO: how to interpret kernel information in task description?
# docker build --tag task3 .;docker run --rm -v "%cd%:/results" task3

# NOTE:
# - play with kernels and their hyperparameters
# - implement different acquisition functions
# https://gitlab.inf.ethz.ch/OU-KRAUSE/pai-demos/-/blob/master/demos/Bayesian%20Optimization%20and%20Active%20Learning.ipynb?ref_type=heads

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA (K)

# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.


class GaussianProcessRegressorWithMeanPrior(GaussianProcessRegressor):
    def __init__(self, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0, normalize_y=False, copy_X_train=True,
                 mean_prior=0):
        super().__init__(kernel=kernel, alpha=alpha, optimizer=optimizer,
                         n_restarts_optimizer=n_restarts_optimizer,
                         normalize_y=normalize_y, copy_X_train=copy_X_train)
        self.mean_prior = mean_prior
        self.y_std = None

    def fit(self, X, y):
        """
        Fit the Gaussian Process with the target data shifted by the mean prior.
        If the standard deviation of y is zero, skip normalization.
        """
        self.y_std = y.std()
        if self.y_std == 0:
            y_transformed = y - self.mean_prior
        else:
            y_transformed = (y - self.mean_prior) / self.y_std

        return super().fit(X, y_transformed)

    def predict(self, X, return_std=False, return_cov=False):
        """
        Predict using the Gaussian Process, adding back the mean prior and
        handling cases where normalization was skipped.
        """
        if return_cov:
            y_mean, y_cov = super().predict(X, return_cov=True)
            if self.y_std == 0:
                y_mean += self.mean_prior
            else:
                y_mean = y_mean * self.y_std + self.mean_prior
                y_cov = y_cov * self.y_std**2
            return y_mean, y_cov

        y_mean, y_std = super().predict(X, return_std=True)
        if self.y_std == 0:
            y_mean += self.mean_prior
        else:
            y_mean = y_mean * self.y_std + self.mean_prior
            y_std = y_std * self.y_std
        return (y_mean, y_std) if return_std else y_mean
 

class BOAlgorithm():
    def __init__(self, 
                 af_type = "PoF",
                 f_kernel_type = "gaussian", 
                 v_kernel_type = "matern",
                 f_lengthscale=.5,
                 v_lengthscale = 1.,
                 f_outputscale=0.5,
                 v_outputscale=2**0.5, 
                 f_mean_prior = 0, # TODO put mean on f?
                 v_mean_prior = 0, # SAFETY_THRESHOLD, 
                 f_observation_noise = 0.15,
                 v_observation_noise = 1e-04,
                 ):
        """Initializes the algorithm with a parameter configuration."""
        if f_kernel_type == "gaussian":
            self.kernel_f = ConstantKernel(f_outputscale, 
                                           constant_value_bounds=(1e-3, 1e3)) * \
                            RBF(length_scale=f_lengthscale,
                                length_scale_bounds=(0.5, 10))
        elif f_kernel_type == "matern":
            self.kernel_f = ConstantKernel(f_outputscale, 
                                           constant_value_bounds=(1e-3, 1e3)) * \
                            Matern(nu=2.5, 
                                length_scale=f_lengthscale,
                                length_scale_bounds=(0.5, 10))
        else:
            raise ValueError("Invalid kernel type. Choose from 'gaussian' or 'matern'.")

        if v_kernel_type == "gaussian":
            self.kernel_v = (
                            DotProduct(sigma_0=1e-03, sigma_0_bounds=(1e-3, 1e3)) + \
                            ConstantKernel(v_outputscale, 
                                           constant_value_bounds=(1e-3, 1e3)) * \
                            RBF(length_scale=v_lengthscale,
                                length_scale_bounds=(0.5, 10))
                            )
        elif v_kernel_type == "matern":
            self.kernel_v = (
                            DotProduct(sigma_0=1e-03, sigma_0_bounds=(1e-3, 1e3)) + \
                            # ConstantKernel(v_outputscale, 
                            #                constant_value_bounds=(1e-3, 1e3)) * \
                            Matern(nu=2.5, 
                                    length_scale=v_lengthscale,
                                    length_scale_bounds=(0.5, 10))
                            )
        else:
            raise ValueError("Invalid kernel type. Choose from 'gaussian' or 'matern'.")

        # self.kernel_f = ConstantKernel(0.5) * RBF(length_scale=0.5, length_scale_bounds=(0.5, 10))
        # self.kernel_v = DotProduct(sigma_0=0) + Matern(nu=2.5, length_scale=1, length_scale_bounds=(0.5, 10))

        
        
        self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f, 
                                             alpha=f_observation_noise**2,
                                             optimizer=None, # keep the hyperparameters fixed
                                             normalize_y=True)
        # self.gp_v = GaussianProcessRegressorWithMeanPrior(mean_prior=v_mean_prior, # we're setting a mean so we need to handle normalization ourselves
        #                                                 kernel=self.kernel_v, 
        #                                                 alpha=v_observation_noise**2,
        #                                                 optimizer=None, # keep the hyperparameters fixed 
        #                                                 ) 

        self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v, 
                                            alpha=v_observation_noise**2,
                                            optimizer=None, # keep the hyperparameters fixed
                                            normalize_y=True)
        
        self.data_points = []
        self.af_type = af_type
        self.beta = 2.0
        self.pof_threshold = 0.95

    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        self.recommended_point = np.array([self.optimize_acquisition_function()]).reshape(-1, 1)

        return self.recommended_point

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    
    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        # Implement the acquisition function you want to optimize.
        x = np.atleast_2d(x).reshape(-1, 1)

        if self.af_type == "UCB":
            return self.constrained_UCB(x)
        elif self.af_type == "CEI":
            return self.constrained_EI(x)
        elif self.af_type == "PoF":
            return self.safe_PoF(x)
        else:
            raise ValueError("Invalid acquisition function type. Choose 'CEI' or 'UCB'.")
    
    def constrained_UCB(self, x: np.ndarray):
        """Compute the Upper Confidence Bound (UCB) acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        # Predict mean and variance for f(x)
        mean_f, std_f = self.gp_f.predict(x, return_std=True)
        mean_v, std_v = self.gp_v.predict(x, return_std=True)

        return mean_f + self.beta * std_f - np.maximum(mean_v - SAFETY_THRESHOLD, 0)
    
    def safe_PoF(self, x: np.ndarray):
        """Compute the Probability of Failure (PoF) acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        mean_v, std_v = self.gp_v.predict(x, return_std=True)

        return norm.cdf(SAFETY_THRESHOLD, mean_v, std_v)
    
    def constrained_EI(self, x: np.ndarray):
        """Compute the Constrained Expected Improvement (CEI) acquisition function for x. Griffiths, 2020

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """

        mean_f, std_f = self.gp_f.predict(x, return_std=True)
        mean_v, std_v = self.gp_v.predict(x, return_std=True)
        
        # Expected Improvement (EI)
        f_best = max(d["f"] for d in self.data_points)
        z = (mean_f - f_best) / (std_f + 1e-9)
        EI = (mean_f - f_best) * norm.cdf(z) + std_f * norm.pdf(z)

        # P(v(x) < K)
        PoF = norm.cdf(SAFETY_THRESHOLD, mean_v, std_v) 
        
        # Return EI if within safety constraint, else return 0
        return np.where(PoF > self.pof_threshold, EI * PoF, PoF)
    

        

    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        x, f, v = float(x), float(f), float(v)
        # append to data points list
        self.data_points.append({"x": x, "f": f, "v": v})
        # extract current data
        X = np.array([d["x"] for d in self.data_points]).reshape(-1, 1)
        y_f = np.array([d["f"] for d in self.data_points]).reshape(-1, 1)
        y_v = np.array([d["v"] for d in self.data_points]).reshape(-1, 1)

        # update the GPs
        self.gp_f.fit(X, y_f)
        self.gp_v.fit(X, y_v)



    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # xs = np.linspace(DOMAIN[:, 0].item(), DOMAIN[:, 1].item(), 10000).reshape(-1, 1)
        # # Predict mean and variance for both objective and constraint
        # mean_f, _ = self.gp_f.predict(xs, return_std=True)
        # mean_v, std_v = self.gp_v.predict(xs, return_std=True)

        # # Find the optimal solution based on the mean of the objective function
        # PoF = norm.cdf(SAFETY_THRESHOLD, mean_v, std_v) 

        # idx_opt = np.argmax(mean_f[PoF == 1])

        # return xs[idx_opt].item()

        X = np.array([d["x"] for d in self.data_points]).reshape(-1, 1)
        y_f = np.array([d["f"] for d in self.data_points]).reshape(-1, 1)
        y_v = np.array([d["v"] for d in self.data_points]).reshape(-1, 1)

        return X[np.argmax(y_f)].item()
    


    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        # Generate x values based on the domain and predict mean and std for f and v
        xs = np.linspace(DOMAIN[:, 0].item(), DOMAIN[:, 1].item(), 1000).reshape(-1, 1)
        mean_f, std_f = self.gp_f.predict(xs, return_std=True)
        mean_v, std_v = self.gp_v.predict(xs, return_std=True)

        X = np.array([d["x"] for d in self.data_points]).reshape(-1, 1)
        y_f = np.array([d["f"] for d in self.data_points]).reshape(-1, 1)
        y_v = np.array([d["v"] for d in self.data_points]).reshape(-1, 1)


        ########### Plot the model ###############
        # Create figure and axis objects
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        
        # Plot mean lines for f (logP) and v (SA)
        axs[0].plot(xs, mean_f, label="logP", color="blue")
        axs[0].scatter(X, y_f, color = "blue")
        axs[0].plot(xs, mean_v, label="SA", color="orange")
        axs[0].scatter(X, y_v, color = "orange")
        
        # Fill the uncertainty regions (mean ± std)
        axs[0].fill_between(xs.reshape(-1), mean_f - std_f, mean_f + std_f, color="blue", alpha=0.2)
        axs[0].fill_between(xs.reshape(-1), mean_v - std_v, mean_v + std_v, color="orange", alpha=0.2)

        # Add safe threshold line, if applicable
        axs[0].axhline(SAFETY_THRESHOLD, color="black", linestyle="--", label="Safe Threshold")
        axs[0].fill_between(xs.reshape(-1), 0, max(mean_v)+1, where=(mean_v <= 4), color='green', alpha=0.2)

        # Optionally plot a recommended point
        if plot_recommendation:
            axs[0].plot(self.recommended_point, self.gp_f.predict(self.recommended_point)[0], 'ro', label="Recommended Point")
            axs[1].plot(self.recommended_point, self.acquisition_function(self.recommended_point)[0], 'ro', label="Recommended Point")

        # Add labels and legend
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("Value")
        axs[0].set_title("Posterior")
        axs[0].legend(loc="upper right")


        ############ plot the aquisition function ################
        aq_fn = self.acquisition_function(xs[:, None])
        axs[1].plot(xs, aq_fn, label="Acquisition Function", color="purple")
        axs[1].set_xlabel("x")
        axs[1].set_title("Acquisition Function")



        # Return the figure and axis objects to allow further customization outside the function
        return fig, axs

        


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


# def f(x: float):
#     """Dummy logP objective"""
#     mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
#     return - np.linalg.norm(x - mid_point, 2, axis = -1)

# def f(x: np.ndarray):
#     """Dummy logP objective"""
#     # Calculate the midpoint (assuming DOMAIN has two columns)
#     mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    
#     # Ensure mid_point has the same shape as x for broadcasting
#     # Expand mid_point to have shape (1, n) if necessary
#     mid_point = mid_point.reshape(1, -1)
    
#     # Calculate the Euclidean distance element-wise
#     return -np.sqrt(np.sum((x - mid_point) ** 2, axis=1, keepdims=True))



# def v(x: float):
#     """Dummy SA"""
#     return (0.3 * x - 1.5)**2 + 3


# def f(x):
#     return np.sin(0.5 * x) + 0.5 * np.cos(2 * x) + 1.5  # Objective function (logP), shifted above zero

# def v(x):
#     return 4 + np.sin(1.5 * x) + 2 * np.cos(0.5 * x)  # Constraint function (SA), shifted above zero

def f(x):
    return -(0.2*x)**2 + 1  # Objective function (logP), shifted above zero

def v(x):
    return x - 1  # Constraint function (SA), shifted above zero


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    import os
    # True functions
    xs_ = np.linspace(0, 10, 500)
    y_f = f(xs_)
    y_v = v(xs_)

    

    # Init problem
    agent = BOAlgorithm()

    approach_string = agent.af_type
    # FW-EI(x)=EI(x)+λ×P(v(x)<4)
    # RA-EI(x)=EI(x)−λE[max(v(x),0)]

    os.makedirs(f"3_BO_synthetization/debugging/{approach_string}", exist_ok=True)

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_observation(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.recommend_next()

        fig, axs = agent.plot(plot_recommendation=True)
        axs[2].plot(xs_, y_f, label='objective (logP)', color='blue')
        axs[2].plot(xs_, y_v, label='constraint (SA)', color='orange')
        axs[2].fill_between(xs_, 0, max(y_v), where=(y_v <= 4), color='green', alpha=0.2)
        axs[2].axhline(SAFETY_THRESHOLD, color='black', linestyle='--', label='Safe Threshold')
        axs[2].scatter(x_init, f(x_init), color='black', marker='^', label='Initial point')
        axs[2].scatter(x, f(x), color='red', marker='o', label='Recommended point')
        axs[2].set_xlabel('x')
        axs[2].set_title('True Objective and Constraint Functions')
        # ax.set_title(f"Posterior of iteration {j}")
        fig.savefig(f"3_BO_synthetization/debugging/{approach_string}/Posterior_{j}.png")

        # IF recommended point doesn't satisfy the constraint, 
        # it's an unsafe evaluation

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function recommend_next must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution() # maximizer of the acquisition function
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    max_f = max(y_f)
    # Compute regret
    regret = (max_f - f(solution))

    print(f'Optimal value: {max_f}\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()

