"""Solution."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import torch
import gpytorch

# TODO: how to interpret kernel information in task description?
# docker build --tag task3 .;docker run --rm -v "%cd%:/results" task3

# NOTE:
# - play with kernels and their hyperparameters
# - implement different acquisition functions
# https://gitlab.inf.ethz.ch/OU-KRAUSE/pai-demos/-/blob/master/demos/Bayesian%20Optimization%20and%20Active%20Learning.ipynb?ref_type=heads
# TODO
# - fix unnormalized fitting, number of iterations, learning rate -> for now switch to sklearn
# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA (K)

class ExactGP(gpytorch.models.ExactGP):
    def __init__(
            self, 
            train_x,
            train_y, 
            likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8)),
            kernel_type="linear_matern", 
            outputscale=0.5,
            lengthscale=10.0,
            prior_mean=0.0):
        """
        Initializes the ExactGP model with a specified kernel type.
        
        Args:
            train_x (Tensor): Training input data.
            train_y (Tensor): Training output data.
            likelihood (GaussianLikelihood): Likelihood function.
            mean (float): Mean value for the ConstantMean module.
            kernel_type (str): Type of kernel to use. Options are "linear_matern", "matern", "gaussian".
        """
        super().__init__(train_x, train_y, likelihood=likelihood)
        
        # Set up the mean module with a constant mean
        self.mean_module = gpytorch.means.ConstantMean()
        self.set_mean(prior_mean)

        # Select kernel type based on the input argument
        if kernel_type == "linear_matern":
            # Linear + Matern kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.LinearKernel() + gpytorch.kernels.MaternKernel(nu=2.5, lengthscale = lengthscale), 
                outputscale = outputscale
            )
        elif kernel_type == "matern":
            # Matern kernel only
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, lengthscale = lengthscale), 
                outputscale = outputscale
            )
        elif kernel_type == "gaussian":
            # Gaussian (RBF) kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(lengthscale = lengthscale), 
                outputscale = outputscale
            )
        else:
            raise ValueError("Invalid kernel type. Choose from 'linear_matern', 'matern', or 'gaussian'.")

        self.kernel_type = kernel_type

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale
    
    @property
    def length_scale(self):
        """Get length scale."""
        return self.covar_module.base_kernel.lengthscale
    
    @length_scale.setter
    def length_scale(self, value):
        self.covar_module.base_kernel.lengthscale = value 
    
    @output_scale.setter
    def output_scale(self, value):
        self.covar_module.outputscale = value 

    def set_mean(self, mean_value):
        """Set the mean value of the GP model."""
        self.mean_module.constant.data.fill_(mean_value)



# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
class BOAlgorithm():
    def __init__(self, 
                 af_type = "CEI",
                 f_kernel = "gaussian", 
                 v_kernel = "gaussian",
                 f_lengthscale=10.0,
                 v_lengthscale = 1.0,
                 f_outputscale=0.5,
                 v_outputscale=2**0.5, 
                 f_mean = 0, 
                 v_mean = SAFETY_THRESHOLD
                 ):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        # Initialize training data as empty tensors
        self.train_x = torch.tensor([], dtype=torch.float32)
        self.train_y_f = torch.tensor([], dtype=torch.float32)  # For objective f(x)
        self.train_y_v = torch.tensor([], dtype=torch.float32)  # For constraint v(x)


        # Initialize Gaussian likelihoods
        self.f_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.f_likelihood.noise = torch.tensor(0.15**2).detach()  # Set initial noise variance
        self.v_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
        # self.v_likelihood.noise = torch.tensor(0.0001**2).detach()

        # Initialize GP prior for bioavailability (logP)
        self.f = ExactGP(self.train_x, self.train_y_f, self.f_likelihood, f_kernel, f_lengthscale, f_outputscale, f_mean)
        # # Kernel Hypers
        # self.f.length_scale = f_lengthscale
        # if f_kernel == "gaussian": self.f.output_scale = f_outputscale
        # # Mean Hyper
        # self.f.set_mean(f_mean)

        # Initialize GP prior for synthetization constraint (SA)
        self.v = ExactGP(self.train_x, self.train_y_v, self.v_likelihood, v_kernel, v_lengthscale, v_outputscale, v_mean)
        # # Kernel Hypers
        # self.v.length_scale = v_lengthscale
        # if v_kernel == "gaussian": self.v.output_scale = v_outputscale
        # # Mean Hyper
        # self.v.set_mean(v_mean)

        # the next recommended point
        self.recommended_point = -1
        # set the aquistion function
        self.af_type = af_type
        # initialize a standard normal distribution
        self.standard_normal = torch.distributions.Normal(0, 1)
        # the minimum confidence level
        self.beta = 2.0             # regulates exploration -> can be made time dependent
        self.lambda_ = 0.1          # regulates the constraint violation
        self.gamma = 0.1            # regulates exploration for high variances in constraint
        self.pof_threshold = 0.99   # for Griffiths, 2020

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

        self.recommended_point = torch.tensor([[self.optimize_acquisition_function()]])

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
        x = np.atleast_2d(x)
        x_new = torch.tensor(x, dtype=torch.float32)

        if self.af_type == "UCB":
            return self.constrained_UCB(x_new)
        
        elif self.af_type == "CEI":
            return self.constrained_EI(x_new)

        else: 
            raise ValueError("Invalid acquisition function type. Choose from 'UCB', 'CEI', or 'RA_EI'.")
    
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
        f_mean, f_var = self.predict_f(x)
        f_std = torch.sqrt(f_var)
        v_mean, v_var = self.predict_v(x)

        # Compute the constrained UCB
        return f_mean + self.beta * f_std - self.lambda_ * torch.max(v_mean - SAFETY_THRESHOLD, torch.tensor(0.0)) + self.gamma * torch.sqrt(v_var)
    
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

        # Best observed objective value
        f_max = torch.max(self.train_y_f) 

        # Predict mean and variance for f(x)
        f_mean, f_var = self.predict_f(x)
        std = torch.sqrt(f_var)

        # Expected Improvement (EI) with variance
        z = (f_mean - f_max) / std
        EI = (f_mean - f_max) * self.standard_normal.cdf(z) + std * self.standard_normal.log_prob(z).exp()

        # Check if point satisfies the constraint
        v_mean, v_var = self.predict_v(x)
        PoF = torch.distributions.Normal(v_mean, v_var).cdf(torch.tensor(SAFETY_THRESHOLD)) # P(V(x) < K)
        
        # Return EI if within safety constraint, else return 0
        return torch.where(PoF > self.pof_threshold, EI * PoF, PoF)
    

        

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
        if isinstance(x, (np.ndarray, np.float64)):  # Check if x is a NumPy array or NumPy scalar
            x = torch.tensor(x, dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):  # Handle other unsupported types
            raise TypeError(f"Unsupported type for x: {type(x)}. Expected numpy.ndarray or torch.Tensor.")

        if isinstance(f, (np.ndarray, np.float64)):  # Check if f is a NumPy array or NumPy scalar
            f = torch.tensor(f, dtype=torch.float32)
        elif not isinstance(f, torch.Tensor):  # Handle other unsupported types
            raise TypeError(f"Unsupported type for f: {type(f)}. Expected numpy.ndarray or torch.Tensor.")

        if isinstance(v, (np.ndarray, np.float64)):  # Check if v is a NumPy array or NumPy scalar
            v = torch.tensor(v, dtype=torch.float32)
        elif not isinstance(v, torch.Tensor):  # Handle other unsupported types
            raise TypeError(f"Unsupported type for v: {type(v)}. Expected numpy.ndarray or torch.Tensor.")
        
        # Update tensors
        self.train_x = torch.cat([self.train_x, x.clone().detach().float().view(-1)])
        self.train_y_f = torch.cat([self.train_y_f, f.clone().detach().float().view(-1)])
        self.train_y_v = torch.cat([self.train_y_v, v.clone().detach().float().view(-1)])

        # Update GP models with new training data
        self.f.set_train_data(inputs=self.train_x, targets=self.train_y_f, strict=False)
        self.v.set_train_data(inputs=self.train_x, targets=self.train_y_v, strict=False)

        self.train_model()

    def train_model(self, num_iterations=50, learning_rate=0.1):
        """Trains the models with current training data."""
        self.f.train()
        self.v.train()

        # Optimizer and marginal log likelihood for both models
        optimizer = torch.optim.Adam([
            {'params': self.f.parameters()},
            {'params': self.v.parameters()}
        ], lr=learning_rate)

        mll_f = gpytorch.mlls.ExactMarginalLogLikelihood(self.f_likelihood, self.f)
        mll_v = gpytorch.mlls.ExactMarginalLogLikelihood(self.v_likelihood, self.v)

        # Training loop
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output_f = self.f(self.train_x)
            output_v = self.v(self.train_x)
            loss_f = -mll_f(output_f, self.train_y_f)
            loss_v = -mll_v(output_v, self.train_y_v)
            (loss_f + loss_v).backward()
            optimizer.step()

    def predict_f(self, test_x):
        """Make predictions on f(x) with new input data test_x."""
        self.f.eval()
        self.f_likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_f = self.f_likelihood(self.f(test_x))
        return pred_f.mean, pred_f.variance

    def predict_v(self, test_x):
        """Make predictions on v(x) with new input data test_x."""
        self.v.eval()
        self.v_likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_v = self.v_likelihood(self.v(test_x))
        return pred_v.mean, pred_v.variance


    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        xs = torch.linspace(DOMAIN[:, 0].item(), DOMAIN[:, 1].item(), 10000)
        f_mean, f_var = self.predict_f(xs)
        v_mean, v_var = self.predict_v(xs)

        # Find the optimal solution based on the mean of the objective function
        PoF = torch.distributions.Normal(v_mean, v_var).cdf(torch.tensor(SAFETY_THRESHOLD))

        idx_opt = torch.argmax(torch.where(PoF == 1, f_mean, torch.nan)).item()

        return xs[idx_opt].item()
    


    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        # Generate x values based on the domain and predict mean and std for f and v
        xs = torch.linspace(DOMAIN[:, 0].item(), DOMAIN[:, 1].item(), 1000)
        mean_f, std_f = self.predict_f(xs)
        mean_v, std_v = self.predict_v(xs)

        ########### Plot the model ###############
        # Create figure and axis objects
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        
        # Plot mean lines for f (logP) and v (SA)
        axs[0].plot(xs, mean_f, label="logP", color="blue")
        axs[0].scatter(self.train_x, self.train_y_f, color = "blue")
        axs[0].plot(xs, mean_v, label="SA", color="orange")
        axs[0].scatter(self.train_x, self.train_y_v, color = "orange")
        
        # Fill the uncertainty regions (mean ± std)
        axs[0].fill_between(xs, mean_f - std_f, mean_f + std_f, color="blue", alpha=0.2)
        axs[0].fill_between(xs, mean_v - std_v, mean_v + std_v, color="orange", alpha=0.2)

        # Add safe threshold line, if applicable
        axs[0].axhline(SAFETY_THRESHOLD, color="black", linestyle="--", label="Safe Threshold")
        axs[0].fill_between(xs, 0, max(mean_v)+1, where=(mean_v <= 4), color='green', alpha=0.2)

        # Optionally plot a recommended point
        if plot_recommendation:
            axs[0].plot(self.recommended_point, self.predict_f(self.recommended_point)[0], 'ro', label="Recommended Point")

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

    approach_string = "UCB"
    # FW-EI(x)=EI(x)+λ×P(v(x)<4)
    # RA-EI(x)=EI(x)−λE[max(v(x),0)]

    os.makedirs(f"3_BO_synthetization/debugging/{approach_string}", exist_ok=True)

    # Init problem
    agent = BOAlgorithm()

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

