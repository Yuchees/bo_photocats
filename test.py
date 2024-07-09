import numpy as np
import pandas as pd

from scipy.spatial import distance
import matplotlib.pyplot as plt
from src.BayesOpt import GaussianProcessRegressor, BayesOptimizer
from src.BayesOpt.util import kwargs_generator


def y_test(x, noise_sigma=1e-3):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()


# Test the Gaussian process
full_X = np.arange(0, 10, 0.1).reshape(-1, 1)
d = distance.cdist(full_X, full_X).reshape((1, 100, 100)) / 2
ds = np.vstack((d, d))
train_id = np.array([10, 30, 40, 50, 90])
test_id = np.array([i for i in range(100)])
train_y = y_test(x=full_X[train_id], noise_sigma=1e-3)

gpr = GaussianProcessRegressor(
    kernel='precomputed', kernel_matrix=ds,
    length_scale=np.array([1, 1]),
    length_scale_bounds=(1e-4, 1e4)
)
gpr.fit(train_id, train_y)
mu_, std_ = gpr.predict(test_id, return_std=True)
test_y = mu_.ravel()
plt.figure()
plt.title(
    "length_scale={} theta={:.2f}".format(gpr.length_scale, gpr.theta))
plt.fill_between(full_X.ravel(), test_y + 1.96 * std_, test_y - 1.96 * std_,
                 alpha=0.1)
plt.plot(full_X, test_y, label="predict")
plt.scatter(full_X[train_id], train_y, label="train", c="red", marker="x")
plt.legend()

# Test the Bayes loop
dist_matrix = np.load('./data/opt_conditions/dis_matrix.pkl',
                      allow_pickle=True)
df_exp = pd.read_excel('./data/opt_conditions/exp_results.xlsx',
                       sheet_name='step_0', index_col=0)
df_reactions = pd.read_pickle(
    './data/opt_conditions/df_reaction_conditions.pkl')
train_x_id = df_exp.index.values
train_y = df_exp['yield'].values.reshape(-1, 1)
test_id = np.array([i for i in range(4500)])
# Hyper parameters
gpr_kwargs = {
    'kernel': 'precomputed',
    'constant': 1,
    'constant_bounds': (1e-3, 1e3),
    'length_scale': [1, 1, 1, 1],
    'length_scale_bounds': (1e-4, 1e4)
}
bo_kwargs = {
    'bounds': np.array([0, 1]),
    'optimizer': 'sampling',
    'acq_func': 'UCB'
}
# Init the optimiser
opt = BayesOptimizer(
    base_estimator=GaussianProcessRegressor(
        kernel_matrix=dist_matrix, **gpr_kwargs
    ),
    sampling=test_id.reshape(-1, 1),
    **bo_kwargs
)
# Tell step 0 results and fitting GPs
opt.tell(train_x_id.reshape(-1, 1), train_y)
parallel_param = kwargs_generator(mean=3, size=8)
# parallel_param = {'kappa': np.random.random(8)/0.5 + 1}
print(parallel_param)
next_x = opt.parallel_ask(acq_func_args=parallel_param, num_samples=1)
print('')
