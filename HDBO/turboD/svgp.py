import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_svgp(inducing_points, train_x, train_y):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(inducing_points, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    iters = 500 + 1
    for i in range(iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print("Iteration: ", i, "\t Loss:", loss.item())
    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model


def testSVGP():
    from matplotlib import pyplot as plt
    inducing_points = torch.randn(25, 1) + 2.
    train_x = torch.linspace(0, 3, 10).view(-1, 1).contiguous()
    train_y = torch.sin(6. * train_x) + 0.3 * torch.randn_like(train_x)

    # >>> Plot the training points
    plt.scatter(train_x, train_y, marker = "*", color = "black")
    plt.xlabel("x")
    plt.ylabel("y")
    # <<< Plot the training points

    model = train_svgp(inducing_points, train_x, train_y.squeeze())
    # from gp import train_gp
    # model = train_gp(train_x, train_y.squeeze(), True, 50)
    test_x = torch.linspace(0, 8, 250).view(-1,1)
    with torch.no_grad():
        posterior = model.likelihood(model(test_x))

    # >>> Plot the posterior
    plt.plot(test_x, posterior.mean, color = "blue", label = "Post Mean")
    plt.fill_between(test_x.squeeze(), *posterior.confidence_region(), color = "blue", alpha = 0.3, label = "Post Conf Region")
    plt.scatter(train_x, train_y, color = "black", marker = "*", alpha = 0.5, label = "Training Data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    # <<< Plot the posterior

    val_x = torch.linspace(3, 5, 25).view(-1,1)
    val_y = torch.sin(6. * val_x) + 0.3 * torch.randn_like(val_x)

    cond_model = model.variational_strategy.get_fantasy_model(inputs=val_x, targets=val_y.squeeze())

    with torch.no_grad():
        updated_posterior = cond_model.likelihood(cond_model(test_x))

    # >>> Plot the updated posterior
    plt.plot(test_x, posterior.mean, color = "blue", label = "Post Mean")
    plt.fill_between(test_x.squeeze(), *posterior.confidence_region(), color = "blue", alpha = 0.3, label = "Post Conf Region")
    plt.scatter(train_x, train_y, color = "black", marker = "*", alpha = 0.5, label = "Training Data")

    plt.plot(test_x, updated_posterior.mean, color = "orange", label = "Fant Mean")
    plt.fill_between(test_x.squeeze(), *updated_posterior.confidence_region(), color = "orange", alpha = 0.3, label = "Fant Conf Region")

    plt.scatter(val_x, val_y, color = "grey", marker = "*", alpha = 0.5, label = "New Data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    # <<< Plot the updated posterior
    plt.show()


def testGP():
    from matplotlib import pyplot as plt
    train_x = torch.linspace(0, 3, 10).view(-1, 1).contiguous()
    train_y = torch.sin(6. * train_x) + 0.3 * torch.randn_like(train_x)

    # >>> Plot the training points
    plt.scatter(train_x, train_y, marker = "*", color = "black")
    plt.xlabel("x")
    plt.ylabel("y")
    # <<< Plot the training points

    from gp import train_gp
    model = train_gp(train_x, train_y.squeeze(), True, 50)
    test_x = torch.linspace(0, 8, 250).view(-1,1)
    with torch.no_grad():
        posterior = model.likelihood(model(test_x))

    # >>> Plot the posterior
    plt.plot(test_x, posterior.mean, color = "blue", label = "Post Mean")
    plt.fill_between(test_x.squeeze(), *posterior.confidence_region(), color = "blue", alpha = 0.3, label = "Post Conf Region")
    plt.scatter(train_x, train_y, color = "black", marker = "*", alpha = 0.5, label = "Training Data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    # <<< Plot the posterior

    val_x = torch.linspace(3, 5, 25).view(-1,1)
    val_y = torch.sin(6. * val_x) + 0.3 * torch.randn_like(val_x)

    cond_model = model.get_fantasy_model(val_x, val_y.squeeze())
    with torch.no_grad():
        updated_posterior = cond_model.likelihood(cond_model(test_x))

    # >>> Plot the updated posterior
    plt.plot(test_x, posterior.mean, color = "blue", label = "Post Mean")
    plt.fill_between(test_x.squeeze(), *posterior.confidence_region(), color = "blue", alpha = 0.3, label = "Post Conf Region")
    plt.scatter(train_x, train_y, color = "black", marker = "*", alpha = 0.5, label = "Training Data")

    plt.plot(test_x, updated_posterior.mean, color = "orange", label = "Fant Mean")
    plt.fill_between(test_x.squeeze(), *updated_posterior.confidence_region(), color = "orange", alpha = 0.3, label = "Fant Conf Region")

    plt.scatter(val_x, val_y, color = "grey", marker = "*", alpha = 0.5, label = "New Data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    # <<< Plot the updated posterior
    plt.show()
if __name__ == '__main__':
    testGP()