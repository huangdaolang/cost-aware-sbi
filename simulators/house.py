import torch


def threshold(n):
    """
    Generate thresholds for internal household transmission.
    """
    thres = torch.zeros(n-1)
    thres[0] = torch.exp(torch.tensor([1.0 / (n-1)]))
    if n > 2:
        for i in range(1, n-1):
            thres[i] = thres[i-1] + torch.exp(torch.tensor([1.0 / (n-i-1)]))
    return thres


def House_epi(n, k, lambda_L):
    """
    Simulate an epidemic transmission within a household.
    """
    sev = None

    t = threshold(n)
    print(t)
    q = torch.distributions.Gamma(k, k).sample((n,))
    t = torch.cat((t, 2*lambda_L*torch.sum(q).unsqueeze(0)))
    i = 0
    test = False
    while not test:
        i += 1
        if t[i-1] > (lambda_L * torch.sum(q[:i])):
            test = True
            sev = torch.sum(q[:i]).item()
    return i, sev


if __name__ == "__main__":

    n = 5  # Number of household members
    k = 2.0  # Shape parameter of the Gamma distribution for the infectious period
    exp_distribution = torch.distributions.Exponential(rate=torch.tensor([1.0]))
    lambda_L = exp_distribution.sample()
    print(lambda_L)
    # lambda_L = 10  # Infection rate within the household

    infected_count, severity_sum = House_epi(n, k, lambda_L)
    print(f"Number of infected in the household: {infected_count}, Total severity (sum of infectious periods): {severity_sum}")