import torch

def bernSIR(n, beta, gamma, p):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = torch.tensor(0.0, device=device)
    MAT = torch.distributions.Bernoulli(torch.tensor([p], device=device)).sample((n, n)).squeeze()
    rowM = MAT.sum(dim=1)
    I = torch.zeros(n, device=device)
    I[0] = 1  # Set individual 1 infectious
    output = []  # Use a list for dynamic append
    count = 0  # Number of recoveries observed

    while (I == 1).sum() > 0:
        rec = (I == 1).sum()
        infe = torch.mv(MAT, (I == 1).float()).sum()
        rate = gamma * rec + beta * infe
        t += torch.distributions.Exponential(1 / rate).sample()
        u = torch.rand(1, device=device)

        if u <= beta * infe / (gamma * rec + beta * infe):
            S = MAT @ (I == 1).float()  # Project infection probabilities
            K = torch.multinomial(S, 1)  # Select an infectious individual
            J = torch.multinomial(MAT[K], 1)  # Select a susceptible contact
            if I[J] == 0:
                I[J] = 1  # Infect the chosen susceptible
        else:
            S = (I == 1).float()
            K = torch.multinomial(S, 1)  # Select a recovering individual
            I[K] = 2  # Recover the chosen individual
            count += 1
            output.append(t.item())

    return {'output': torch.tensor(output, device=device), 'count': count}

# Example usage
n = 100
beta = 0.1
gamma = 0.05
p = 0.05
result = bernSIR(n, beta, gamma, p)
print("Recovery times:", result['output'])
print("Number of recoveries:", result['count'])
