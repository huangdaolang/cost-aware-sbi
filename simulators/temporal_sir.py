import torch
import time


def simSIR(beta, gamma, device):
    N = 5000  # population size
    I = torch.tensor(1.0, device=device)  # infected individuals
    S = torch.tensor(N - 1.0, device=device)  # susceptible individuals
    t = torch.tensor(0.0, device=device)  # time
    times = [t.item()]
    types = [1]  # 1 for infection, 2 for removal

    while I > 0:
        rate = (beta / N) * I * S + gamma * I
        t += torch.distributions.Exponential(1 / rate).sample()
        times.append(t.item())

        if torch.rand(1, device=device) < (beta * S) / ((beta * S) + N * gamma):
            I += 1
            S -= 1
            types.append(1)
        else:
            I -= 1
            types.append(2)

    removal_times = [times[i] - min([times[j] for j in range(len(types)) if types[j] == 2]) for i in range(len(times))
                     if types[i] == 2]
    final_size = N - S.item()
    T = times[-1]
    # print(removal_times)

    return {'removal_times': torch.tensor(removal_times, device=device),
            'final_size': torch.tensor(final_size, device=device),
            'T': torch.tensor(T, device=device)}


if __name__ == "__main__":
    N = 10
    # beta = torch.linspace(0.01, 0.5, N)
    beta = 1
    # gamma = 0.05
    gamma = torch.linspace(0.01, 0.5, N)
    times = torch.zeros(N)
    for i in range(N):
        st = time.time()
        for _ in range(10):
            result = simSIR(beta, gamma[i], device=torch.device('cpu'))
        et = time.time()
        times[i] = (et - st) / 100

    print(times)
