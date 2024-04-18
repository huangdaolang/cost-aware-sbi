import torch


def homoSIR(l, k, device=torch.device('cpu')):
    N = 1000
    S = N - 1
    y = 1
    while y > 0:
        y -= 1
        if k == 0:
            I = torch.tensor(1.0, device=device)
        else:
            I = torch.distributions.Gamma(k, k).sample().to(device)
        Z = torch.poisson(l * I).int()  # number of infectious contacts

        if Z > 0:
            for _ in range(Z):
                u = torch.rand(1, device=device)
                if u < (S / N):
                    S -= 1
                    y += 1

    return N - S


if __name__ == "__main__":
    l = 3
    k = 1
    print(homoSIR(l, k, device=torch.device('cpu')))

