import matplotlib.pyplot as plt
import openturns as ot
import numpy as np
import math


def main():
    x1 = ot.Normal(0, 1)
    x2 = ot.Normal(0, 1)

    #For normal copula - initialised to N dimensional identity matrix
    R = ot.CorrelationMatrix(2)
    R[0, 1] = 0.7
    R[1, 0] = 0.7

    #Specify copula and inputs
    copula_C = ot.ClaytonCopula(2.2)
    copula_N = ot.NormalCopula(R)

    X = ot.ComposedDistribution([x1, x2], copula_C)
    sample = X.getSample(1000000)
    print(np.array(sample))
    print(np.corrcoef(np.array(sample)[:, 0], np.array(sample)[:, 1]))

    plt.scatter(sample[:, 0], sample[:, 1], s=2)
    plt.show()


if __name__ == "__main__":
    main()


