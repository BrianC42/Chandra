'''
Created on May 15, 2019

@author: Brian
'''

if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(19680801)

    # example data
    mu = 100  # mean of distribution
    sigma = 15  # standard deviation of distribution
    
    np_2d = np.zeros([437, 6])
    
    for ndx_i in range (0, 5) :
        np_2d[:, ndx_i] = mu + sigma * np.random.randn(437)
        
    x = mu + sigma * np.random.randn(437)

    num_bins = 50

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, density=1)
    n, bins, patches = ax.hist(np_2d[:, 0], num_bins, density=1)
    
    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    ax.plot(bins, y, '--')
    ax.set_xlabel('Smarts')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

    pass