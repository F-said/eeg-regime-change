class Online:
    """Bayesian w/innovation

    Would it be possible to make a change-point
    detection algorithm with NO change-point, and
    rather "train as you go", with some metric for
    outlier detection.

    Well, PCA would definitley help, although it
    would also assume some time for "training"
    but maybe it could be minimal

    The problem with PCA is the fact that it needs
    the entire df to figure 

    Power Spectral Analysis of PCA components
    Needed: Live PCA (X)
    Needed: Live wavelet transform

    1) Check if wavelet transform of PCA is similair to original
    2) Find iterative PCA algo
    3) Find wavelet transform algo
    4) Find good way to combine the two without being too slow
    5) Run & get results
    """
    ...

class Offline:
    ...