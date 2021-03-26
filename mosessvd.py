"""
Implementation of MOSES SVD (2nd algorithm)
arXiv:1806.01304v3
"""

import numpy as np
# import scipy.linalg as spLinalg
from numba import jit


@jit(nopython=True)
def gramSchmidt_T(A):
    """
    Applies the Gram-Schmidt method to A
    and returns Q and R, so Q*R = A.
    """
    R = np.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)
    Q = np.zeros(A.shape, dtype=A.dtype)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j]*Q[:, k]
    return Q, R




class MOSESSVD():
    def __init__(self, rank, dtype=np.complex64):
        """
        Implementation of the MOSES SVD algorithm (2nd algorithm)
        arXiv:1806.01304v3

        Parameters
        ----------
        rank : int
            The rank for the SVD. The first r largest singular values are
            calculated.
        dtype : numpy.dtype, optional
            The desired datatype. The default is np.complex64.
        Returns
        -------
        None.

        """
        self.X = None
        self.rank = rank
        self.S = None
        self.Gamma = None
        self.Q = None
        self.first_iter = True
        self.dtype = dtype

    # @profile
    def update(self, x):
        """
        Execute one iteration of the MOSES SVD algorithm.

        Parameters
        ----------
        x : numpy.ndarray
            Chunk of input data that is added to the SVD.

        Returns
        -------
        None.

        """
        if self.first_iter:
            self.S, self.Gamma, self.Q = self._update_first(x, self.rank, self.dtype)
            self.first_iter = False
        else:
            self.S, self.Gamma, self.Q = self._update(x, self.S, self.Gamma, self.Q, self.rank, self.dtype)

    # @profile
    @staticmethod
    @jit(nopython=True)
    # def _update_first(self, x, r, dtype):
    def _update_first(x, r, dtype):
        """
        Initialize the algorithm with a regular SVD.

        Parameters
        ----------
        x : numpy.ndarray
            Chunk of input data that is added to the SVD.
        r : int
            The rank for the SVD. The first r largest singular values are
            calculated.
        dtype : numpy.dtype
            The desired datatype.

        Returns
        -------
        S : numpy.ndarray
            U matrix.
        Gamma : numpy.ndarray
            Sigma matrix.
        Q : numpy.ndarray
            V matrix.

        """

        S, Gamma, Q = np.linalg.svd(x, full_matrices=False)
        Q = Q.conj().T
        Gamma = np.diag(Gamma)

        S = S[:, :r].astype(dtype)
        Gamma = Gamma[:r, :r].astype(dtype)
        Q = Q[:, :r].astype(dtype)

        return S, Gamma, Q

    # @profile
    @staticmethod
    @jit(nopython=True)
    # def _update(self, x, S, Gamma, Q, r, dtype):
    def _update(x, S, Gamma, Q, r, dtype):
        """
        The main loop of MOSES SVD. It iteratively updates U, s, V, by taking
        in data in chunks.

        Parameters
        ----------
        x : numpy.ndarray
            Chunk of input data that is added to the SVD.
        S : numpy.ndarray
            U matrix.
        Gamma : numpy.ndarray
            Sigma matrix.
        Q : numpy.ndarray
            V matrix.
        r : int
            The rank for the SVD. The first r largest singular values are
            calculated.
        dtype : numpy.dtype
            The desired datatype.

        Returns
        -------
        S : numpy.ndarray
            U matrix.
        Gamma : numpy.ndarray
            Sigma matrix.
        Q : numpy.ndarray
            V matrix.

        """

        # n = x.shape[0]
        b = x.shape[1]

        qq = S.conj().T.dot(x)
        z = x - S.dot(qq)

        ss, v = np.linalg.qr(z)
        ss = ss.astype(dtype)
        v = v.astype(dtype)

        M_1 = np.hstack((Gamma, qq))
        M_2 = np.hstack((np.zeros((b, r)).astype(dtype), v))
        M = np.vstack((M_1, M_2))

        u, Gamma, q_h = np.linalg.svd(M, full_matrices=False)
        u = u[:, :r].astype(dtype)
        Gamma = np.diag(Gamma)[:r, :r].astype(dtype)
        q_h = q_h.conj().T[:, :r].astype(dtype)

        S = np.hstack((S, ss)).dot(u)
        S = S[:, :r]

        Q_1 = np.hstack((Q, np.zeros((Q.shape[0], b)))).astype(dtype)
        Q_2 = np.hstack((np.zeros((b, Q.shape[1])), np.eye(b))).astype(dtype)
        Q = np.vstack((Q_1, Q_2)).dot(q_h)
        Q = Q[:, :r]

        # optional alternative mentioned in the paper
        # if Q.shape[0] < n:
        #     Q_1 = np.hstack([Q, np.zeros((Q.shape[0], b))])
        #     Q_2 = np.hstack([np.zeros((b, Q.shape[1])), np.eye(b)])
        #     Q = np.vstack([Q_1, Q_2]).dot(q_h)
        #     Q = Q[:, :r]
        # else:
        #     Q = q

        return S, Gamma, Q
    
    # helper method to compute a full SVD from start to end
    # @profile
    def iterated_svd(self, inp, b):
        """
        Helper method to compute a full SVD from start to end.

        Parameters
        ----------
        inp : numpy.ndarray
            The input data where the snapshots are the collumns in a matrix.
        b : int
            The horizontal size for the chunks given to MOSES SVD.

        Returns
        -------
        Returns
        -------
        S : numpy.ndarray
            U matrix.
        Gamma : numpy.ndarray
            Sigma matrix.
        Q : numpy.ndarray
            V matrix.
        """
        shape = inp.shape
        for i in range(0, shape[1], b):
            chunk = inp[:, i:i+b].astype(np.complex64)
            self.update(chunk)

        return self.S, np.diagonal(self.Gamma), self.Q


# debug tests
if __name__ == "__main__":
    Y = np.random.rand(int(1e4), int(1e3))

    moses = MOSESSVD(rank=10)
    # for i in range(0, 20, 4):
    #     x = a[:,i:i+4]
    #     moses.update(x)
    moses.iterated_svd(Y, b=16)

    S = moses.S
    G = moses.Gamma
    Q = moses.Q

    _ = np.linalg.svd(Y, full_matrices=True)

    Yr = S.dot(G.dot(Q.conj().T))
