# coding = utf8

import numpy as np
import matplotlib.pyplot as plt

# def SOBI(X, n, p):
#     [m, N] = np.shape(X)
#     n = m
#     p = min(100, int(N / 3))
#
#     for i in range(m):
#         avg = np.mean(X[i, :])
#         for j in range(N):
#             X[i, j] -= avg
#
#     [UU, SS, VV] = np.linalg.svd(np.transpose(X))
#     S = np.zeros((len(SS), len(SS)))
#     for i in range(len(SS)):
#         S[i, i] = SS[i]
#     Q = np.dot(np.linalg.pinv(S), np.transpose(VV))
#     X[:, :] = np.dot(Q, X[:, :])
#     k = 0
#     pm = p * m
#     M = np.zeros((n, pm), dtype = complex)
#     for u in range(0, pm, m):
#         k += 1
#         Rxp = np.dot(X[:, k:N], np.transpose(X[:, 0:N-k])) / (N - k + 1)
#         M[:, u:u+m] = np.dot(np.linalg.norm(Rxp, 'fro'), Rxp)
#
#     epsil = 1 / np.sqrt(N) / 100
#     encore = 1
#     V = np.zeros((m, m), dtype = complex)
#     for i in range(m):
#         V[i, i] = 1
#     while encore:
#         encore = 0
#         for p in range(0, m - 1):
#             for q in range(p + 1, m):
#                 g=[M[p, range(p, pm, m)] - M[q, range(q, pm, m)],
#                    M[p, range(q, pm, m)] + M[q, range(p, pm, m)],
#                    1j * (M[q, range(p, pm, m)] - M[p, range(q, pm, m)])]
#                 D, vcp = np.linalg.eig((np.dot(g, np.transpose(g))).real)
#                 la = np.sort(D)
#                 K = 0
#                 for i in range(len(D)):
#                     if D[i] == la[2]:
#                         K = i
#                 angles = vcp[:, K]
#                 angles = np.sign(angles[0]) * angles
#                 c = np.sqrt(0.5 + angles[0] / 2)
#                 sr = 0.5 * (angles[1] - 1j*angles[2])/c
#                 sc = np.conj(sr)
#                 oui = np.abs(sr) > epsil
#                 encore |= oui
#                 if oui:
#                     colp = M[:, range(p, pm, m)]
#                     colq = M[:, range(q, pm, m)]
#                     M[:, range(p, pm, m)] = c * colp + sr * colq
#                     M[:, range(q, pm, m)] = c * colq - sc * colp
#                     rowp = M[p, :]
#                     rowq = M[q, :]
#                     M[p, :] = c * rowp + sc * rowq
#                     M[q, :] = c * rowq - sr * rowp
#                     temp = V[:, p]
#                     V[:, p] = c * V[:, p] + sr * V[:, q]
#                     V[:, q] = c * V[:, q] - sc * temp
#     H = np.dot(np.linalg.pinv(Q), V)
#     S = np.dot(np.transpose(V), X[:, :])
#     return H, S



# Test

def stdcov(X, tau):
    m, N = np.shape(X)
    m1 = np.zeros((m, 1))
    m2 = np.zeros((m, 1))
    R = np.dot(X[:, 0:N-tau], np.transpose(X[:, tau:N])) / (N - tau)
    for i in range(m):
        m1[i] = np.mean(X[i, 0:N-tau])
        m2[i] = np.mean(X[i, tau:N])
    C = R - np.dot(m1, np.transpose(m2))
    C = (C + np.transpose(C)) / 2
    return C

def joint_diag(A, jthresh):
    m, nm = np.shape(A)
    D = np.array(A, dtype = complex)
    B = np.array([[1, 0, 0], [0, 1, 1], [0, -1j, 1j]], dtype = complex)
    Bt = np.transpose(B)
    # Ip = np.zeros((1, nm))
    # Iq = np.zeros((1, nm))
    # g = np.zeros((3, m), dtype = complex)
    # G = np.zeros(3, dtype = complex)
    # vcp = np.zeros((3, 3))
    # D = np.zeros((3, 3))
    # la = np.zeros((3, 1))
    # angles = np.zeros((3, 1), dtype = complex)
    # pair = np.zeros((1, 2), dtype = complex)
    V = np.zeros((m, m), dtype = complex)
    for i in range(m):
        V[i, i] = 1.0
    encore = 1
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = range(p, nm, m)
                Iq = range(q, nm, m)
                g = [D[p, Ip] - D[q, Iq], D[p, Iq], D[q, Ip]]
                D1, vcp = np.linalg.eig(np.dot(np.dot(B, np.dot(g, np.transpose(g))), Bt).real)
                la = np.sort(D1)
                K = 0
                for i in range(len(D1)):
                    if D1[i] == la[2]:
                        K = i
                angles = vcp[:, K]
                if angles[0] < 0:
                    angles = -1 * angles
                c = np.sqrt(0.5 + angles[0] / 2)
                s = 0.5 * (angles[1] - 1j * angles[2]) / c
                if np.abs(s) > jthresh:
                    encore = 1
                    G = np.array([[c , -np.conj(s)], [s, c]], dtype = complex)
                    V[:, [p,q]] = np.dot(V[:, [p, q]], G)
                    D[[p, q], :] = np.dot(np.transpose(G), D[[p, q], :])
                    # A[p, :] = np.dot(np.transpose(G), A[p, :])
                    # V[:, q] = np.dot(V[:, q], G)
                    # A[q, :] = np.dot(np.transpose(G), A[q, :])
                    D[:, Ip] = c * D[:, Ip] + s * D[:, Iq]
                    D[:, Iq] = c * D[:, Iq] - np.conj(s) * D[:, Ip]
    return V, D

def SOBI(X, n, num_tau):

    m, N = np.shape(X)
    tau = range(num_tau)
    tiny = 10 ** (-8)
    Rx = stdcov(X, 0)
    [uu, dd, vv] = np.linalg.svd(Rx)
    d = np.zeros((len(dd), len(dd)))
    for i in range(len(dd)):
        d[i, i] = dd[i]
    Q = np.dot(np.sqrt(np.linalg.pinv(d[0:n, 0:n])), np.transpose(uu[:, 0:n]))
    z = np.dot(Q, X)

    Rz = np.zeros((n, num_tau * n))
    for i in range(1, num_tau + 1):
        ii = range((i - 1) * n, i * n)
        Rz[:, ii] = stdcov(z, tau[i - 1])

    v, d = joint_diag(Rz, tiny)

    return np.dot(np.transpose(v), Q)

if __name__ == '__main__':
    s1 = np.cos(2 * np.pi * 0.013 * np.arange(1, 201))
    s2 = np.sign(np.sin(2 * np.pi * 0.013 * np.arange(1, 201)))
    s3 = np.random.randn(200)
    print np.shape(s1), np.shape(s2), np.shape(s3)
    S = np.ones((3, 200))
    print np.shape(S)
    test = [1, 2, 3, 4]
    S[0, :] = s1
    S[1, :] = s2
    S[2, :] = s3
    A = np.random.randn(3, 3)
    noiseamp = 10 ** (-20 / 20)
    X = np.dot(A , S) # + noiseamp * np.random.randn(3, 200)
    H = SOBI(X, 3, 30)
    Se = np.dot(H, X)

    plt.figure(1)
    for i in range(3):
        p = plt.subplot(331 + i)
        p.plot(Se[i, :])
    for i in range(3):
        p = plt.subplot(334 + i)
        p.plot(S[i, :])
    for i in range(3):
        p = plt.subplot(337 + i)
        p.plot(X[i, :])
    plt.show()


