#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>

using namespace Eigen;

MatrixXf SchurLog(const MatrixXf& V) {
    int n = V.rows();
    RealSchur<MatrixXf> schur(V);
    MatrixXf Q = schur.matrixU();
    MatrixXf S = schur.matrixT();

    MatrixXf logS = MatrixXf::Zero(n, n);
    int k = 0;
    while (k < n-1) {
        if (abs(S(k+1, k)) < 1e-10) {
            k++;
        } else {
            // there is a 2x2 block
            float phi = atan2(S(k+1, k), S(k, k));
            logS(k, k+1) = -phi;
            logS(k+1, k) = phi;
            k += 2;
        }
    }

    // for log matrix
    MatrixXf logV = Q * logS * Q.transpose();
    return logV;
}

MatrixXf solvsymsyl(const MatrixXf& A, const MatrixXf& C) {
    SelfAdjointEigenSolver<Eigen::MatrixXf> eigh(A);
    VectorXf L = eigh.eigenvalues();
    MatrixXf Q = eigh.eigenvectors();

    MatrixXf C2 = Q.transpose() * C * Q;
    int n = C.rows();
    MatrixXf X = MatrixXf::Zero(n, n);
    for (int j = 0; j < n; j++) {
        for (int k = j+1; k < n; k++) {
            X(j, k) = (C2(j, k) / (L(j) + L(k)));
            X(k, j) = -X(j, k);
        }
    }
    X = Q * X * Q.transpose();
    X = 0.5 * (X - X.transpose());
    return X;
}

MatrixXf Cayley(const MatrixXf& X) {
    int p = X.rows();
    MatrixXf Xminus = -0.5*X + MatrixXf::Identity(p, p);
    MatrixXf Xplus = 0.5*X + MatrixXf::Identity(p, p);
    MatrixXf Cay = Xminus.inverse() * Xplus;
    return Cay;
}

MatrixXf general_exp(const MatrixXf& U, const MatrixXf& Delta, float beta) {
    int n = U.rows();
    int p = U.cols();

    MatrixXf A = U.transpose() * Delta;
    MatrixXf K = Delta - U * A;

    MatrixXf thinQ(MatrixXf::Identity(n,p)), Q, B;
    HouseholderQR<MatrixXf> qr(K);
    Q = qr.householderQ() * thinQ;
    B = Q.transpose() * K;

    MatrixXf L(2*p, 2*p);
    L << 2*beta*A, -B.transpose(), B, MatrixXf::Zero(p, p);
    MatrixXf expL = L.exp();

    MatrixXf U1 = U * expL.topLeftCorner(p, p) + Q * expL.bottomLeftCorner(p, p);

    // check if beta ~ 0.5
    if (abs(beta - 0.5) < 1e-5) {
        return U1;
    } else {
        return U1 * ((1-2*beta)*A).exp();
    }
}

MatrixXf general_log(const MatrixXf& U0, const MatrixXf& U1, float beta) {
    int n = U0.rows();
    int p = U0.cols();

    MatrixXf M = U0.transpose() * U1;
    MatrixXf U0orth = U1 - U0 * M;

    MatrixXf thinQ(MatrixXf::Identity(n,p)), Qhat, Nhat;
    HouseholderQR<MatrixXf> qr(U0orth);
    Qhat = qr.householderQ() * thinQ;
    Nhat = Qhat.transpose() * U0orth;

    MatrixXf MN(2*p, p);
    MN << M, Nhat;

    HouseholderQR<MatrixXf> qr_mn(MN);
    MatrixXf cols = qr_mn.householderQ();
    cols = cols.rightCols(p);

    MatrixXf Ohat = cols.topRightCorner(p, p);
    MatrixXf Phat = cols.bottomRightCorner(p, p);

    JacobiSVD<MatrixXf> svd(Phat, ComputeThinU | ComputeThinV);
    MatrixXf R = svd.matrixU();
    MatrixXf Sigma = svd.singularValues();
    MatrixXf R1 = svd.matrixV();

    MatrixXf Q = Qhat * R;
    MatrixXf N = R.transpose() * Nhat;
    MatrixXf O = Ohat * R1;
    MatrixXf P = Sigma.asDiagonal();
    MatrixXf V(2*p, 2*p);
    V << M, O, N, P;

    // Ensure V \in SO(n)
    float DetV = V.determinant();
    V.col(p) *= (DetV < 0) ? -1 : 1;

    // approximate Ahat_0
    MatrixXf LV0, E, F, S, Ahat;
    LV0 = SchurLog(V);
    E = LV0.topLeftCorner(p, p);
    F = LV0.bottomLeftCorner(p, p);

    S = 0.5 * MatrixXf::Identity(p, p) - ((1 - 2*beta) / 12) * F.transpose() * F;
    Ahat = solvsymsyl(S, E);

    MatrixXf A, B, C, Vblock, LV, Gamma;
    A = Ahat;
    int iter = 20;
    for (int k = 0; k < iter; k++) {
        Ahat = A;

        Vblock = V;
        Vblock.topLeftCorner(2*p, p) = (V.topLeftCorner(2*p, p) * (-(1-2*beta)*Ahat).exp());

        LV = SchurLog(Vblock);
        LV = 0.5 * (LV - LV.transpose());

        A = LV.topLeftCorner(p, p) / (2*beta);
        B = LV.bottomLeftCorner(p, p);
        C = LV.bottomRightCorner(p, p);

        if (C.norm() + (Ahat-A).norm() < 1e-5) {
            break;
        }

        S = ((1/12) * B * B.transpose() * B) - 0.5 * MatrixXf::Identity(p, p);

        Gamma = solvsymsyl(S, C);
        V.topRightCorner(2*p, p) = V.topRightCorner(2*p, p) * Gamma.exp();
    }

    MatrixXf Delta = U0 * A + Q * B;
    return Delta;
}

MatrixXf Stiefel_Log_alg(const MatrixXf& U0, const MatrixXf& U1) {
    // Get dimensions
    int n = U0.rows();
    int p = U0.cols();
    
    // Step 1
    MatrixXf M = U0.transpose() * U1;
    
    // Step 2
    MatrixXf U0orth = U1 - U0 * M;
    
    // Thin QR decomposition
    MatrixXf thinQ(MatrixXf::Identity(n,p)), Q, N;
    HouseholderQR<MatrixXf> qr(U0orth);
    Q = qr.householderQ() * thinQ;
    N = Q.transpose() * U0orth;

    // Step 3
    MatrixXf MN(2*p, p);
    MN << M, N;

    // Orthogonal completion
    HouseholderQR<MatrixXf> qr_mn(MN);
    MatrixXf Vright = qr_mn.householderQ();
    Vright = Vright.rightCols(p);

    MatrixXf V(2*p, 2*p);
    V << MN, Vright;
    
    // Ensure V \in SO(n)
    float DetV = V.determinant();
    V.col(p) *= (DetV < 0) ? -1 : 1;

    // Step 4
    MatrixXf LV, C, Msym, Phi;
    int iter = 20;
    // if (p > 10) {
    //     iter = 25 - p;
    // }
    for (int k = 0; k < iter; k++) {
        LV = SchurLog(V);

        LV = 0.5 * (LV - LV.transpose());

        C = LV.bottomRightCorner(p, p);
        if (C.norm() < 1e-5) {
            break;
        }

        Msym = (-1.0/12.0) * LV.bottomLeftCorner(p, p) * LV.topRightCorner(p, p);
        Msym = Msym - 0.5 * MatrixXf::Identity(p, p);

        C = -solvsymsyl(Msym, C);

        Phi = Cayley(-C);

        // step 10: rotate the last p columns
        V.rightCols(p) = V.rightCols(p) * Phi;
    }

    // prepare output                         |A  -B'|
    // upon convergence, we have  logm(V) =   |B   0 | = LV
    //     A = LV(1:p,1:p);     B = LV(p+1:2*p, 1:p)
    // Delta = U0*A+Q*B
    MatrixXf Delta = U0 * LV.topLeftCorner(p, p) + Q * LV.bottomLeftCorner(p, p);

    // project Delta to the tangent space
    Delta = Delta - 0.5 * U0 * (U0.transpose() * Delta + Delta.transpose() * U0);
    
    return Delta;
}

MatrixXf Stiefel_Log_alg_iter(const MatrixXf& U0, const MatrixXf& U1, int iter) {
    // Get dimensions
    int n = U0.rows();
    int p = U0.cols();
    
    // Step 1
    MatrixXf M = U0.transpose() * U1;
    
    // Step 2
    MatrixXf U0orth = U1 - U0 * M;
    
    // Thin QR decomposition
    MatrixXf thinQ(MatrixXf::Identity(n,p)), Q, N;
    HouseholderQR<MatrixXf> qr(U0orth);
    Q = qr.householderQ() * thinQ;
    N = Q.transpose() * U0orth;

    // Step 3
    MatrixXf MN(2*p, p);
    MN << M, N;

    // Orthogonal completion
    HouseholderQR<MatrixXf> qr_mn(MN);
    MatrixXf Vright = qr_mn.householderQ();
    Vright = Vright.rightCols(p);

    MatrixXf V(2*p, 2*p);
    V << MN, Vright;
    
    // Ensure V \in SO(n)
    float DetV = V.determinant();
    V.col(p) *= (DetV < 0) ? -1 : 1;

    // Step 4
    MatrixXf LV, C, Msym, Phi;
    for (int k = 0; k < iter; k++) {
        LV = SchurLog(V);

        LV = 0.5 * (LV - LV.transpose());

        C = LV.bottomRightCorner(p, p);
        if (C.norm() < 1e-5) {
            break;
        }

        Msym = (-1.0/12.0) * LV.bottomLeftCorner(p, p) * LV.topRightCorner(p, p);
        Msym = Msym - 0.5 * MatrixXf::Identity(p, p);

        C = -solvsymsyl(Msym, C);

        Phi = Cayley(-C);

        // step 10: rotate the last p columns
        V.rightCols(p) = V.rightCols(p) * Phi;
    }

    // prepare output                         |A  -B'|
    // upon convergence, we have  logm(V) =   |B   0 | = LV
    //     A = LV(1:p,1:p);     B = LV(p+1:2*p, 1:p)
    // Delta = U0*A+Q*B
    MatrixXf Delta = U0 * LV.topLeftCorner(p, p) + Q * LV.bottomLeftCorner(p, p);

    // project Delta to the tangent space
    Delta = Delta - 0.5 * U0 * (U0.transpose() * Delta + Delta.transpose() * U0);
    
    return Delta;
}

MatrixXf Stiefel_Log_alg_noproj(const MatrixXf& U0, const MatrixXf& U1) {
    // Get dimensions
    int n = U0.rows();
    int p = U0.cols();
    
    // Step 1
    MatrixXf M = U0.transpose() * U1;
    
    // Step 2
    MatrixXf U0orth = U1 - U0 * M;
    
    // Thin QR decomposition
    MatrixXf thinQ(MatrixXf::Identity(n,p)), Q, N;
    HouseholderQR<MatrixXf> qr(U0orth);
    Q = qr.householderQ() * thinQ;
    N = Q.transpose() * U0orth;

    // Step 3
    MatrixXf MN(2*p, p);
    MN << M, N;

    // Orthogonal completion
    HouseholderQR<MatrixXf> qr_mn(MN);
    MatrixXf Vright = qr_mn.householderQ();
    Vright = Vright.rightCols(p);

    MatrixXf V(2*p, 2*p);
    V << MN, Vright;
    
    // Ensure V \in SO(n)
    float DetV = V.determinant();
    V.col(p) *= (DetV < 0) ? -1 : 1;

    // Step 4
    MatrixXf LV, C, Msym, Phi;
    int iter = 20;
    // if (p > 10) {
    //     iter = 25 - p;
    // }
    for (int k = 0; k < iter; k++) {
        LV = SchurLog(V);

        LV = 0.5 * (LV - LV.transpose());

        C = LV.bottomRightCorner(p, p);
        if (C.norm() < 1e-5) {
            break;
        }

        Msym = (-1.0/12.0) * LV.bottomLeftCorner(p, p) * LV.topRightCorner(p, p);
        Msym = Msym - 0.5 * MatrixXf::Identity(p, p);

        C = -solvsymsyl(Msym, C);

        Phi = Cayley(-C);

        // step 10: rotate the last p columns
        V.rightCols(p) = V.rightCols(p) * Phi;
    }

    // prepare output                         |A  -B'|
    // upon convergence, we have  logm(V) =   |B   0 | = LV
    //     A = LV(1:p,1:p);     B = LV(p+1:2*p, 1:p)
    // Delta = U0*A+Q*B
    MatrixXf Delta = U0 * LV.topLeftCorner(p, p) + Q * LV.bottomLeftCorner(p, p);
    
    return Delta;
}

MatrixXf Stiefel_Log_alg_part(const MatrixXf& U0, const MatrixXf& U1) {
    // Get dimensions
    int n = U0.rows();
    int p = U0.cols();
    
    // Step 1
    MatrixXf M = U0.transpose() * U1;
    
    // Step 2
    MatrixXf U0orth = U1 - U0 * M;
    
    // Thin QR decomposition
    MatrixXf thinQ(MatrixXf::Identity(n,p)), Q, N;
    HouseholderQR<MatrixXf> qr(U0orth);
    Q = qr.householderQ() * thinQ;
    N = Q.transpose() * U0orth;

    // Step 3
    MatrixXf MN(2*p, p);
    MN << M, N;

    // Orthogonal completion
    HouseholderQR<MatrixXf> qr_mn(MN);
    MatrixXf Vright = qr_mn.householderQ();
    Vright = Vright.rightCols(p);

    MatrixXf V(2*p, 2*p);
    V << MN, Vright;
    
    // Ensure V \in SO(n)
    float DetV = V.determinant();
    V.col(p) *= (DetV < 0) ? -1 : 1;

    // Step 4
    MatrixXf LV, C, Msym, Phi;

    // iteration 1
    LV = SchurLog(V);

    LV = 0.5 * (LV - LV.transpose());

    C = LV.bottomRightCorner(p, p);

    Msym = (-1.0/12.0) * LV.bottomLeftCorner(p, p) * LV.topRightCorner(p, p);
    Msym = Msym - 0.5 * MatrixXf::Identity(p, p);

    C = -solvsymsyl(Msym, C);

    Phi = Cayley(-C);

    // step 10: rotate the last p columns
    V.rightCols(p) = V.rightCols(p) * Phi;

    // iteration 2
    // LV = SchurLog(V);

    // LV = 0.5 * (LV - LV.transpose());

    // C = LV.bottomRightCorner(p, p);

    // Msym = (-1.0/12.0) * LV.bottomLeftCorner(p, p) * LV.topRightCorner(p, p);
    // Msym = Msym - 0.5 * MatrixXf::Identity(p, p);

    // C = -solvsymsyl(Msym, C);

    // Phi = Cayley(-C);

    // // step 10: rotate the last p columns
    // V.rightCols(p) = V.rightCols(p) * Phi;

    // prepare output                         |A  -B'|
    // upon convergence, we have  logm(V) =   |B   0 | = LV
    //     A = LV(1:p,1:p);     B = LV(p+1:2*p, 1:p)
    // Delta = U0*A+Q*B
    MatrixXf Delta = U0 * LV.topLeftCorner(p, p) + Q * LV.bottomLeftCorner(p, p);

    MatrixXf A = U0.transpose() * Delta;
    Delta = Delta - 0.5 * U0 * (A + A.transpose());
    
    return Delta;
}

MatrixXf Stiefel_Exp(const MatrixXf& U0, const MatrixXf& Delta) {
    int n = U0.rows();
    int p = U0.cols();

    MatrixXf A = U0.transpose() * Delta;
    MatrixXf K = Delta - U0 * A;

    MatrixXf thinQ(MatrixXf::Identity(n,p)), QE, Re;
    HouseholderQR<MatrixXf> qr(K);
    QE = qr.householderQ() * thinQ;
    Re = QE.transpose() * K;

    MatrixXf L(2*p, 2*p);
    L << A, -Re.transpose(), Re, MatrixXf::Zero(p, p);
    MatrixXf MNe = L.exp();

    MatrixXf U1 = U0 * MNe.topLeftCorner(p, p) + QE * MNe.bottomLeftCorner(p, p);
    return U1;
}


float metric(const MatrixXf& U, const MatrixXf& Delta1, const MatrixXf& Delta2) {
    int n = U.rows();
    return (Delta1.transpose() * (MatrixXf::Identity(n, n) - 0.5 * U * U.transpose()) * Delta2).trace();
}

float approx_dist(const MatrixXf& U0, const MatrixXf& U1) {
    MatrixXf Delta = Stiefel_Log_alg_part(U0, U1);
    return sqrt(metric(U0, Delta, Delta));
}

float dist(const MatrixXf& U0, const MatrixXf& U1) {
    MatrixXf Delta = Stiefel_Log_alg(U0, U1);
    return sqrt(metric(U0, Delta, Delta));
}

std::vector<std::pair<size_t, size_t>> same_atom_pairs(const std::vector<int>& atoms) {
    std::vector<std::pair<size_t, size_t>> pairs;
    std::unordered_map<int, std::vector<size_t>> indices;

    size_t n = atoms.size();

    for (size_t i = 0; i < n; ++i) {
        int atom = atoms[i];
        if (indices.find(atom) == indices.end()) {
            indices[atom] = std::vector<size_t>{i};
        } else {
            for (size_t j : indices[atom]) {
                pairs.push_back(std::make_pair(j, i));
            }
            indices[atom].push_back(i);
        }
    }

    return pairs;
}

std::vector<size_t> permute(const std::vector<int>& atoms) {
    std::random_device rd;
    std::default_random_engine rng(rd());

    // Create a vector of indices from 0 to atoms.size() - 1
    std::vector<size_t> perm(atoms.size());
    for (size_t i = 0; i < atoms.size(); ++i) {
        perm[i] = i;
    }

    // Shuffle the indices randomly
    std::shuffle(perm.begin(), perm.end(), rng);

    // Sort the shuffled indices based on the values of atoms
    std::sort(perm.begin(), perm.end(), [&atoms](size_t i1, size_t i2) {
        return atoms[i1] < atoms[i2];
    });

    return perm;
}

std::vector<size_t> local_search(std::vector<size_t>& perm, float best_cost, const std::vector<std::pair<size_t, size_t>>& atom_pairs, const MatrixXf& U0, const MatrixXf& U1, const int limit) {
    std::vector<size_t> best_perm = perm;
    float cost;

    int calls = 0;
    bool stop = false;
    bool found = false;

    while (true) {
        found = false;
        for (auto pair : atom_pairs) {
            int i = pair.first;
            int j = pair.second;
            std::swap(perm[i], perm[j]);
            cost = approx_dist(U1, U0(perm, all));
            calls += 1;

            if (cost < best_cost) {
                best_cost = cost;
                best_perm = perm;
                found = true;
            } else {
                std::swap(perm[i], perm[j]);
            }

            if (calls > limit) {
                stop = true;
                break;
            }
        }
        if (!found) {
            break;
        }

        if (stop) {
            break;
        }
    }

    return best_perm;
}

std::vector<size_t> argsort(const std::vector<int>& v) {
    // Initialize vector of indices
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        idx[i] = i;
    }

    // Sort indices based on values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
        return v[i1] < v[i2];
    });

    return idx;
}

MatrixXf OT_permutation_reflection(const std::vector<int>& atoms, const MatrixXf& U0, const MatrixXf& U1, const int restarts, const int limit) {
    std::random_device rd;
    std::default_random_engine rng(rd());

    std::vector<size_t> idx = argsort(atoms);

    MatrixXf ordered_U0 = U0(idx, all);
    MatrixXf ordered_U1 = U1(idx, all);

    std::vector<int> sorted_atoms;
    for (size_t i : idx) {
        sorted_atoms.push_back(atoms[i]);
    }

    std::vector<size_t> inv;
    for (size_t i = 0; i < idx.size(); ++i) {
        inv.push_back(0);
    }
    for (size_t i = 0; i < idx.size(); ++i) {
        inv[idx[i]] = i;
    }

    std::vector<std::pair<size_t, size_t>> atom_pairs = same_atom_pairs(sorted_atoms);
    std::shuffle(atom_pairs.begin(), atom_pairs.end(), rng);

    // for each reflection, sample 5 random permutations
    // then use the best to initialize local search

    MatrixXf reflected_U0, best_reflected_U0;
    std::vector<size_t> best_perm, p;
    float cost;
    float best_cost = 1000.0;

    std::vector<float> signs = {1.0, -1.0};
    for (float sign_x : signs) {
        for (float sign_y : signs) {
            for (float sign_z : signs) {
                for (int i = 0; i < restarts; ++i) {
                    reflected_U0 = ordered_U0;
                    reflected_U0.col(0) *= sign_x;
                    reflected_U0.col(1) *= sign_y;
                    reflected_U0.col(2) *= sign_z;
                    p = permute(sorted_atoms);
                    cost = approx_dist(ordered_U1, reflected_U0(p, all));
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_reflected_U0 = reflected_U0;
                        best_perm = p;
                    }
                }
            }
        }
    }

    best_perm = local_search(best_perm, best_cost, atom_pairs, best_reflected_U0, ordered_U1, limit);
    return best_reflected_U0(best_perm, all)(inv, all);
}

PYBIND11_MODULE(stiefel_log, m) {
    m.def("Stiefel_Log_alg", &Stiefel_Log_alg, "Calculate the Stiefel logarithm at base_point U0 and target_point U1");
    m.def("Stiefel_Log_alg_iter", &Stiefel_Log_alg_iter, "Calculate the Stiefel logarithm at base_point U0 and target_point U1 using a specified number of iterations");
    m.def("Stiefel_Log_alg_noproj", &Stiefel_Log_alg_noproj, "Calculate the Stiefel logarithm at base_point U0 and target_point U1");
    m.def("Stiefel_Exp", &Stiefel_Exp, "Calculate the Stiefel exponential at base_point U0 and tangent_vector Delta");
    // m.def("sample_bridge", &sample_bridge, "Simulate a bridge backwards in time until t_goal, starting at ONB_noise and ending at ONB_data with noise level gamma");
    m.def("Stiefel_Log_alg_part", &Stiefel_Log_alg_part, "Calculate the Stiefel logarithm at base_point U0 and target_point U1 using 2 iterations");
    m.def("metric", &metric, "Calculate the metric at base point U between two tangent vectors Delta1 and Delta2");
    m.def("approx_dist", &approx_dist, "Calculate the approximate distance between two points U0 and U1");
    m.def("dist", &dist, "Calculate the distance between two points U0 and U1");
    m.def("same_atom_pairs", &same_atom_pairs, "Find all pairs of atoms that are the same in a list of atoms");
    m.def("permute", &permute, "Generate a random permutation of the indices of a list of atoms");
    m.def("OT_permutation_reflection", &OT_permutation_reflection, "Solve the optimal transport problem over permutations and reflections");
    m.def("general_exp", &general_exp, "Calculate the Stiefel exponential at base_point U0 and tangent_vector Delta for metric parameter beta");
    m.def("general_log", &general_log, "Calculate the Stiefel logarithm at base_point U0 and target_point U1 for metric parameter beta");
}

// define a main
// int main() {
//     MatrixXf matA(2, 2);
//     matA << 1, 2, 3, 4;
//     MatrixXf matB(4, 4);
//     matB << matA, matA/20, matA/10, matA;
//     std::cout << matB << std::endl;
//     return 0;
// }
//     MatrixXf U0(23, 4);
//     U0 << -0.06724829424033818,-0.1014910915763611,-0.2360025912304281,0.09084848806824102,
// -0.2735913414374755,0.1204826674930274,0.1259122826249381,0.09084850931376724,
// 0.20068614400644547,0.0074122963032389605,-0.35427304875755883,0.09084846001668488,
// -0.024329851287515986,0.37670570172205564,0.1024181703000265,0.09084847961508205,
// -0.1638498849095765,0.08149609058208046,0.025890219215442926,0.0908485146486155,
// 0.27600475466695273,0.215277038992369,-0.050991937464804314,0.09084845387011926,
// 0.14948074289232702,0.21999276818561353,0.3944883500280732,0.09084851835916342,
// -0.29751319997835546,0.005925894498025656,0.19817778102145972,0.0908484687278863,
// -0.05091467242729254,0.26928371546887536,-0.006380127578501142,0.0908485139562605,
// -0.5113283400522787,-0.08610818262894937,-0.20655149711139245,0.09084849874241714,
// -0.13073941035826292,-0.010244731887672498,0.342496618378927,0.09084848293903354,
// 0.09376213000724012,0.34878910957618214,0.022022085030069652,0.09084850462420921,
// 0.07576958434392107,-0.11448805560212885,-0.34177468854631554,0.09084846924123556,
// -0.04321323559666981,-0.052257307152356214,-0.026913415180891714,0.09084850337155957,
// -0.33586870341786557,0.24501075376854028,-0.1325308818415373,0.3134843424143523,
// -0.2582230273989022,-0.03112817671416712,-0.22881060088823524,0.3134843544113156,
// 0.12566593915221452,-0.06278378712922186,0.13778092581799165,0.31348425430142624,
// -0.03810583093074353,-0.1126650460100251,0.11900879068277158,0.31348430171548114,
// 0.2265774557087624,-0.3921925931448,-0.13625331902890758,0.31348433951434307,
// 0.03811186720746344,-0.2945956979677468,0.22741977252721773,0.3134843206127119,
// 0.1426892772822385,0.3952561464779913,-0.2815283439074704,0.31348428671364775,
// 0.007764104912487468,-0.1871849494436828,0.2470348461849607,0.31348429985512943,
// 0.31367463860602246,0.06911147079282062,0.05120624470906347,0.3134843449938508;
//     MatrixXf U1(23, 4);
//     U1 << 0.09159499406814575,-0.17429125308990479,-0.1367247849702835,0.09084849059581757,
// -0.03952117636799812,0.09489058703184128,-0.34718626737594604,0.09084849059581757,
// -0.14079710841178894,0.16161563992500305,-0.06366092711687088,0.09084849059581757,
// -0.10457654297351837,0.12029916793107986,0.3052689731121063,0.09084849059581757,
// -0.16035033762454987,-0.04935107380151749,-0.19766448438167572,0.09084849059581757,
// -0.15184155106544495,-0.08040229231119156,0.21226662397384644,0.09084849059581757,
// -0.02409929595887661,-0.23044715821743011,0.09468390047550201,0.09084849059581757,
// 0.0838346928358078,-0.12899449467658997,0.25986209511756897,0.09084849059581757,
// 0.04262065514922142,0.19362646341323853,0.00046077961451373994,0.09084849059581757,
// 0.024379219859838486,0.09294819086790085,0.31124451756477356,0.09084849059581757,
// 0.07867748290300369,0.0208489540964365,-0.3359329402446747,0.09084849059581757,
// 0.15521211922168732,0.04820804297924042,0.2881443500518799,0.09084849059581757,
// 0.18909701704978943,-0.013964480720460415,-0.061457760632038116,0.09084849059581757,
// 0.1654283106327057,0.13698551058769226,-0.05260910466313362,0.09084849059581757,
// 0.22269770503044128,-0.3906296491622925,0.06925439834594727,0.3134842813014984,
// -0.18941570818424225,-0.20388135313987732,-0.15530577301979065,0.3134842813014984,
// -0.13683244585990906,0.2416379749774933,-0.33458212018013,0.3134842813014984,
// 0.09572320431470871,0.35376328229904175,0.1876610517501831,0.3134842813014984,
// -0.4447785019874573,-0.12129047513008118,0.03837955743074417,0.3134842813014984,
// 0.2783035635948181,0.052786264568567276,-0.26560139656066895,0.3134842813014984,
// -0.3768254518508911,0.33301010727882385,0.1949654221534729,0.3134842813014984,
// -0.03430473804473877,-0.4890815019607544,0.07871050387620926,0.3134842813014984,
// 0.52467280626297,0.16805151104927063,0.10633150488138199,0.3134842813014984;
//     std::vector<long> atoms = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6};
//     std::cout << "hi" << std::endl;
//     std::cout << U0 << std::endl;
//     std::cout << U1 << std::endl;
//     // std::cout << atoms << std::endl;

//     MatrixXf U = OT_permutation(atoms, U0, U1, 5, 100);
//     std::cout << U << std::endl;
//     std::cout << approx_dist(U1, U0) << std::endl;
//     std::cout << approx_dist(U1, U) << std::endl;
//     return 0;
// }
