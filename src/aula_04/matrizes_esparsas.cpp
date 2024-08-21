#include <pkg/eigen-3.4.0/Eigen/Sparse>
#include <pkg/eigen-3.4.0/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::Triplet<double> Triplet;


// Função para calcular a similaridade do cosseno entre duas colunas de uma matriz esparsa
double cosine_similarity(const SparseMatrix &mat, int col1, int col2) {
    auto v1 = mat.col(col1);
    auto v2 = mat.col(col2);

    double dot_product = v1.dot(v2);
    double norm1 = v1.norm();
    double norm2 = v2.norm();

    if (norm1 == 0 || norm2 == 0) return 0.0; // Evita divisão por zero
    return dot_product / (norm1 * norm2);
}

// Função para realizar SVD e redução de dimensionalidade
Eigen::MatrixXd reduce_dimensionality(const Eigen::MatrixXd &mat, int num_components) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto U = svd.matrixU();
    auto S = svd.singularValues();

    // Selecionar os primeiros 'num_components' componentes
    return U.leftCols(num_components) * S.asDiagonal().toDenseMatrix().block(0, 0, num_components, num_components);
}

int main() {
    // Criando uma matriz esparsa
    std::vector<Triplet> triplets;
    triplets.emplace_back(0, 0, 1.0);
    triplets.emplace_back(1, 0, 2.0);
    triplets.emplace_back(0, 1, 3.0);
    triplets.emplace_back(2, 1, 4.0);

    SparseMatrix mat(30, 20);
    mat.setFromTriplets(triplets.begin(), triplets.end());

    // Calcular similaridade de cosseno
    std::cout << "Similaridade de Cosseno entre colunas 0 e 1: " << cosine_similarity(mat, 0, 1) << std::endl;

    // Convertendo matriz esparsa para densa para SVD
    Eigen::MatrixXd denseMat = Eigen::MatrixXd(mat);

    // Reduzindo a dimensionalidade
    Eigen::MatrixXd reducedMat = reduce_dimensionality(denseMat, 1);
    std::cout << "Matriz após redução de dimensionalidade:" << std::endl << reducedMat << std::endl;

    return 0;
}
