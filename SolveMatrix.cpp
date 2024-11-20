#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <lapacke.h>

using namespace std;

Eigen::MatrixXd readCSVToMatrix(const string &filename, int rows, int cols)
{
    Eigen::MatrixXd matrix(rows, cols);
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    for (int i = 0; i < rows; ++i) {
        getline(file, line);
        stringstream ss(line);
        string value;

        for (int j = 0; j < cols; ++j) {
            getline(ss, value, ',');
            matrix(i, j) = stod(value);
        }
    }
    file.close();
    return matrix;
}

Eigen::VectorXd readCSVToVector(const string &filename, int size)
{
    Eigen::VectorXd vector(size);
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    for (int i = 0; i < size; ++i) {
        getline(file, line);
        vector(i) = stod(line);
    }
    file.close();
    return vector;
}

void writeVectorToCSV(const string &filename, const Eigen::VectorXd &vec)
{
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

    file << fixed << setprecision(7);
    for (int i = 0; i < vec.size(); ++i) {
        file << vec(i) << "\n";
    }
    file.close();
}

Eigen::MatrixXd luDecomposition(const Eigen::MatrixXd &A)
{
    int n = A.rows();
    Eigen::MatrixXd LU = A;

    for (int k = 0; k < n; ++k) {
        if (LU(k, k) == 0.0) {
            cerr << "Zero pivot encountered at row " << k << endl;
            exit(EXIT_FAILURE);
        }

        for (int i = k + 1; i < n; ++i) {
            LU(i, k) /= LU(k, k);
            for (int j = k + 1; j < n; ++j) {
                LU(i, j) -= LU(i, k) * LU(k, j);
            }
        }
    }

    return LU;
}

Eigen::VectorXd solveWithLU(const Eigen::MatrixXd &LU, const Eigen::VectorXd &b)
{
    int n = LU.rows();
    Eigen::VectorXd y = b;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            y(i) -= LU(i, j) * y(j);
        }
    }

    Eigen::VectorXd x = y;

    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            x(i) -= LU(i, j) * x(j);
        }
        x(i) /= LU(i, i);
    }

    return x;
}

Eigen::VectorXd solveWithOpenBLAS(const Eigen::MatrixXd &A, const Eigen::VectorXd &b)
{
    int n = A.rows();
    int nrhs = 1;
    int lda = n;
    int ldb = n;
    vector<int> ipiv(n);
    vector<double> A_data(n * n);
    vector<double> b_data(b.data(), b.data() + n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_data[j * n + i] = A(i, j); // Transpose for column-major
        }
    }

    int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A_data.data(), lda, ipiv.data(), b_data.data(), ldb);
    if (info != 0) {
        throw runtime_error("LAPACKE_dgesv failed with info = " + to_string(info));
    }

    return Eigen::Map<Eigen::VectorXd>(b_data.data(), n);
}

void solveAndLog(const string &methodName,
                 const function<Eigen::VectorXd()> &solutionMethod,
                 const Eigen::MatrixXd &A,
                 const Eigen::VectorXd &b,
                 const string &outputFile)
{
    auto start = chrono::high_resolution_clock::now();
    Eigen::VectorXd x = solutionMethod();
    auto end = chrono::high_resolution_clock::now();

    double time_ms = chrono::duration<double, milli>(end - start).count();
    double residual = (A * x - b).norm();

    writeVectorToCSV(outputFile, x);

    cout << methodName << " time: " << time_ms << " ms\n";
    cout << methodName << " Residual norm: " << residual << "\n\n";
}

int main()
{
    string matrixFile = "A.csv";
    string rightPartFile = "b.csv";

    string solutionLUFile = "x_LU.csv";
    string solutionFullPivLUFile = "x_FullPivLU.csv";
    string solutionSVDFile = "x_SVD.csv";
    string solutionOpenBLASFile = "x_OpenBLAS.csv";
    string solutionSimpleFile = "x_Simple.csv";

    int rows = 1000, cols = 1000;
    int vectorSize = 1000;

    Eigen::MatrixXd A = readCSVToMatrix(matrixFile, rows, cols);
    Eigen::VectorXd b = readCSVToVector(rightPartFile, vectorSize);
    Eigen::VectorXd x;

    // Решение через LU-разложение с частичным pivoting
    solveAndLog("LU with partial pivoting", [&]() {
                        Eigen::VectorXd result = A.lu().solve(b);
                        return result;
                }, A, b, solutionLUFile);

    // Решение через LU-разложение с полным pivoting
    solveAndLog("LU with full pivoting", [&]() {
                    Eigen::VectorXd result =  A.fullPivLu().solve(b); 
                    return result;
                }, A, b, solutionFullPivLUFile);

    // Решение через SVD
    solveAndLog("SVD", [&]() {
                    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
                    Eigen::VectorXd result = svd.solve(b);
                    return result;
                }, A, b, solutionSVDFile);

    // Решение через OpenBLAS (dgesv)
    solveAndLog("OpenBLAS (dgesv)", [&]() {
                    Eigen::VectorXd result = solveWithOpenBLAS(A, b); 
                    return result;
                }, A, b, solutionOpenBLASFile);

    // Решение через Simple LU (custom)
    solveAndLog("Simple LU", [&]() { 
                    Eigen::MatrixXd LU = luDecomposition(A);
                    Eigen::VectorXd result = solveWithLU(LU, b);
                    return result;
                }, A, b, solutionSimpleFile);

    return 0;
}
