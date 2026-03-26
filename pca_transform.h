#include <vector>
#include <cstdlib>
#include <Eigen/Dense> // -I/usr/local/include/eigen3


class PCA_Transform
{
    Eigen::VectorXf avgVector;
    Eigen::MatrixXf pcaMtrx;
    Eigen::VectorXf eigenValues;
    size_t pcaDim;

    public:
        PCA_Transform() = default;
        PCA_Transform(const std::vector<float>& dataset, std::size_t rows, std::size_t cols, unsigned int newDim);
        void CalcPCAMtr(const std::vector<float>& dataset, std::size_t rows, std::size_t cols, unsigned int newDim);
        std::vector<float> ApplyPCATransform(const std::vector<float>& inp, size_t rows, size_t cols) const;
        std::vector<float> GetPCAMatr() const;
        std::vector<float> GetEigenVals() const;
        std::array<size_t, 2> GetPCAMatrSize() const;
        size_t GetPCADim() const;
};
