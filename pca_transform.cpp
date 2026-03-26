#include "pca_transform.h"
#include <iostream>

PCA_Transform::PCA_Transform(const std::vector<float>& dataset, std::size_t rows, std::size_t cols, unsigned int newDim)
{
    CalcPCAMtr(dataset, rows, cols, newDim);
}


void PCA_Transform::CalcPCAMtr(const std::vector<float>& dataset, std::size_t rows, std::size_t cols, unsigned int newDim)
{
    Eigen::setNbThreads(4);
    auto vectorPoints = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 
                                                              Eigen::Dynamic, 
                                                              Eigen::RowMajor>>(dataset.data(), rows, cols);
    avgVector = vectorPoints.colwise().mean();                                 //vector [col]
    Eigen::MatrixXf vpCenter = vectorPoints.rowwise() - avgVector.transpose(); //matrix [row][col]
    Eigen::MatrixXf covarMatr = vpCenter.transpose() * vpCenter;               //mtr[col][col]
    covarMatr = covarMatr/rows;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covarMatr);
    eigenValues = solver.eigenvalues();
    auto projMtrx = solver.eigenvectors().rightCols(newDim);                  //mtr[col][newDim]
    pcaMtrx = projMtrx.transpose();                                           //mtr[newDim][col]
    pcaDim = newDim;
}


std::vector<float> PCA_Transform::ApplyPCATransform(const std::vector<float>& inp, size_t rows, size_t cols) const
{
    auto vectorPoints = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 
                                                              Eigen::Dynamic, 
                                                              Eigen::RowMajor>>(inp.data(), rows, cols);
    Eigen::MatrixXf transformRes = pcaMtrx*((vectorPoints.rowwise() - avgVector.transpose()).transpose());
    std::vector<float> res(transformRes.data(), transformRes.data() + transformRes.size());
    return res;
}

std::array<size_t, 2> PCA_Transform::GetPCAMatrSize() const
{
    std::array<size_t, 2> res;
    res[0] = pcaMtrx.rows();
    res[1] = pcaMtrx.cols();
    return res;
}

std::vector<float> PCA_Transform::GetPCAMatr() const
{
    std::vector<float> res(pcaMtrx.rows()*pcaMtrx.cols());
    for (size_t i = 0; i < pcaMtrx.rows(); i++)
        for (size_t j = 0; j < pcaMtrx.cols(); j++)
            res[pcaMtrx.cols()*i + j] = pcaMtrx(i,j);
    return res;
}

std::vector<float> PCA_Transform::GetEigenVals() const
{
    std::vector<float> res(eigenValues.size());
    for (size_t i = 0; i < eigenValues.size(); i++)
        res[i] = eigenValues(i);
    return res;
}

size_t PCA_Transform::GetPCADim() const
{
    return pcaDim;
}
