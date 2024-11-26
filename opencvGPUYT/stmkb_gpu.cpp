#include "stmkb_gpu.hpp"

STKMBGpu::STKMBGpu(const cv::cuda::GpuMat& firstFrame, float q, float r,
    int maskSize, int bilateralD, double bilateralSigma)
    : q_(q), maskSize_(maskSize), bilateralD_(bilateralD),
    bilateralSigma_(bilateralSigma) {
    cv::Size size = firstFrame.size();
    xPredicted_.create(size, CV_32F);
    pPredicted_.create(size, CV_32F);
    xCorrection_.create(size, CV_32F);
    pCorrection_.create(size, CV_32F);
    k_.create(size, CV_32F);
    r_.create(size, CV_32F);
    blurred_.create(size, CV_32F);
    delta_.create(size, CV_32F);
    aux_.create(size, CV_32F);
    bfFrame_.create(size, CV_32F);
    floatFrame_.create(size, CV_32F);
    temp1_.create(size, CV_32F);
    temp2_.create(size, CV_32F);
    prevCorrection_.create(size, CV_32F);

    xPredicted_.setTo(cv::Scalar(1.0f));
    pPredicted_.setTo(cv::Scalar(1.0f));
    pCorrection_.setTo(cv::Scalar(1.0f));
    k_.setTo(cv::Scalar(0.5f));
    blurred_.setTo(cv::Scalar(0.0f));
    r_.setTo(cv::Scalar(r));

    firstFrame.convertTo(xCorrection_, CV_32F);
    xCorrection_.copyTo(prevCorrection_);

    avgFilter_ =
        cv::cuda::createBoxFilter(CV_32F, CV_32F, cv::Size(maskSize_, maskSize_));
}

void STKMBGpu::process(const cv::cuda::GpuMat& input,
    cv::cuda::GpuMat& output) {

    // z(k)
    input.convertTo(floatFrame_, CV_32F);

    // Pre-filtering
    avgFilter_->apply(floatFrame_, aux_);

    // Apply bilateral filter BF(z(k), d, σ)
    cv::cuda::bilateralFilter(floatFrame_, temp1_, bilateralD_, bilateralSigma_,
        bilateralSigma_ / 2.0);
    cv::cuda::bilateralFilter(temp1_, bfFrame_, bilateralD_, bilateralSigma_,
        bilateralSigma_ / 2.0);

    // Cδ = blurred(k-1) - blurred(k)
    cv::cuda::subtract(blurred_, aux_, delta_);
    aux_.copyTo(blurred_);

    // Update measurement noise R(k) = 1 + R(k-1)/(1 + K(k-1))
    cv::cuda::GpuMat ones(r_.size(), CV_32F, cv::Scalar(1.0f));
    cv::cuda::add(ones, k_, temp1_);
    cv::cuda::divide(r_, temp1_, r_);
    cv::cuda::add(ones, r_, r_);

    // Kalman prediction
    xCorrection_.copyTo(xPredicted_);

    // P(k|k-1) = P(k-1|k-1) + q·δ^2
    cv::cuda::multiply(delta_, delta_, temp1_);
    cv::cuda::multiply(temp1_, cv::Scalar(q_), temp1_);
    cv::cuda::add(pCorrection_, temp1_, pPredicted_);

    // Update Kalman gain K(k) = P(k|k-1)/(P(k|k-1) + R(k))
    cv::cuda::add(pPredicted_, r_, temp1_);
    cv::cuda::divide(pPredicted_, temp1_, k_);

    // State correction
    cv::cuda::subtract(floatFrame_, xPredicted_, temp1_);
    cv::cuda::multiply(k_, temp1_, temp1_);
    cv::cuda::add(xPredicted_, temp1_, temp1_);

    cv::cuda::subtract(ones, k_, temp2_);
    cv::cuda::multiply(temp2_, temp1_, temp1_);
    cv::cuda::multiply(k_, bfFrame_, temp2_);
    cv::cuda::add(temp1_, temp2_, xCorrection_);

    // Update error covariance
    cv::cuda::subtract(ones, k_, temp1_);
    cv::cuda::multiply(pPredicted_, temp1_, pCorrection_);

    cv::cuda::addWeighted(xCorrection_, 0.7, prevCorrection_, 0.3, 0.0,
        xCorrection_);
    xCorrection_.copyTo(prevCorrection_);

    xCorrection_.convertTo(output, CV_8UC1);
}