#pragma once
#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/highgui.hpp"

class STKMBGpu {
private:
    cv::cuda::GpuMat xPredicted_;    // x̂(k|k-1) predicted state
    cv::cuda::GpuMat pPredicted_;    // P(k|k-1) predicted error covariance
    cv::cuda::GpuMat xCorrection_;   // x̂(k|k) corrected state
    cv::cuda::GpuMat pCorrection_;   // P(k|k) corrected error covariance
    cv::cuda::GpuMat k_;             // K(k) Kalman gain
    cv::cuda::GpuMat r_;             // R(k) measurement noise
    cv::cuda::GpuMat blurred_;       // Previous blurred frame for delta
    cv::cuda::GpuMat delta_;         // δ temporal difference
    cv::cuda::GpuMat aux_;           // Auxiliary blur buffer
    cv::cuda::GpuMat bfFrame_;       // Bilateral filter output
    cv::cuda::GpuMat floatFrame_;    // z(k) current measurement
    cv::cuda::GpuMat temp1_, temp2_; // Temporary computation buffers
    cv::cuda::GpuMat prevCorrection_;

    cv::Ptr<cv::cuda::Filter> avgFilter_;

    const float q_;               // Process noise
    const int maskSize_;          // Average filter mask size
    const int bilateralD_;        // Bilateral filter diameter
    const double bilateralSigma_; // Bilateral filter sigma

public:
    STKMBGpu(const cv::cuda::GpuMat& firstFrame, float q = 0.05f, float r = 5.0f,
        int maskSize = 5, int bilateralD = 7, double bilateralSigma = 35.0);

    void process(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output);
};