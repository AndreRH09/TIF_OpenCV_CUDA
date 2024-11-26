#pragma once
#include <deque>
#include <opencv2/opencv.hpp>

class STKMBCpu {
public:
	STKMBCpu(const cv::Mat& firstFrame, int historySize = 5);

	cv::Mat processFrame(const cv::Mat& frame);

private:
	cv::Mat blockMatchingWithHistory(const cv::Mat& current);

	cv::Mat calculateWeights(const cv::Mat& motionMeasure);

	// Dimensions and parameters
	int width, height, blockSize;
	int maxHistory_;
	float q_;          // Process noise
	float sigma_c_;    // Weight calculation parameter
	int d_;            // Bilateral filter diameter
	float sigmaValue_; // Bilateral filter sigma

	// Frame history
	std::deque<cv::Mat> pastFrames_;

	// Kalman filter matrices
	cv::Mat xPredicted_;  // Predicted state
	cv::Mat pPredicted_;  // Predicted error covariance
	cv::Mat xCorrection_; // Corrected state
	cv::Mat pCorrection_; // Corrected error covariance
	cv::Mat kalmanGain_;  // Kalman gain
	cv::Mat r_;           // Measurement noise
	cv::Mat blurred_;     // Previous blurred frame

	// Auxiliary matrices
	cv::Mat delta_;
	cv::Mat aux_;
	cv::Mat bfFrame_;
};