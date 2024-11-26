#pragma once
#include <opencv2/opencv.hpp>

// Add noise to grayscale image
void addNoiseGray(const cv::Mat& input, cv::Mat& output,
    float noiseStd = 10.0f) {
    // Check if input is empty
    if (input.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    // Convert to grayscale if input is color
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

    // Create noise matrix
    cv::Mat noise = cv::Mat::zeros(gray.size(), CV_32F);
    cv::randn(noise, 0, noiseStd);

    // Convert to float for arithmetic
    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F);

    // Add noise
    grayFloat += noise;

    // Clip values to valid range [0, 255]
    cv::min(cv::max(grayFloat, 0.0f), 255.0f, grayFloat);

    // Convert back to 8-bit
    grayFloat.convertTo(gray, CV_8U);

    // Convert back to color if input was color
    cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);
}