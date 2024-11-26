#include "stmkb_cpu.hpp"

STKMBCpu::STKMBCpu(const cv::Mat& firstFrame, int historySize)
    : maxHistory_(historySize) {

    height = firstFrame.rows;
    width = firstFrame.cols;
    blockSize = 8;

    cv::Mat firstGray;
    cv::cvtColor(firstFrame, firstGray, cv::COLOR_BGR2GRAY);
    firstGray.convertTo(xCorrection_, CV_32F);

    pastFrames_.push_back(xCorrection_.clone());

    xPredicted_ = cv::Mat::zeros(height, width, CV_32F);
    pPredicted_ = cv::Mat::ones(height, width, CV_32F);
    pCorrection_ = cv::Mat::ones(height, width, CV_32F);
    kalmanGain_ = cv::Mat::ones(height, width, CV_32F) * 0.5f;
    blurred_ = cv::Mat::zeros(height, width, CV_32F);
    r_ = cv::Mat::ones(height, width, CV_32F) * 10.0f;

    q_ = 0.1f;           // Noise
    d_ = 5;              // Bilateral filter diameter
    sigmaValue_ = 25.0f; // Bilateral filter sigma
}

cv::Mat STKMBCpu::processFrame(const cv::Mat& frame) {
    cv::Mat currentGray, currentFloat;
    cv::cvtColor(frame, currentGray, cv::COLOR_BGR2GRAY);
    currentGray.convertTo(currentFloat, CV_32F);

    cv::Mat preFiltered;
    cv::blur(currentFloat, preFiltered, cv::Size(3, 3));

    cv::Mat motionMeasure = blockMatchingWithHistory(preFiltered);

    cv::blur(currentFloat, aux_, cv::Size(3, 3));
    delta_ = blurred_ - aux_;
    aux_.copyTo(blurred_);

    r_ = 1.0f + (r_).mul(1.0f / (1.0f + kalmanGain_));

    // Prediction step
    xPredicted_ = xCorrection_.clone();
    pPredicted_ = pCorrection_ + q_ * delta_.mul(delta_);

    // Update step - Kalman gain
    kalmanGain_ = pPredicted_ / (pPredicted_ + r_);

    cv::bilateralFilter(currentFloat, bfFrame_, d_, sigmaValue_, sigmaValue_);

    // Calculate weights based on motion measure
    cv::Mat weights = calculateWeights(motionMeasure);

    // Final correction step combining Kalman and bilateral results
    xCorrection_ =
        (1.0f - kalmanGain_)
        .mul(xPredicted_ + kalmanGain_.mul(currentFloat - xPredicted_)) +
        kalmanGain_.mul(bfFrame_);
    pCorrection_ = pPredicted_.mul(1.0f - kalmanGain_);

    // Update frame history
    if (pastFrames_.size() >= maxHistory_) {
        pastFrames_.pop_front();
    }
    pastFrames_.push_back(xCorrection_.clone());

    // Convert result to BGR
    cv::Mat result;
    xCorrection_.convertTo(result, CV_8U);
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);

    return result;
}

cv::Mat STKMBCpu::blockMatchingWithHistory(const cv::Mat& current) {
    cv::Mat totalMotion = cv::Mat::zeros(height, width, CV_32F);

    // Compare with each frame in history
    for (const auto& pastFrame : pastFrames_) {
        cv::Mat motionForFrame = cv::Mat::zeros(height, width, CV_32F);

        // Block matching
        for (int y = 0; y < height - blockSize; y += blockSize) {
            for (int x = 0; x < width - blockSize; x += blockSize) {
                cv::Rect block(x, y, blockSize, blockSize);

                // Calculate block difference as described in paper
                cv::Mat currentBlock = current(block);
                cv::Mat pastBlock = pastFrame(block);

                // L2 distance between blocks
                cv::Mat diff;
                cv::subtract(currentBlock, pastBlock, diff);
                cv::multiply(diff, diff, diff);
                float blockDist = cv::sum(diff)[0] / (blockSize * blockSize);

                motionForFrame(block).setTo(blockDist);
            }
        }

        // Accumulate motion information
        totalMotion += motionForFrame;
    }

    // Average motion over all past frames
    totalMotion /= static_cast<float>(pastFrames_.size());

    return totalMotion;
}

cv::Mat STKMBCpu::calculateWeights(const cv::Mat& motionMeasure) {
    cv::Mat weights;
    cv::exp(-motionMeasure.mul(motionMeasure) / (2.0f * sigma_c_ * sigma_c_),
        weights);
    return weights;
}