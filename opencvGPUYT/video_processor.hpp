#pragma once
#include "progress_bar.hpp"
#include "stmkb_cpu.hpp"
#include "stmkb_gpu.hpp"
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

class VideoProcessor {
public:
    explicit VideoProcessor(const std::string_view& path, bool isCpu)
        : isCpu_(isCpu), input_path_(path), cap_(path.data()) {
        initializeVideo();
        initializeWriter();
    }

    ~VideoProcessor() { cleanup(); }
    VideoProcessor(const VideoProcessor&) = delete;
    VideoProcessor& operator=(const VideoProcessor&) = delete;
    VideoProcessor(VideoProcessor&&) = delete;
    VideoProcessor& operator=(VideoProcessor&&) = delete;

    // Main function
    void process();

private:
    bool isCpu_;
    std::string input_path_;
    cv::VideoCapture cap_;
    cv::VideoWriter output_;
    std::unique_ptr<STKMBGpu> stkmbGpu_;
    std::unique_ptr<STKMBCpu> stkmbCpu_;
    cv::cuda::GpuMat frameGpu_;  // Input frame on GPU
    cv::cuda::GpuMat outputGpu_; // Output frame on GPU
    int frame_count_ = 0;
    double fps_ = 0.0;
    int frame_width_ = 0;
    int frame_height_ = 0;
    static constexpr const char* OUTPUT_DIR = "results";
    static constexpr const char* OUTPUT_FILE = "results/output.avi";

    // Initialize video capture and properties
    void initializeVideo();

    // Initialize video writer with codec selection
    void initializeWriter();

    // Process a single frame
    void processFrame(const cv::Mat& input);
    void processFrameWithDenoiser(const cv::Mat& frame);  // Added this function

    bool readAndInitializeFirstFrame(cv::Mat& frame);

    // Cleanup resources
    void cleanup();

    // Finalize processing and verify output
    void finalizeProcessing();
    void printVideoProperties() const;
};
