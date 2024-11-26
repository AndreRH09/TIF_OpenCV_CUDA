#include "video_processor.hpp"
#include <filesystem>
#include <iostream>

#include <chrono>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << std::filesystem::path(argv[0]).filename().string()
            << " <video_path>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string video_path{ argv[1] };

    if (!std::filesystem::exists(video_path)) {
        std::cerr << "Error: File '" << video_path << "' does not exist"
            << std::endl;
        return EXIT_FAILURE;
    }

    if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
        std::cout << "No CUDA devices found" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Enabled CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount()
        << std::endl;

    cv::cuda::setDevice(0);
    std::cout << "Chosen device: --------------------> "
        << cv::cuda::DeviceInfo(cv::cuda::getDevice()).name() << std::endl;

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    VideoProcessor processor(video_path, false);
    processor.process();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end -
            begin)
        .count()
        << "[ms]" << std::endl;

    return EXIT_SUCCESS;
}