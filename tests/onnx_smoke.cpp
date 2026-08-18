#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: onnx_smoke MODEL.onnx\n";
        return 2;
    }
    try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_smoke");
    Ort::SessionOptions options;
    std::cerr << "loading model\n";
    Ort::Session session(env, argv[1], options);
    std::cerr << "checking counts\n";
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) return 3;

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    if (std::string(input_name.get()) != "obs" ||
        std::string(output_name.get()) != "actions") return 4;

    std::cerr << "checking names and shapes\n";
    const auto input_type = session.GetInputTypeInfo(0);
    const auto output_type = session.GetOutputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetShape() != std::vector<int64_t>({1, 441}) ||
        output_info.GetShape() != std::vector<int64_t>({1, 16})) return 5;

    std::array<float, 441> observation{};
    constexpr std::array<int64_t, 2> shape{1, 441};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(
        memory, observation.data(), observation.size(), shape.data(), shape.size());
    const char* input_names[] = {"obs"};
    const char* output_names[] = {"actions"};
    std::cerr << "running inference\n";
    auto outputs = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    const float* actions = outputs.at(0).GetTensorData<float>();
    for (std::size_t i = 0; i < 16; ++i) {
        if (!std::isfinite(actions[i]) || actions[i] != 0.0F) return 6;
    }
    std::cout << "ONNX signature and zero output verified\n";
    return 0;
    } catch (const std::exception& error) {
        std::cerr << "ONNX smoke failed: " << error.what() << "\n";
        return 10;
    }
}
