#include "DSP/NamModelEngine.h"
#include <cmath>
#include <memory>
#include <string>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

using namespace namparametric::dsp;
namespace {
thread_local std::string error;
struct Playback {
  NamModelEngine engine;
  std::string info;
};
}

API const char* pn_error() { return error.c_str(); }
API void* pn_load(const char* path, double sampleRate, int blockSize, int lite) {
  try {
    auto model = std::make_unique<Playback>();
    if (!model->engine.LoadModel(path, error, lite ? ModelSize::Lite : ModelSize::Full)) return nullptr;
    model->engine.SetControlTransition(ControlTransition::Eco);
    model->engine.Reset(sampleRate, blockSize);
    nlohmann::json controls = nlohmann::json::array();
    const auto& values = model->engine.GetParameterValues();
    const auto& params = model->engine.GetParameterInfos();
    for (size_t i = 0; i < params.size(); ++i) {
      const auto& p = params[i];
      controls.push_back({{"name", p.name}, {"min", p.minValue}, {"max", p.maxValue},
                          {"default", values[i]}, {"positions", p.enumNames}});
    }
    model->info = nlohmann::json({{"controls", controls},
        {"size_selection", model->engine.SupportsSizeSelection()},
        {"sample_rate", model->engine.GetExpectedSampleRate()}}).dump();
    return model.release();
  } catch (const std::exception& e) { error = e.what(); return nullptr; }
}
API const char* pn_info(void* ptr) { return static_cast<Playback*>(ptr)->info.c_str(); }
API void pn_destroy(void* ptr) { delete static_cast<Playback*>(ptr); }
API int pn_parameters(void* ptr, const float* values, int count) {
  return static_cast<Playback*>(ptr)->engine.SetParameterValues(std::span<const float>(values, count));
}
API double pn_input_gain_db(void* ptr, double level, int calibrated) {
  return static_cast<Playback*>(ptr)->engine.GetInputCalibrationGainDb(level, calibrated != 0);
}
API double pn_output_gain_db(void* ptr, double level, int calibrated) {
  return static_cast<Playback*>(ptr)->engine.GetOutputCalibrationGainDb(level, calibrated ? OutputMode::Calibrated : OutputMode::Raw);
}
API int pn_process(void* ptr, float* input, float* output, int frames) {
  try {
    static_cast<Playback*>(ptr)->engine.Process(input, output, frames);
    return 1;
  } catch (const std::exception& e) { error = e.what(); return 0; }
}
