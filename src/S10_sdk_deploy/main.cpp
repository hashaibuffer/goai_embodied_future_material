#include "quadruped_wheel/qw_state_machine.hpp"

#include <stdexcept>
#include <string>

#ifdef USE_SIMULATION
    #define BACKWARD_HAS_DW 1
    #include "backward.hpp"
    namespace backward{
        backward::SignalHandling sh;
    }
#endif

using namespace types;
MotionStateFeedback StateBase::msfb_ = MotionStateFeedback();

namespace {
struct CliOptions {
    std::string controller = "learned";
    std::string model_path = S10_TERRAIN_DEFAULT_MODEL;
};

CliOptions ParseArgs(int argc, char** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if ((arg == "--controller" || arg == "--model-path") && i + 1 >= argc) {
            throw std::invalid_argument("missing value after " + arg);
        }
        if (arg == "--controller") {
            options.controller = argv[++i];
        } else if (arg == "--model-path") {
            options.model_path = argv[++i];
        } else if (arg == "--ros-args") {
            break;
        } else if (arg.rfind("__", 0) == 0) {
            continue;
        } else if (arg == "-r" || arg == "--remap") {
            if (i + 1 < argc) ++i;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    TerrainPolicyRunner::ParseController(options.controller);
    return options;
}
}  // namespace

int main(int argc, char** argv){
    std::cout << "State Machine Start Running" << std::endl;
    CliOptions options;
    try {
        options = ParseArgs(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "rl_deploy: " << error.what() << std::endl;
        return 2;
    }
    rclcpp::init(argc, argv);
    // AutoNav control (default; set S10_AUTONAV_MODE=collect for collect mode)
    std::shared_ptr<StateMachineBase> fsm = std::make_shared<qw::QwStateMachine>(
        RobotName::S10,
        RemoteCommandType::kAutoNav,
        TerrainPolicyRunner::ParseController(options.controller),
        options.model_path);
    // KeyBoard control (fallback)
    // std::shared_ptr<StateMachineBase> fsm = std::make_shared<qw::QwStateMachine>(
    //     RobotName::S10, RemoteCommandType::kKeyBoard,
    //     TerrainPolicyRunner::ParseController(options.controller), options.model_path);
    //Gamepad control
    // std::shared_ptr<StateMachineBase> fsm = std::make_shared<qw::QwStateMachine>(RobotName::S10, RemoteCommandType::kGamepad);
    
    fsm->Start();
    fsm->Run();
    fsm->Stop();

    rclcpp::shutdown();
    return 0;
}
