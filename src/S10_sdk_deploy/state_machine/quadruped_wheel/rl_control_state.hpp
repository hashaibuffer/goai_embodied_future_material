/**
 * @file rl_control_state.hpp
 * @brief rl policy runnning state for quadruped-wheel robot
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once
#include "state_base.h"
#include "policy_runner_base.hpp"
#include "terrain_policy_runner.hpp"
#include "teacher_collect_runner.hpp"
#include "robot_interface.h"
#include "user_command_interface.h"
#include "json.hpp"
#include "basic_function.hpp"

namespace qw {
    class RLControlState : public StateBase {
    private:
        RobotBasicState rbs_[2];
        std::atomic<int> rbs_write_index_{0};
        int getrbsReadIndex() const { return 1 - rbs_write_index_.load(std::memory_order_acquire); }

        std::atomic<int> state_run_cnt_{-1};

        std::shared_ptr<PolicyRunnerBase> policy_ptr_;
        std::shared_ptr<TerrainPolicyRunner> terrain_policy_;
        std::shared_ptr<TeacherCollectRunner> teacher_collect_policy_;

        std::thread run_policy_thread_;
        std::atomic<bool> start_flag_{true};

        float policy_cost_time_ = 1;

        Eigen::MatrixXf acc_rot = Eigen::MatrixXf::Zero(20, 3);
        int acc_rot_count = 0;

        void UpdateRobotObservation() {
            int write_idx = rbs_write_index_.load(std::memory_order_relaxed);
            RobotBasicState& buffer = rbs_[write_idx];

            buffer.base_rpy = ri_ptr_->GetImuRpy();
            buffer.base_rot_mat = RpyToRm(buffer.base_rpy);
            buffer.base_omega = ri_ptr_->GetImuOmega();
            buffer.base_acc = ri_ptr_->GetImuAcc();
            buffer.joint_pos = ri_ptr_->GetJointPosition();
            buffer.joint_vel = ri_ptr_->GetJointVelocity();
            buffer.joint_tau = ri_ptr_->GetJointTorque();

            // 储存
            buffer.flt_base_acc_mat.row(acc_rot_count) = buffer.base_acc.transpose();
            acc_rot_count += 1;
            acc_rot_count = acc_rot_count % 20;

            rbs_write_index_.store(1 - write_idx,  std::memory_order_release);
        }

        void PolicyRunner() {
            int run_cnt_record = -1;
            while (start_flag_.load(std::memory_order_acquire)) {
                const int state_run_cnt = state_run_cnt_.load(std::memory_order_acquire);
                if (state_run_cnt % policy_ptr_->decimation_ == 0 && state_run_cnt != run_cnt_record) {
                    timespec start_timestamp, end_timestamp;
                    clock_gettime(CLOCK_MONOTONIC, &start_timestamp);
                    auto ra = policy_ptr_->getRobotAction(rbs_[getrbsReadIndex()], *(uc_ptr_->GetUserCommand()));
                    
                    MatXf res = ra.ConvertToMat();

                    ri_ptr_->SetJointCommand(res);
                    run_cnt_record = state_run_cnt;
                    clock_gettime(CLOCK_MONOTONIC, &end_timestamp);
                    policy_cost_time_ = (end_timestamp.tv_sec - start_timestamp.tv_sec) * 1e3
                                        + (end_timestamp.tv_nsec - start_timestamp.tv_nsec) / 1e6;

                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

    public:
        RLControlState(
            const RobotName &robot_name,
            const std::string &state_name,
            std::shared_ptr<ControllerData> data_ptr,
            TerrainPolicyRunner::Controller controller,
            const std::string& model_path,
            bool teacher_collect,
            const std::string& teacher_model_path) : StateBase(robot_name, state_name, data_ptr) {
            if (robot_name_ == RobotName::S10) {
                if (teacher_collect) {
                    teacher_collect_policy_ = std::make_shared<TeacherCollectRunner>(
                        teacher_model_path, ri_ptr_->get_node());
                } else {
                    terrain_policy_ = std::make_shared<TerrainPolicyRunner>(
                        model_path, controller, ri_ptr_->get_node());
                }
            }

            policy_ptr_ = teacher_collect ?
                std::static_pointer_cast<PolicyRunnerBase>(teacher_collect_policy_) :
                std::static_pointer_cast<PolicyRunnerBase>(terrain_policy_);
            if (!policy_ptr_) {
                std::cerr << "error policy" << std::endl;
                exit(0);
            }
            policy_ptr_->DisplayPolicyInfo();
        }

        ~RLControlState() {
            start_flag_.store(false, std::memory_order_release);
            if (run_policy_thread_.joinable()) run_policy_thread_.join();
        }

        virtual void OnEnter() {
            state_run_cnt_.store(-1, std::memory_order_release);
            start_flag_.store(true, std::memory_order_release);
            policy_ptr_->OnEnter();
            run_policy_thread_ = std::thread(std::bind(&RLControlState::PolicyRunner, this));
            StateBase::msfb_.UpdateCurrentState(RobotMotionState::RLControlMode);
        };

        virtual void OnExit() {
            start_flag_.store(false, std::memory_order_release);
            if (run_policy_thread_.joinable()) run_policy_thread_.join();
            state_run_cnt_.store(-1, std::memory_order_release);
        }

        virtual void Run() {
            UpdateRobotObservation();
            state_run_cnt_.fetch_add(1, std::memory_order_release);
        }

        virtual bool LoseControlJudge() {
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
            return PostureUnsafeCheck();
        }

        bool PostureUnsafeCheck() {
            // Vec3f rpy = ri_ptr_->GetImuRpy();
            // if(rpy(0) > 30./180*M_PI || rpy(1) > 45./180*M_PI){
            //     std::cout << "posture value: " << 180./M_PI*rpy.transpose() << std::endl;
            //     return true;
            // }
            return false;
        }

        virtual StateName GetNextStateName() {
            if (uc_ptr_->GetUserCommand()->safe_control_mode != 0) 
                return StateName::kJointDamping;
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::LieDown))
                return StateName::kLieDown;
            
            return StateName::kRLControl;
        }
    };
};
