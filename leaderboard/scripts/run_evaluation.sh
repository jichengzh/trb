#!/bin/bash
# 必须设置CARLA_ROOT，指定CARLA模拟器的根目录
export CARLA_ROOT=/data1/jcz/Bench2Drive/Bench2DriveZoo/carla
# 指定CARLA服务器的启动脚本路径
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
# 将CARLA的Python API路径添加到PYTHONPATH中
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
# 将Leaderboard相关路径添加到PYTHONPATH中
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
# 将Scenario Runner路径添加到PYTHONPATH中
export PYTHONPATH=$PYTHONPATH:scenario_runner
# 设置Scenario Runner的根目录
export SCENARIO_RUNNER_ROOT=scenario_runner

# 设置Leaderboard的根目录
export LEADERBOARD_ROOT=leaderboard
# 设置挑战赛的模式名称，例如SENSORS表示传感器模式
export CHALLENGE_TRACK_CODENAME=SENSORS
# 设置CARLA服务器的端口号
export PORT=$1
# 设置Traffic Manager的端口号
export TM_PORT=$2
# 是否启用调试模式，0表示关闭
export DEBUG_CHALLENGE=0
# 设置评估的重复次数
export REPETITIONS=1 # 多次评估运行
# 是否从上次中断的地方继续
export RESUME=True
# 是否为Bench2Drive模式
export IS_BENCH2DRIVE=$3
# 设置规划器的类型
export PLANNER_TYPE=$9
# 设置使用的GPU编号
export GPU_RANK=${10}

# TCP评估相关参数
# 设置路线文件路径
export ROUTES=$4
# 设置团队的智能体脚本路径
export TEAM_AGENT=$5
# 设置团队智能体的配置文件路径
export TEAM_CONFIG=$6
# 设置评估结果的保存路径
export CHECKPOINT_ENDPOINT=$7
# 设置评估过程中生成数据的保存路径
export SAVE_PATH=$8

CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-rank=${GPU_RANK} \
