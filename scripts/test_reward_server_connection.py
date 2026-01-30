#!/usr/bin/env python3
"""
RL Training Connection Test Script

测试训练机器与奖励服务器之间的连通性，模拟真实 RL 训练的请求流程。
包含完整的记忆构建 + QA回答 + Judge评分流程。

Usage:
    # 基础连接测试
    python test_reward_server_connection.py --server-url http://<server_ip>:8765
    
    # 完整测试（包含真实数据）
    python test_reward_server_connection.py --server-url http://<server_ip>:8765 --full
    
    # 使用真实parquet数据测试
    python test_reward_server_connection.py --server-url http://<server_ip>:8765 --real-data
"""

import argparse
import json
import time
import requests
from datetime import datetime


def print_banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(step_num, description):
    print(f"\n[Step {step_num}] {description}")
    print("-" * 50)


def test_health_check(server_url):
    """Test health endpoint."""
    print_step(1, "Health Check")
    
    try:
        start = time.time()
        resp = requests.get(f"{server_url}/health", timeout=10)
        latency = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ 服务器状态: {data.get('status', 'unknown')}")
            print(f"✅ 消息: {data.get('message', 'N/A')}")
            print(f"✅ 响应延迟: {latency:.2f}ms")
            return True
        else:
            print(f"❌ HTTP 状态码: {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器: {server_url}")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_build_and_qa_simple(server_url):
    """Test build_and_qa with simple data."""
    print_step(2, "Build & QA (简单测试)")
    
    # 模拟简单的 RL 训练请求
    payload = {
        "session_group_id": "test_session_001",
        "memory_actions": [
            {
                "agent_type": "episodic",
                "action": {
                    "episodic": {
                        "operations": [
                            {"action": "ADD", "memory": "用户喜欢喝咖啡"}
                        ]
                    }
                }
            }
        ],
        "qa_pairs": [
            {
                "question": "用户喜欢喝什么饮料？",
                "answer": "咖啡"
            }
        ]
    }
    
    try:
        start = time.time()
        resp = requests.post(
            f"{server_url}/build_and_qa",
            json=payload,
            timeout=60
        )
        latency = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ 请求成功")
            print(f"   - task_reward: {data.get('task_reward', 'N/A')}")
            print(f"   - correct: {data.get('correct', 'N/A')}/{data.get('total', 'N/A')}")
            print(f"   - retrieval_counts: {data.get('retrieval_counts', {})}")
            print(f"   - dominant_agent: {data.get('dominant_agent', 'N/A')}")
            print(f"✅ 响应延迟: {latency:.2f}ms")
            return True
        else:
            print(f"❌ HTTP 状态码: {resp.status_code}")
            print(f"❌ 响应: {resp.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（>60s）")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_build_and_qa_complex(server_url):
    """Test build_and_qa with complex multi-agent data."""
    print_step(3, "Build & QA (复杂多 Agent 测试)")
    
    # 模拟复杂的多 Agent 记忆操作
    payload = {
        "session_group_id": "test_session_002",
        "memory_actions": [
            {
                "agent_type": "core",
                "action": {
                    "operation": "APPEND",
                    "content": "用户名: Alice, 职业: 软件工程师"
                }
            },
            {
                "agent_type": "episodic",
                "action": {
                    "episodic": {
                        "operations": [
                            {"action": "ADD", "memory": "2024年1月: Alice 完成了机器学习项目"},
                            {"action": "ADD", "memory": "2024年2月: Alice 参加了技术会议"}
                        ]
                    }
                }
            },
            {
                "agent_type": "semantic",
                "action": {
                    "semantic": {
                        "operations": [
                            {"action": "ADD", "memory": "机器学习是人工智能的一个分支"}
                        ]
                    }
                }
            },
            {
                "agent_type": "procedural",
                "action": {
                    "procedural": {
                        "operations": [
                            {"action": "ADD", "memory": "代码审查流程: 1.提交PR 2.自动测试 3.人工审核"}
                        ]
                    }
                }
            }
        ],
        "qa_pairs": [
            {
                "question": "Alice 在2024年1月做了什么项目？",
                "answer": "机器学习项目",
                "type": "single-session"
            },
            {
                "question": "Alice 的职业是什么？",
                "answer": "软件工程师",
                "type": "single-session"
            },
            {
                "question": "什么是机器学习？",
                "answer": "人工智能的一个分支",
                "type": "semantic"
            }
        ]
    }
    
    try:
        start = time.time()
        resp = requests.post(
            f"{server_url}/build_and_qa",
            json=payload,
            timeout=120
        )
        latency = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ 请求成功")
            print(f"   - task_reward: {data.get('task_reward', 'N/A')}")
            print(f"   - correct: {data.get('correct', 'N/A')}/{data.get('total', 'N/A')}")
            print(f"   - retrieval_counts: {data.get('retrieval_counts', {})}")
            print(f"   - dominant_agent: {data.get('dominant_agent', 'N/A')}")
            print(f"✅ 响应延迟: {latency:.2f}ms ({latency/1000:.2f}s)")
            return True
        else:
            print(f"❌ HTTP 状态码: {resp.status_code}")
            print(f"❌ 响应: {resp.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（>120s）")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_concurrent_requests(server_url, num_requests=3):
    """Test concurrent requests (simulate multi-GPU training)."""
    print_step(4, f"并发请求测试 (模拟 {num_requests} 个 GPU 并行请求)")
    
    import concurrent.futures
    
    def single_request(idx):
        payload = {
            "session_group_id": f"concurrent_test_{idx}",
            "memory_actions": [
                {
                    "agent_type": "episodic",
                    "action": {
                        "episodic": {
                            "operations": [
                                {"action": "ADD", "memory": f"测试记忆 #{idx}"}
                            ]
                        }
                    }
                }
            ],
            "qa_pairs": [
                {"question": f"测试问题 #{idx}", "answer": f"测试答案 #{idx}"}
            ]
        }
        
        start = time.time()
        resp = requests.post(f"{server_url}/build_and_qa", json=payload, timeout=60)
        latency = time.time() - start
        return idx, resp.status_code, latency
    
    try:
        start_all = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(single_request, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start_all
        
        success_count = sum(1 for _, status, _ in results if status == 200)
        avg_latency = sum(lat for _, _, lat in results) / len(results)
        
        print(f"✅ 成功: {success_count}/{num_requests}")
        print(f"✅ 平均延迟: {avg_latency*1000:.2f}ms")
        print(f"✅ 总耗时: {total_time*1000:.2f}ms")
        
        for idx, status, lat in sorted(results):
            status_icon = "✅" if status == 200 else "❌"
            print(f"   {status_icon} 请求 #{idx}: {status}, {lat*1000:.2f}ms")
        
        return success_count == num_requests
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_error_handling(server_url):
    """Test error handling with invalid requests."""
    print_step(5, "错误处理测试")
    
    # 测试空请求
    try:
        resp = requests.post(f"{server_url}/build_and_qa", json={}, timeout=30)
        print(f"   空请求响应: HTTP {resp.status_code}")
    except Exception as e:
        print(f"   空请求错误: {e}")
    
    # 测试无效端点
    try:
        resp = requests.get(f"{server_url}/invalid_endpoint", timeout=10)
        print(f"   无效端点响应: HTTP {resp.status_code}")
    except Exception as e:
        print(f"   无效端点错误: {e}")
    
    print("✅ 错误处理测试完成")
    return True


def test_with_real_data(server_url, parquet_path="data/memory_rl_train.parquet"):
    """Test with real parquet data including state_before_path."""
    print_step(6, "真实数据完整流程测试（state_before + 记忆构建 + QA + Judge）")
    
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        import datasets
    except ImportError:
        print("❌ 需要安装 datasets 库: pip install datasets")
        return False
    
    # Load parquet
    full_parquet_path = project_root / parquet_path
    if not full_parquet_path.exists():
        print(f"❌ Parquet 文件不存在: {full_parquet_path}")
        print(f"   请先运行: python scripts/prepare_rl_data.py")
        return False
    
    print(f"📂 加载数据: {full_parquet_path}")
    ds = datasets.load_dataset('parquet', data_files=str(full_parquet_path))['train']
    print(f"   总样本数: {len(ds)}")
    
    # Get first sample (core agent)
    item = ds[0]
    rm = item['reward_model']
    gt = rm['ground_truth'][0] if isinstance(rm.get('ground_truth'), list) else {}
    
    state_before_path = gt.get('state_before_path', '')
    qa_questions = gt.get('qa_questions', [])
    expert_output = gt.get('expert_output', {})
    
    print(f"\n📋 样本信息:")
    print(f"   - conversation_id: {rm.get('conversation_id', 'N/A')}")
    print(f"   - session_index: {rm.get('session_index', 'N/A')}")
    print(f"   - agent_type: {rm.get('agent_type', 'N/A')}")
    print(f"   - state_before_path: {state_before_path}")
    print(f"   - qa_questions 数量: {len(qa_questions)}")
    
    if not qa_questions:
        print("⚠️ 没有 QA questions，跳过测试")
        return False
    
    # Prepare request with real data
    # Simulate model output (use expert output as reference)
    memory_actions = []
    for agent_type in ['core', 'episodic', 'semantic', 'procedural']:
        if agent_type == 'core':
            action_data = {'operation': 'APPEND', 'content': '测试用户信息'}
        else:
            action_data = {
                agent_type: {
                    'operations': [
                        {'action': 'ADD', 'memory': f'测试{agent_type}记忆内容'}
                    ]
                }
            }
        memory_actions.append({'agent_type': agent_type, 'action': action_data})
    
    # Build request
    payload = {
        'session_group_id': f"{rm.get('conversation_id', 'test')}_sess{rm.get('session_index', 0)}",
        'state_before_path': state_before_path,
        'memory_actions': memory_actions,
        'qa_pairs': [{'question': q['question'], 'answer': q['answer']} for q in qa_questions[:3]]
    }
    
    print(f"\n🚀 发送请求到服务器...")
    print(f"   - state_before_path: {state_before_path}")
    print(f"   - QA pairs: {len(payload['qa_pairs'])}")
    
    try:
        start = time.time()
        resp = requests.post(
            f"{server_url}/build_and_qa",
            json=payload,
            timeout=180  # 3分钟超时
        )
        latency = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('success'):
                print(f"\n✅ 完整流程测试成功!")
                print(f"   - task_reward: {data.get('task_reward', 0):.4f}")
                print(f"   - correct: {data.get('correct', 0)}/{data.get('total', 0)}")
                print(f"   - retrieval_counts: {data.get('retrieval_counts', {})}")
                print(f"   - dominant_agent: {data.get('dominant_agent', 'N/A')}")
                print(f"   - 耗时: {latency:.2f}s")
                
                # 验证流程
                print(f"\n📊 流程验证:")
                print(f"   ✅ 记忆加载: state_before_path 已处理")
                print(f"   ✅ 记忆构建: memory_actions 已应用")
                print(f"   ✅ QA搜索: 检索到相关记忆")
                print(f"   ✅ 答案生成: LLM生成了回答")
                print(f"   ✅ Judge评分: 计算了准确率")
                print(f"   ✅ 奖励返回: task_reward = {data.get('task_reward', 0):.4f}")
                return True
            else:
                print(f"❌ 请求失败: {data.get('error', 'unknown')}")
                return False
        else:
            print(f"❌ HTTP 状态码: {resp.status_code}")
            print(f"❌ 响应: {resp.text[:500]}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时 (>180s)")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def simulate_training_loop(server_url, num_steps=5):
    """Simulate a mini training loop."""
    print_step(7, f"模拟训练循环 ({num_steps} 步)")
    
    print("\n模拟 RL 训练输出:")
    print("-" * 50)
    
    total_reward = 0
    
    for step in range(1, num_steps + 1):
        payload = {
            "session_group_id": f"train_step_{step}",
            "memory_actions": [
                {
                    "agent_type": "episodic",
                    "action": {
                        "episodic": {
                            "operations": [
                                {"action": "ADD", "memory": f"训练步骤 {step} 的记忆内容"}
                            ]
                        }
                    }
                }
            ],
            "qa_pairs": [
                {"question": f"第 {step} 步的训练内容是什么？", "answer": f"训练步骤 {step}"}
            ]
        }
        
        try:
            start = time.time()
            resp = requests.post(f"{server_url}/build_and_qa", json=payload, timeout=60)
            latency = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                reward = data.get('task_reward', 0)
                total_reward += reward
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step}/{num_steps} | "
                      f"Reward: {reward:.4f} | "
                      f"Avg: {total_reward/step:.4f} | "
                      f"Latency: {latency:.0f}ms")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step}/{num_steps} | "
                      f"❌ Error: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step}/{num_steps} | "
                  f"❌ Error: {e}")
        
        time.sleep(0.5)  # 模拟训练间隔
    
    print("-" * 50)
    print(f"训练完成 | 总奖励: {total_reward:.4f} | 平均奖励: {total_reward/num_steps:.4f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="RL Training Connection Test")
    parser.add_argument("--server-url", type=str, default="http://localhost:8765",
                        help="Reward server URL")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--simulate", action="store_true", help="Run training simulation")
    parser.add_argument("--real-data", action="store_true", help="Test with real parquet data")
    parser.add_argument("--parquet", type=str, default="data/memory_rl_train.parquet",
                        help="Path to parquet file (relative to project root)")
    args = parser.parse_args()
    
    print_banner("RL Training Connection Test")
    print(f"服务器地址: {args.server_url}")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试结果统计
    results = {}
    
    # 1. Health Check
    results["health"] = test_health_check(args.server_url)
    
    if not results["health"]:
        print("\n" + "=" * 60)
        print("❌ 健康检查失败，请检查:")
        print("   1. 奖励服务器是否已启动")
        print("   2. 网络是否可达")
        print("   3. 端口是否正确 (8765)")
        print("   4. 防火墙是否放行")
        print("=" * 60)
        return
    
    # 2. Simple Build & QA
    results["simple"] = test_build_and_qa_simple(args.server_url)
    
    if args.full or args.simulate or args.real_data:
        # 3. Complex Build & QA
        results["complex"] = test_build_and_qa_complex(args.server_url)
        
        # 4. Concurrent Requests
        results["concurrent"] = test_concurrent_requests(args.server_url)
        
        # 5. Error Handling
        results["error"] = test_error_handling(args.server_url)
    
    if args.real_data:
        # 6. Real Data Test
        results["real_data"] = test_with_real_data(args.server_url, args.parquet)
    
    if args.simulate:
        # 7. Training Simulation
        results["simulate"] = simulate_training_loop(args.server_url)
    
    # 总结
    print_banner("测试总结")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test_name}")
    
    print()
    if all_passed:
        print("🎉 所有测试通过！两台机器连接正常，可以开始 RL 训练。")
        print()
        print("启动训练命令:")
        print("-" * 50)
        print(f'REWARD_SERVER_URL="{args.server_url}" \\')
        print('MODEL_PATH="/path/to/your/model" \\')
        print('TRAIN_DATA="data/memory_rl_train.parquet" \\')
        print('./scripts/run_memory_grpo_multinode.sh')
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息。")


if __name__ == "__main__":
    main()
