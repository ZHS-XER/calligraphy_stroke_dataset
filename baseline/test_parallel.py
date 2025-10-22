#!/usr/bin/env python3
"""
测试不同并行数的处理速度和失败率
"""

import subprocess
import time
import json
from pathlib import Path

def count_stats(dataset_dir="../dataset"):
    """统计失败率"""
    total_runs = 0
    failed_runs = 0
    
    dataset = Path(dataset_dir)
    for char_dir in dataset.iterdir():
        if not char_dir.is_dir():
            continue
        rec_dir = char_dir / "recognition"
        if not rec_dir.exists():
            continue
        
        for json_file in rec_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                for run in data.get('runs', []):
                    total_runs += 1
                    if run['llm_output'] is None:
                        failed_runs += 1
            except:
                pass
    
    return total_runs, failed_runs


def test_workers(workers, test_chars=20):
    """测试指定并行数"""
    print("=" * 60)
    print(f"测试并行数: {workers}")
    print("=" * 60)
    
    # 清理之前的测试文件
    test_output = f"test_w{workers}.json"
    if Path(test_output).exists():
        Path(test_output).unlink()
    
    # 记录开始状态
    before_total, before_failed = count_stats()
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行测试
    cmd = ["uv", "run", "recognize_strokes.py", 
           "--test", str(test_chars), 
           "-w", str(workers),
           "-o", test_output]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 测试运行出错: {e}")
        return
    
    # 记录结束时间
    elapsed = time.time() - start_time
    
    # 记录结束状态
    after_total, after_failed = count_stats()
    
    # 计算增量
    new_total = after_total - before_total
    new_failed = after_failed - before_failed
    
    print(f"\n{'='*60}")
    print(f"📊 测试结果 (并行数: {workers})")
    print(f"{'='*60}")
    print(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.2f} 分钟)")
    
    if new_total > 0:
        print(f"API 调用次数: {new_total}")
        print(f"失败次数: {new_failed}")
        print(f"失败率: {new_failed/new_total*100:.2f}%")
        print(f"平均速度: {elapsed/new_total:.2f} 秒/次")
        print(f"调用速率: {new_total/elapsed:.2f} 次/秒")
    else:
        print("⚠️ 未找到新的测试数据")
    
    print(f"{'='*60}\n")
    
    return {
        'workers': workers,
        'elapsed': elapsed,
        'total_calls': new_total,
        'failed_calls': new_failed,
        'avg_time_per_call': elapsed/new_total if new_total > 0 else 0,
        'calls_per_second': new_total/elapsed if elapsed > 0 else 0,
        'failure_rate': new_failed/new_total*100 if new_total > 0 else 0
    }


def main():
    # 配置参数
    TEST_CHARS = 50  # 每次测试 50 个字符 (约 1250 次 API 调用)
    COOLDOWN_MINUTES = 60  # 测试间隔 60 分钟
    
    print("\n" + "=" * 60)
    print("🧪 测试不同并行数的处理速度和失败率（长时间观察限流）")
    print("=" * 60)
    print(f"每次测试: {TEST_CHARS} 个字符 (约 {TEST_CHARS * 25} 次 API 调用)")
    print(f"测试并行数: 5, 10, 20")
    print(f"测试间隔: {COOLDOWN_MINUTES} 分钟")
    print(f"预计总耗时: 约 {COOLDOWN_MINUTES * 2 / 60:.1f} 小时 (不含测试本身耗时)")
    print(f"目的: 长时间观察高并发下的 API 限流情况")
    print("=" * 60 + "\n")
    
    results = []
    
    for i, workers in enumerate([5, 10, 20], 1):
        print(f"\n{'='*60}")
        print(f"开始第 {i}/3 个测试 (并行数: {workers})")
        print(f"{'='*60}\n")
        
        result = test_workers(workers, test_chars=TEST_CHARS)
        if result:
            results.append(result)
        
        # 测试之间冷却
        if i < 3:  # 最后一个测试不需要等待
            cooldown_seconds = COOLDOWN_MINUTES * 60
            print(f"\n{'='*60}")
            print(f"⏳ API 冷却时间: {COOLDOWN_MINUTES} 分钟")
            print(f"下一个测试 (并行数: {[5, 10, 20][i]}) 将在 {COOLDOWN_MINUTES} 分钟后开始")
            print(f"{'='*60}\n")
            
            # 显示倒计时
            for remaining in range(cooldown_seconds, 0, -60):
                mins = remaining // 60
                print(f"⏰ 剩余等待时间: {mins} 分钟...")
                time.sleep(60)
            
            print("✅ 冷却完成，开始下一个测试\n")
    
    # 输出对比总结
    if len(results) > 0:
        print("\n" + "=" * 60)
        print("📈 对比总结 - 限流情况分析")
        print("=" * 60)
        print(f"{'并行数':<10} {'总时间(秒)':<15} {'速度(秒/次)':<15} {'速率(次/秒)':<15} {'失败率':<10}")
        print("-" * 75)
        for r in results:
            print(f"{r['workers']:<10} {r['elapsed']:<15.1f} "
                  f"{r['avg_time_per_call']:<15.2f} "
                  f"{r['calls_per_second']:<15.2f} "
                  f"{r['failure_rate']:<10.2f}%")
        print("=" * 75)
        
        # 分析限流情况
        print("\n💡 限流分析:")
        if len(results) >= 2:
            # 对比速率提升
            for i in range(1, len(results)):
                prev = results[i-1]
                curr = results[i]
                speedup = curr['calls_per_second'] / prev['calls_per_second']
                worker_ratio = curr['workers'] / prev['workers']
                efficiency = (speedup / worker_ratio) * 100
                
                print(f"\n{prev['workers']} → {curr['workers']} 并发:")
                print(f"  - 理论加速: {worker_ratio:.1f}x")
                print(f"  - 实际加速: {speedup:.2f}x")
                print(f"  - 并发效率: {efficiency:.1f}%")
                print(f"  - 失败率变化: {prev['failure_rate']:.2f}% → {curr['failure_rate']:.2f}%")
                
                if efficiency < 70:
                    print(f"  ⚠️ 效率较低，可能遇到限流或其他瓶颈")
                elif curr['failure_rate'] > prev['failure_rate'] * 1.5:
                    print(f"  ⚠️ 失败率显著上升，可能触发限流")
                else:
                    print(f"  ✅ 并发效率良好")
        
        # 保存结果
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 详细结果已保存到: test_results.json\n")
        
        # 给出建议
        print("=" * 60)
        print("💡 建议:")
        if len(results) > 0:
            best = min(results, key=lambda x: x['avg_time_per_call'])
            print(f"  - 最快速度: 并行数 {best['workers']} ({best['avg_time_per_call']:.2f} 秒/次)")
            
            lowest_fail = min(results, key=lambda x: x['failure_rate'])
            print(f"  - 最低失败率: 并行数 {lowest_fail['workers']} ({lowest_fail['failure_rate']:.2f}%)")
            
            # 综合推荐
            if best['workers'] == lowest_fail['workers']:
                print(f"\n  🎯 推荐使用并行数: {best['workers']} (速度最快且失败率最低)")
            else:
                print(f"\n  ⚖️ 需要在速度和稳定性之间权衡:")
                print(f"     - 追求速度: 使用 {best['workers']}")
                print(f"     - 追求稳定: 使用 {lowest_fail['workers']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
