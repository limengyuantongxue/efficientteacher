import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import sys
import subprocess
import time
from datetime import datetime

# 添加项目根目录到路径
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))  # efficientteacher目录
sys.path.insert(0, ROOT)

def run_train_once(run_number, total_runs):
    """运行一次train.py"""
    print(f"\n{'='*60}")
    print(f"开始第 {run_number}/{total_runs} 次运行")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # 清理GPU缓存
    try:
        import torch
        torch.cuda.empty_cache()
        print("GPU缓存已清理")
    except:
        pass
    
    start_time = time.time()
    
    # 运行train.py，禁用警告
    cmd = [sys.executable, '-W', 'ignore', os.path.join(ROOT, "train.py")]
    process = subprocess.Popen(cmd, cwd=ROOT)
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    
    success = process.returncode == 0
    
    print(f"\n{'='*60}")
    print(f"第 {run_number}/{total_runs} 次运行 {'成功' if success else '失败'}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行时长: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    print(f"{'='*60}\n")
    
    return success, duration

def main():
    total_runs = 4
    results = []
    
    print(f"准备运行 train.py 共 {total_runs} 次")
    print(f"项目目录: {ROOT}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for i in range(1, total_runs + 1):
        success, duration = run_train_once(i, total_runs)
        results.append({
            'run': i,
            'success': success,
            'duration': duration
        })
        
        # 如果不是最后一次，等待几秒让系统释放资源
        if i < total_runs:
            wait_time = 10
            print(f"等待 {wait_time} 秒后开始下一次运行...\n")
            time.sleep(wait_time)
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"运行总结")
    print(f"{'='*60}")
    print(f"总运行次数: {total_runs}")
    print(f"成功次数: {sum(1 for r in results if r['success'])}")
    print(f"失败次数: {sum(1 for r in results if not r['success'])}")
    print(f"总时长: {sum(r['duration'] for r in results):.2f} 秒 ({sum(r['duration'] for r in results)/60:.2f} 分钟)")
    print(f"\n各次运行详情:")
    for r in results:
        status = "✓ 成功" if r['success'] else "✗ 失败"
        print(f"  第{r['run']}次: {status}, 时长 {r['duration']:.2f}秒")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()