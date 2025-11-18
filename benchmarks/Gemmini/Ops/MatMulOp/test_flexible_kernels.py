#!/usr/bin/env python3
"""
测试脚本：验证schedule_matmul_flexible函数在不同矩阵尺寸下的运行情况
使用方法：
1. source ~/.venv/exo/bin/activate
2. python test_flexible_kernels.py
"""

import os
import sys
import tempfile
import traceback

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

try:
    from matmul_dynamic import schedule_matmul_flexible
except ImportError as e:
    print(f"❌ Error importing schedule_matmul_flexible: {e}")
    print("Make sure you're in the correct directory and the module exists.")
    sys.exit(1)

def test_kernel_generation(name, nn, mm, kk):
    """
    测试指定尺寸的kernel是否能成功生成
    
    Args:
        name: kernel名称
        nn: 矩阵A的行数
        mm: 矩阵B的列数  
        kk: 矩阵A的列数/矩阵B的行数
    
    Returns:
        tuple: (success, error_message, cpu_kernel, gemmini_kernel)
    """
    try:
        print(f"Testing {name} with shape ({nn}, {mm}, {kk})...")
        
        # 生成kernel
        cpu_kernel, gemmini_kernel = schedule_matmul_flexible(name, nn, mm, kk)
        
        # 简单验证kernel对象存在且有基本属性
        if cpu_kernel is None or gemmini_kernel is None:
            return False, "Generated kernels are None", None, None
            
        print(f"✅ SUCCESS: {name} generated successfully")
        return True, None, cpu_kernel, gemmini_kernel
        
    except Exception as e:
        error_msg = f"Kernel generation failed: {str(e)}"
        print(f"❌ FAILED: {name} - {error_msg}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False, error_msg, None, None

def save_kernel_to_file(name, cpu_kernel, gemmini_kernel, output_dir="generated_kernels"):
    """
    将生成的kernel保存到文件
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存kernel到文件
        with open(os.path.join(output_dir, f"{name}.py"), 'w') as f:
            f.write("from __future__ import annotations\n")
            f.write("from exo.platforms.gemmini import *\n")
            f.write("from exo.stdlib.scheduling import *\n\n")
            f.write(f"# Generated kernels for {name}\n\n")
            f.write(f"cpu_{name} = {repr(cpu_kernel)}\n\n")
            f.write(f"{name} = {repr(gemmini_kernel)}\n")
            
        print(f"📁 Saved {name} to {output_dir}/{name}.py")
        return True
    except Exception as e:
        print(f"⚠️  Failed to save {name}: {e}")
        return False

def main():
    """
    主测试函数
    """
    print("=" * 60)
    print("Testing schedule_matmul_flexible with different shapes")
    print("=" * 60)
    
    # 定义测试用例：(name, NN, MM, KK)
    # 注意：所有维度必须能被16整除
    test_cases = [
        # 小尺寸矩阵
        ("matmul_16_16_16", 16, 16, 16),
        ("matmul_32_32_32", 32, 32, 32),
        ("matmul_48_48_48", 48, 48, 48),
        ("matmul_64_64_64", 64, 64, 64),
        
        # 测试不同的分块行为 (n_tiles < 4)
        ("matmul_32_64_32", 32, 64, 32),  # n_tiles = 2
        ("matmul_48_32_64", 48, 32, 64),  # n_tiles = 3
        
        # 测试大尺寸矩阵 (n_tiles >= 4，触发分块优化)
        ("matmul_64_128_32", 64, 128, 32),   # n_tiles = 4
        ("matmul_80_64_96", 80, 64, 96),     # n_tiles = 5
        ("matmul_128_128_128", 128, 128, 128), # n_tiles = 8
        ("matmul_256_256_256", 256, 256, 256), # n_tiles = 16
        
        # 不同形状的矩阵
        ("matmul_128_64_96", 128, 64, 96),
        ("matmul_256_128_64", 256, 128, 64),
        ("matmul_512_256_128", 512, 256, 128),
        
        # 大尺寸矩阵
        ("matmul_512_512_512", 512, 512, 512),
        
        # 极端形状
        ("matmul_16_512_16", 16, 512, 16),
        ("matmul_512_16_512", 512, 16, 512),
    ]
    
    # 运行测试
    results = []
    total_tests = len(test_cases)
    successful_kernels = []
    
    for i, (name, nn, mm, kk) in enumerate(test_cases, 1):
        print(f"\n[{i}/{total_tests}] ", end="")
        success, error, cpu_kernel, gemmini_kernel = test_kernel_generation(name, nn, mm, kk)
        results.append((name, nn, mm, kk, success, error))
        
        if success:
            successful_kernels.append((name, cpu_kernel, gemmini_kernel))
    
    # 保存成功的kernels
    print(f"\n📁 Saving generated kernels...")
    for name, cpu_kernel, gemmini_kernel in successful_kernels:
        save_kernel_to_file(name, cpu_kernel, gemmini_kernel)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    successful = sum(1 for _, _, _, _, success, _ in results if success)
    failed = total_tests - successful
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total_tests*100:.1f}%")
    
    # 分析分块行为
    print(f"\n📊 Blocking behavior analysis:")
    for name, nn, mm, kk, success, error in results:
        if success:
            n_tiles = nn // 16
            if n_tiles >= 4:
                print(f"  ✅ {name}: n_tiles={n_tiles} -> Uses divide_loop optimization")
            else:
                print(f"  ✅ {name}: n_tiles={n_tiles} -> No divide_loop (small matrix)")
    
    # 显示失败的测试
    if failed > 0:
        print(f"\n❌ Failed tests:")
        for name, nn, mm, kk, success, error in results:
            if not success:
                print(f"  - {name} ({nn}x{mm}x{kk}): {error[:100] if error else 'Unknown error'}...")
    
    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("- Generated kernels are saved in 'generated_kernels/' directory")
    print("- You can use exocc to compile these kernels to C code")
    print("- Example: exocc generated_kernels/matmul_64_64_64.py -o matmul_64_64_64.c --target gemmini")
    print("=" * 60)
    
    # 返回0表示全部成功，否则返回失败数量
    return failed

if __name__ == "__main__":
    sys.exit(main())