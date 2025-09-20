import numpy as np

# 确保 'A-1.npy' 文件在您的代码运行的目录中
file_path = 'T-5.npy'

try:
    # 使用 numpy.load() 读取文件
    data = np.load(file_path)

    # 打印一些信息来验证
    print("成功读取文件！")
    print("数组的形状 (Shape):", data.shape)
    print("数组的数据类型 (Data Type):", data.dtype)
    print("\n数组的前5行数据:")
    print(data[:5])

    # --- 新增代码：打印中间部分的数据 ---
    print("\n数组的中间部分数据 (例如，从第500行开始):")
    print(data[500:505])

except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。请检查文件路径是否正确。")
except Exception as e:
    print(f"读取文件时发生错误: {e}")