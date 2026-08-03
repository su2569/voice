import torch
import genie_tts as genie
from pathlib import Path

# ---------- 配置 ----------
MODEL_DIR = Path("./GenieData/model")
PTH_PATH = MODEL_DIR / "Elysia_e16_s8720.pth"  # 使用你的 .pth 文件
CKPT_PATH = MODEL_DIR / "Elysia-e15.ckpt"      # 使用你的 .ckpt 文件
OUTPUT_DIR = MODEL_DIR / "onnx"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 转换函数 ----------
def convert_to_onnx(pth_path, ckpt_path, output_dir):
    """将 PyTorch 模型转换为 ONNX"""
    print(f"🚀 开始转换模型...")
    print(f"  - .pth 文件: {pth_path}")
    print(f"  - .ckpt 文件: {ckpt_path}")
    print(f"  - 输出目录: {output_dir}")
    
    try:
        # 调用 genie_tts 的转换函数（不需要 language 参数）
        genie.convert_to_onnx(
            torch_pth_path=str(pth_path),
            torch_ckpt_path=str(ckpt_path),
            output_dir=str(output_dir)
        )
        print(f"✅ 转换成功！")
        print(f"📁 输出文件位于: {output_dir}")
        
        # 列出生成的文件
        for f in output_dir.glob("*"):
            print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
            
    except TypeError as e:
        # 如果仍然报参数错误，尝试不带参数名调用
        print(f"⚠️ 关键字参数失败，尝试位置参数...")
        try:
            genie.convert_to_onnx(
                str(pth_path),
                str(ckpt_path),
                str(output_dir)
            )
            print(f"✅ 转换成功（位置参数）！")
        except Exception as e2:
            print(f"❌ 位置参数也失败: {e2}")
            raise
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        raise

# ---------- 执行转换 ----------
if __name__ == "__main__":
    # 检查文件是否存在
    if not PTH_PATH.exists():
        print(f"❌ 找不到 .pth 文件: {PTH_PATH}")
        print("请确认文件路径是否正确，或修改 PTH_PATH 变量")
        exit(1)
    
    if not CKPT_PATH.exists():
        print(f"❌ 找不到 .ckpt 文件: {CKPT_PATH}")
        print("请确认文件路径是否正确，或修改 CKPT_PATH 变量")
        exit(1)
    
    convert_to_onnx(
        pth_path=PTH_PATH,
        ckpt_path=CKPT_PATH,
        output_dir=OUTPUT_DIR
    )
