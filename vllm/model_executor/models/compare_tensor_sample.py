import os
import threading
import torch
from typing import Any, Optional, Union

import json
from datetime import datetime

from vllm.logger import init_logger
logger = init_logger(__name__)

_STEP3P5_COMPARE_SAMPLE_SIZE = int(os.environ.get("VLLM_STEP3P5_COMPARE_SAMPLE_SIZE","8"))

_step3p5_compare_local = threading.local()

def step3p5_compare_log(
        tag: str,
        obj: Any,
        *,
        layer_idx: Optional[int] = None,
) -> None:
    '''
    if not _step3p5_compare_is_enabled():
        return
    if not _step3p5_compare_rank_ok():
        return 
    '''

    if str(obj.device) != "npu:2":
        return
    
    call_id = _step3p5_compare_id()
    layer_str = str(layer_idx) if layer_idx is not None else "-"

    if obj is None:
        logger.info("[STEP3P5_CMP] call=% layer=% %=None", call_id, 
                    layer_str, tag)
        return

    if not isinstance(obj, torch.Tensor):
        logger.info("[STEP3P5_CMP] call=% layer=% type=%", call_id, 
                    layer_str, tag, type(obj))
        return
    
    tensor = obj
    tensor_l1n = tensor.to(torch.float).norm(p=1)
    sample = _step3p5_compare_tensor_sample(tensor_l1n)
    logger.info(
        "[STEP3P5_CMP] call=% layer=% % shape=% dtype=% device=% sample=%", 
        call_id, 
        layer_str, 
        tag,
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        sample
    )

def _step3p5_compare_id() -> int:
    return int(getattr(_step3p5_compare_local, "call_id", 0))

def _step3p5_compare_tensor_sample(t: torch.Tensor) -> list[float] :
    if _STEP3P5_COMPARE_SAMPLE_SIZE <= 0:
        return []
    flat = t.detch().flatten()
    if flat.numel() == 0:
        return []
    n = min(_STEP3P5_COMPARE_SAMPLE_SIZE, flat.numel())
    return flat[:n].to(dtype=torch.float32).cpu.tolist()


# class TensorDumper:

#     def __init__(self, save_dir="./tensor_dumps"):
#         self.save_dir = save_dir
#         os.makedirs(save_dir, exist_ok=True)

#     def dump_tensor(self, tensor, name):
#         if tensor.is_cuda:
#             tensor_cpu = tensor.cpu()
#             print(f"Tensor '{name}' from {tensor.device} to CPU")
#         else:
#             tensor_cpu = tensor

#         timestamp = datetime.now.strftime("%Y%m%d_%H%M%S")
#         filename = f"{name}_{timestamp}.pt"
#         filepath = os.path.join(self.save_dir, filename)

#         torch.save(tensor_cpu, filepath)

#         print(f"Tensor 'name' is saved to: {filepath}")
#         print(f"tensor_cpu.shape:{tensor_cpu.shape}")
#         print(f"tensor_cpu.dtype:{tensor_cpu.dtype}")
#         print(f"tensor_cpu.device:{tensor_cpu.device}")

#         if tensor.is_cuda:
#             print(f"tensor.device: {tensor.device}")
        
#         return filepath
    
#     def dump_tensor(self, tensor_dict):
#         timestamp = datetime.now.strftime("%Y%m%d_%H%M%S")
#         save_dir = os.path.join(self.save_dir, f"batch_{timestamp}")
#         os.mkdir(save_dir, exist_ok = True)

#         saved_files = []

#         for name, tensor in tensor_dict.items():
#             # 保存单个Tensor
#             tensor_path = os.path.join(save_dir, f"{name}.pt")

#             # 确保Tensor在CPU上
#             if tensor.is_cuda:
#                 tensor_cpu = tensor.cpu()
#             else:
#                 tensor_cpu = tensor

#             torch.save(tensor_cpu, tensor_path)
#             saved_files.append(tensor_path)

#         print(f"批量保存完成,共保存{len(tensor_dict)}个Tensor")
#         print(f" 保存目录：{save_dir}")
#         print(f" 元数据文件: {meta_path}")

#         return saved_files, meta_path
    
# # ========================================
# # 使用实例：在VLLM推理代码中保存输入Tensor
# # ========================================

# def save_vllm_input_tensors():

#     # 创建保存器
#     dumper = TensorDumper(save_dir="./vllm_input_dumps")

#     # 模拟vLLM的输入数据
#     # 假设这里从vLLM获取的输入Tensor
#     input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]).cuda() #假设在CPU上
#     attention_mask = torch.tensor([[1,1,1,1,1,1]]).cuda
#     position_ids = torch.tensor([[0,1,2,3,4,5]]).cuda

#     # 创建输入字典(模拟vLLM的输入格式)
#     model_inputs = {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "position_ids": position_ids
#     }

#     # 保存所有Tensor
#     saved_files, meta_path = dumper.dump_tensor(model_inputs)

#     # 也可以单独保存某个Tensor
#     # dumper.dump_tensor(input_ids, "input_ids_example")

#     return saved_files, meta_path

# if __name__ == "__main__":
#     # 运行保存示例
#     print("设备1 - 保存Tensor示例")
#     print("=" * 60)

#     saved_files, meta_path = save_vllm_input_tensors()

#     # 打印总结
#     print("\n 保存总结")
#     print(f"Tensor 文件")
#     for file in saved_files:
#         print(f" - {os.path.basename(file)}")
#     print(f"元数据文件：{os.path.basename(meta_path)}")