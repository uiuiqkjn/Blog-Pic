# 将.pth权重转换为适用于yolo的.pt权重

**文章目录**

[前言](#index1)

二、[.pth与.pt格式的差异](#index2)

三、[.pth转.pt](#index3)

## <span id='index1'>前言</span>
在使用其他特征提取网络替换yolov8的backbone时，需要使用兼容的预训练模型，而这些网络的Pretrained models通常以.pth格式保存，与yolo所需的.pt格式存在差异，无法直接使用。

## <span id='index2'>一、 .pth与.pt格式的差异</span>
pt格式可以保存整个PyTorch模型，包括模型结构、模型参数以及优化器状态等信息。

pth格式只保存了模型参数，没有保存模型结构和其他相关信息。

## <span id='index3'>二、 .pth转.pt</span>
首先定义模型保存的路径
```{python}
simplified_model_path = '/path/model.pth' 
complete_model_path = '/path/yolov8n.pt'  
```

打印当前键值对，供参考
```{python}
print("Simplified model keys:", simplified_checkpoint.keys())
print("Complete model keys:", complete_checkpoint.keys())
```

输出
```{python}
Simplified model keys: dict_keys(['meta', 'state_dict', 'optimizer'])
Complete model keys: dict_keys(['date', 'version', 'license', 'docs', 'epoch', 'best_fitness', 'model', 'ema', 'updates', 'optimizer', 'train_args'])
```

补全简化模型中的缺失部分，并初始化缺失的训练元数据
```{python}
for key in complete_checkpoint.keys():
    if key not in simplified_checkpoint:
        simplified_checkpoint[key] = complete_checkpoint[key]
        print(f"Added missing key: {key}")

if 'model' not in simplified_checkpoint:
    simplified_checkpoint['model'] = complete_checkpoint['model']
    print("Added model structure.")

# 初始化缺失的训练元数据
if 'epoch' not in simplified_checkpoint:
    simplified_checkpoint['epoch'] = 0  # 初始化训练轮次为0或其他适当的值
    print("Initialized epoch to 0.")

if 'best_fitness' not in simplified_checkpoint:
    simplified_checkpoint['best_fitness'] = None  # 设置默认值
    print("Initialized best_fitness to None.")

if 'ema' not in simplified_checkpoint:
    simplified_checkpoint['ema'] = complete_checkpoint.get('ema', None)  # 使用完整模型的EMA状态，如果不存在则设置为None
    print("Added EMA state.")

if 'updates' not in simplified_checkpoint:
    simplified_checkpoint['updates'] = complete_checkpoint.get('updates', None)  # 如果有的话，添加更新状态
    print("Added updates state.")

if 'train_args' not in simplified_checkpoint:
    simplified_checkpoint['train_args'] = complete_checkpoint.get('train_args', None)  # 如果有的话，添加训练参数
    print("Added train_args.")
```

保存补全后的模型
```{python}
updated_model_path = 'path_to_updated_model.pth'  
torch.save(simplified_checkpoint, updated_model_path)
```