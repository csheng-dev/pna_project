# Faster-RCNN NMS 使用分析

## 结论

**是的，当前项目在使用 Faster-RCNN 时自动进行了 NMS（非极大值抑制）。**

## NMS 实现位置

### 1. 模型内置 NMS（自动调用链）

torchvision 的 Faster-RCNN 模型在推理时（`model.eval()`）会自动进行 NMS。虽然项目代码中没有显式导入或调用 `RoIHeads.postprocess_detections`，但 NMS 会通过以下调用链自动执行：

#### 调用链详解

当你在项目代码中调用 `model(images)` 时（如 `src/test.py:192` 和 `src/engine.py:237`），会触发以下自动调用链：

```
1. model(images) 
   ↓
2. FasterRCNN.forward(images, targets=None)
   ↓
3. self.roi_heads(features, proposals, images.image_sizes, targets=None)
   ↓
4. RoIHeads.forward(...) 
   ↓ (在推理模式下，self.training = False)
5. self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
   ↓
6. box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)  # NMS在这里执行
```

#### 关键代码位置（torchvision 内部）

**FasterRCNN.forward** (`torchvision/models/detection/faster_rcnn.py`):
```python
def forward(self, images, targets=None):
    # ... 预处理 ...
    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    # ... 后处理 ...
```

**RoIHeads.forward** (`torchvision/models/detection/roi_heads.py`):
```python
def forward(self, features, proposals, image_shapes, targets=None):
    # ... 特征提取和预测 ...
    if self.training:
        # 训练模式：计算损失
        loss_classifier, loss_box_reg = fastrcnn_loss(...)
    else:
        # 推理模式：调用postprocess_detections（包含NMS）
        boxes, scores, labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        # 构建结果字典
        result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i]})
```

**为什么项目代码中没有显式调用？**

- `postprocess_detections` 是 torchvision 库内部的私有方法
- 它通过 PyTorch 的 `__call__` 机制自动调用（当你调用 `model(images)` 时）
- 这是封装设计：用户只需要调用 `model(images)`，NMS 会自动在内部执行
- 类似于你调用 `torch.nn.Linear(x)` 时，不需要手动调用矩阵乘法，框架会自动处理

### 2. NMS 参数

模型的 `roi_heads` 包含以下 NMS 相关参数：

- **`nms_thresh: 0.5`** - NMS 的 IoU 阈值，默认值为 0.5
- **`score_thresh: 0.05`** - 分数阈值，低于此值的检测框会被过滤
- **`detections_per_img: 100`** - 每张图片最多保留的检测框数量

### 3. NMS 执行流程

在 `postprocess_detections` 方法中，NMS 的执行顺序如下：

1. **分数过滤**：移除低分检测框（`scores > self.score_thresh`）
2. **移除小框**：移除过小的检测框（`remove_small_boxes`）
3. **NMS**：使用 `box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)` 进行非极大值抑制
   - 按类别独立进行 NMS
   - IoU 阈值为 `self.nms_thresh`（默认 0.5）
4. **限制数量**：每张图片最多保留 `detections_per_img` 个检测框（默认 100）

## 代码中的使用

### 训练阶段

在 `src/engine.py` 的 `train_one_epoch` 函数中：
```python
model.train()  # 训练模式，不进行NMS
loss_dict = model(images, targets)  # 返回损失，不进行NMS
```

**说明**：训练时 `model.training = True`，`RoIHeads.forward` 会走训练分支，只计算损失，不会调用 `postprocess_detections`。

### 推理阶段

在 `src/test.py` 和 `src/engine.py` 的 `evaluate_map` 函数中：

**src/test.py:192**:
```python
model.eval()  # 设置评估模式
outputs = model(images)  # 这里会触发完整的调用链，自动执行NMS
# outputs 中的 boxes 已经经过 NMS 处理
for patient_id, output in zip(batch_patient_ids, outputs):
    boxes = output["boxes"]  # 已经过NMS的检测框
    scores = output["scores"]  # 对应的分数
```

**src/engine.py:237**:
```python
model.eval()  # 评估模式
outputs = model(images)  # 自动执行NMS
for output, target in zip(outputs, targets):
    pred_boxes = output["boxes"]  # 已经过NMS的检测框
    pred_scores = output["scores"]
```

**关键点**：
- 项目代码中只需要调用 `model(images)`
- 不需要手动调用 `postprocess_detections` 或 `batched_nms`
- NMS 在 torchvision 内部自动执行，对用户透明

## 验证 NMS 参数

可以通过以下代码查看和修改 NMS 参数：

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(weights=None)

# 查看当前NMS参数
print(f"score_thresh: {model.roi_heads.score_thresh}")
print(f"nms_thresh: {model.roi_heads.nms_thresh}")
print(f"detections_per_img: {model.roi_heads.detections_per_img}")

# 修改NMS参数（如果需要）
model.roi_heads.score_thresh = 0.3  # 提高分数阈值
model.roi_heads.nms_thresh = 0.4    # 降低NMS IoU阈值（更严格的NMS）
model.roi_heads.detections_per_img = 200  # 增加每张图片的检测框数量
```

## 项目中的额外过滤

虽然模型已经进行了 NMS，但在以下位置还进行了额外的分数阈值过滤：

### 1. `src/test.py` - 测试推理

```python
# 模型已经进行了NMS，这里只是根据score_threshold进一步过滤
predictions[patient_id] = detections_to_prediction_string(
    boxes, scores, args.score_threshold  # 默认0.5
)
```

### 2. `src/engine.py` - mAP 评估

```python
# 在evaluate_map函数中，模型输出已经经过NMS
# 这里只是根据score_threshold进一步过滤用于mAP计算
keep = pred_scores >= score_threshold
pred_boxes = pred_boxes[keep]
pred_scores = pred_scores[keep]
```

## 总结

1. **NMS 是自动的**：torchvision 的 Faster-RCNN 在推理时自动进行 NMS
   - 通过 `model(images)` 调用触发，无需显式调用 `postprocess_detections`
   - 这是 PyTorch 框架的封装设计，对用户透明

2. **默认参数**：
   - NMS IoU 阈值：0.5 (`model.roi_heads.nms_thresh`)
   - 分数阈值：0.05 (`model.roi_heads.score_thresh`，模型内部)
   - 每张图片最多检测框：100 (`model.roi_heads.detections_per_img`)

3. **项目中的额外处理**：在测试和评估时，还会根据 `score_threshold` 参数（默认 0.5）进行额外的分数过滤
   - 这是模型输出后的二次过滤，不影响模型内部的 NMS 过程

4. **无需手动调用**：
   - 不需要导入 `RoIHeads` 或 `postprocess_detections`
   - 不需要手动调用 `batched_nms`
   - 只需要调用 `model.eval()` 和 `model(images)`，NMS 会自动执行

## 相关文件

- `src/train.py` - 模型构建
- `src/test.py` - 测试推理（使用模型内置NMS）
- `src/engine.py` - 训练和评估（训练时不使用NMS，评估时使用）
- `src/val.py` - 验证脚本（使用模型内置NMS）


