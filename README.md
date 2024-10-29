## Augmenting_Infrared_Texture_Perception

### **data_processing**
该部分主要包含数据处理操作。

**aedat4tomat**：
- 读取 AEDAT4 文件，提取 RGB 图像以及与附近时间戳相关的事件数据。
- 通过事件相机捕获的 RGB 图像进行特征匹配，补全事件数据，依赖前五帧和之前补全的数据。

**deal_data**：
- `eroded_image()`：进行腐蚀操作以产生缩小后的二值图。
- `clean_data()`：清理二值图，仅保留纹理部分。
- `match_buquanevent_deal_grayscale()`：利用纹理部分约束生成对应的事件数据（从事件相机的视角）。

**a**：文件夹重命名操作。

**generate_demo**：生成视频。


### **RIFT-multimodal-image-matching-main**

- 利用红外图像和二值化后的事件 RGB 图像进行特征匹配。
- 使用得到的变换矩阵将处理后的事件数据（`deal_event_img`），在清理掉无关区域后，转换到红外相机视角下（`new_event`），并融合生成目标输出（`target_fusion`）。

红外数据和事件数据的时间戳不一致，这是否会影响处理？

### **train**
**All_model4**：模型架构。
- 讨论之前模型中的注意力机制。

**train**：
- 关于损失函数的细节。
- 当前遇到与最开始相同的问题：在损失很低的情况下，仍然存在黑点。
- 目前输入的事件数据和红外数据似乎不在同一时间戳下。



