# 数据文件介绍

## processed_frames


### 各个文件夹介绍
+ event/event2 对应原位置事件数据和位置校准后的事件数据
+ nir 对应红外数据
+ fusion 对应融合后的数据
+ train_blocks/train_blocks2/train_blocks3 对应分块后的用于训练的数据，后者事件数据是位置校准后的事件数据，最后者融合的真值事件为黑色
+ train_data/train_data2/train_data3 对应训练数据，后者事件数据是位置校准后的事件数据,最后者融合的真值事件为黑色
+ train_data4 场景更复杂的新数据，处理的时候不再进行分块
+ affine_data.npz 相机当前状态下用于校准的仿射变换矩阵

## results

### old
+ epoch图片是在仅主模型，事件数据位置校准后进行训练得到的
+ 几个模型
  + model_initial.pth 伪造数据训练的前置模型
  + my_model_weights_first.pth 伪造数据训练的主模型
  + new_my_model_weights_epoch500.pth 全部模型，事件数据尚未校准
  + only_main_model.pth 主模型，事件数据尚未校准
  + only_main_model2.pth 主模型，事件数据校准后
  + new_my_model_weights_epoch2.pth 事件数据校准，loss真值采用边缘扩充mask和output*mask
  + new_my_model_weights_epoch3.pth 事件数据校准，loss真值采用边缘扩充mask和output*mask,事件数据取反
  + new_my_model_weights_epoch4.pth 事件数据校准，事件数据不取反，模型不在进行分块训练
  + new_my_model_weights_epoch5.pth 事件数据校准，事件数据不取反，新数据训练，模型不在进行分块训练
