# extrat_images.py
## 运行

+ 按照test（）到event_Completion（）两个函数单独运行，在运行event_Completion（）前需要把test（）输出到rgb文件夹里标识牌出视野的图片全部删除，避免因为匹配点过少配准出错，也确实配不上；也可以不删除，程序也会自己停止

+ 执行event_Completion（）的时候需要用方框手动标记标识牌区域，程序在方框区域进行特征匹配

## extrat_images.py

主要包含两个函数

+ test()
  + 输出三路对准的rgb，ir，event
  + 这里rgb和ir时域对准(微小误差)，event由当前ir时间戳前30ms累计而成

+ event_Completion()
  + 输出补全的completed_event




# test_data文件架构

```
- test_data
	- 1
		- video    # 原始视频数据
		- ir       # 红外数据
			- ir_frame_0001.png
			- ...
		- event	 # 和ir在时域上对准的事件数据
			- event1_frame_0001.png
			- ...
			- ...
		- rgb      # rgb
			- rgb_frame_0001.png
			- ...
		- completed_event # 依据event 补全后的事件数据
			- completed_event_frame_0004.png # 从rgb对应的第四帧开始
			- ...
```

`ir_frame_0004.png`，`rgb_frame_0004.png`和`event1_frame_0004.png`，`completed_event_frame_0004.png`时域对准

