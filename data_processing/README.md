# extrat_images.py
## 运行

+ 按照main1（）到main2（）两个函数单独运行，在运行main2（）前需要把main1（）输出到rgb文件夹里标识牌出视野的图片全部删除，避免因为匹配点过少配准出错，也确实配不上

+ 执行main2（）的时候需要用方框手动标记标识牌区域，程序在方框区域进行特征匹配

## extrat_images.py

主要包含两个函数

+ main1()
  + event_nir_fusion——事件和红外在时域上配准到一起，输出到 `event1`和`ir`文件夹下
    + 这里 event1是以红外每帧做参考，取每帧前30 ms的事件作为一帧事件数据
  + save_frame_event——事件和rgb在时域上配准到一起，不涉及红外数据，输出到 `event2`和`rgb`下
    + 这里 event2是以rgb每帧做参考，取每帧前30 ms的事件作为一帧事件数据

+ main2()
  + event_Completion——依据 `event2`和`rgb`补全事件数据，需要在运行完main1（）把标识牌出视野的rgb删除




# test_data文件架构

```
- test_data
	- 1
		- video    # 原始视频数据
		- ir       # 红外数据
			- ir_frame_0001.png
			- ...
		- event1	 # 和ir在时域上对准的事件数据
			- event1_frame_0001.png
			- ...
		- event2   # 和rgb在时域上对准的事件数据
			- event2_frame_0001.png
			- ...
		- rgb      # rgb
			- rgb_frame_0001.png
			- ...
		- completed_event # 依据event2 补全后的事件数据
			- completed_event_frame_0004.png # 从rgb对应的第四帧开始
			- ...
```

`ir_frame_0001.png`和`event1_frame_0001.png`时域对准

`event2_frame_0001.png`和`rgb_frame_0001.png`时域对准

`completed_event_frame_0004.png`和`rgb_frame_0004.png`时域对准
