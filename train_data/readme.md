## 数据介绍
+ ir：红外
+ complemented_event：补全后的事件数据，清理了无关事件点
+ grayscale：事件rgb
+ deal_grayscale：事件rgb二值化处理
+ deal_grayscale_after：deal_grayscale腐蚀操作得到
+ a1：deal_grayscale_after得到的仅纹理部分的二值图
+ deal_event_img：标识牌仅 纹理 部分的事件数据
+ target_fusion：融合后的数据
+ mask：融合后的数据，其中事件数据部分置为255
+ new_event：融合后仅 纹理 部分的事件


