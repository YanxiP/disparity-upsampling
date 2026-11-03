# 利用像素分类改善视差图放大效果

## 项目说明

这是我在学习《图像处理与分析》过程中实践的项目。我以 Upsampling-Bilateral-Filter 项目为基础，学习并改进借助高分辨率 RGB 图像将低分辨率视差图放大到相同尺寸的方法。原项目先对视差图插值放大，再用 RGB 图像引导联合双边滤波（JBU），根据邻近像素的距离和颜色差异决定其视差值在加权平均中的占比。

然而，颜色变化可以帮助保留轮廓，但是物体表面的花纹和阴影也会造成颜色变化，因此它们不一定对应真实的视差边界，原始结果在一些物体边界处还会出现视差混合的问题。为改善这些问题，本项目使用像素分类联合双边滤波（PCJBF）方法，先得到原始 JBU 结果，再分别检测视差边缘和颜色边缘，将像素分为四类，最后按类别从邻近的有效视差中调整结果。

项目中包含了 python 代码以更便捷地进行批量对比实验，脚本可为两种方法批量运行九组场景和三种放大倍率，计算 BPR、RMSE，并保存对比图片。下面说明如何使用这些脚本。

## 1. 数据集

使用 [Middlebury Stereo Dataset](https://vision.middlebury.edu/stereo/data/) 的 2005、2006 年 FullSize 两视图数据，共九组场景：

- 2005：Books、Dolls、Laundry、Moebius、Reindeer。
- 2006：Baby1、Bowling1、Cloth1、Wood1；实验结果中分别简称 Baby、Bowling、Cloth、Wood。

每组使用同一视角的 `view1.png` 作为高分辨率 RGB 引导图，`disp1.png` 作为视差真值。视差值表示像素位移，并非以米计的深度；数值为 0 的像素不参与评价。我将图像对齐裁剪到宽高可被 8 整除，再用最近邻下采样生成 ×2、×4、×8 的低分辨率输入。

## 2. 安装依赖

在本目录运行以下命令（需要 Python 3.10+）：

```powershell
python -m pip install -r requirements.txt
```

## 3. 下载数据

```powershell
python scripts/download_middlebury.py
```

脚本从 Middlebury 官方地址下载九组数据，并在 `results/dataset/middlebury/` 保存 ZIP、所需图像和来源记录。若自动下载失败，可从官方数据页手动取得相应场景的 `SCENE-2views.zip`，放入该目录后重新运行脚本。

## 4. 运行批量对比实验

```powershell
python scripts/run_experiment.py --factors 2 4 8
```

程序对九组场景分别运行原始 JBU 和改进 PCJBF。逐场景的 BPR、RMSE 写入 `results/raw_metrics.csv`；数值结果与对比图片分别保存在 `results/numeric/` 和 `results/visualizations/`。已有 CSV 记录的场景会跳过。

## 5. 汇总结果

```powershell
python scripts/make_report.py
```

这一步生成平均指标、统计图和实验记录，可查看 `results/average_metrics.csv`、`results/charts/` 和 `results/实验记录.md`。
"# disparity-upsampling-with-pixel-classification" 
