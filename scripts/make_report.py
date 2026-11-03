"""Create reproducible aggregates, charts, case selections, and Chinese report."""
import csv
import json
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from experiment.paths import RESULTS_ROOT
RESULTS = RESULTS_ROOT
with (RESULTS / 'raw_metrics.csv').open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
for row in rows:
    for key in ('factor', 'jbu_bpr', 'pcjbf_bpr', 'jbu_rmse', 'pcjbf_rmse', 'jbu_seconds', 'pcjbf_seconds'):
        row[key] = float(row[key])

scenes = ['Baby', 'Books', 'Bowling', 'Cloth', 'Dolls', 'Laundry', 'Moebius', 'Reindeer', 'Wood']
factors = sorted({int(r['factor']) for r in rows})
averages = []
for factor in factors:
    subset = [r for r in rows if r['factor'] == factor]
    avg = {'factor': factor, 'scenes': len(subset)}
    for key in ('jbu_bpr', 'pcjbf_bpr', 'jbu_rmse', 'pcjbf_rmse', 'jbu_seconds', 'pcjbf_seconds'):
        avg[key] = float(np.mean([r[key] for r in subset]))
    avg['bpr_reduction_pp'] = avg['jbu_bpr'] - avg['pcjbf_bpr']
    avg['rmse_reduction'] = avg['jbu_rmse'] - avg['pcjbf_rmse']
    averages.append(avg)
with (RESULTS / 'average_metrics.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, averages[0].keys()); writer.writeheader(); writer.writerows(averages)

charts = RESULTS / 'charts'
charts.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
for factor in factors:
    subset = sorted([r for r in rows if r['factor'] == factor], key=lambda r: scenes.index(r['scene']))
    x = np.arange(len(subset)); width = .38
    for metric, label in (('bpr', 'BPR (%)'), ('rmse', 'RMSE (disparity pixels)')):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(x - width/2, [r['jbu_'+metric] for r in subset], width, label='Original JBU', color='#4c78a8')
        ax.bar(x + width/2, [r['pcjbf_'+metric] for r in subset], width, label='PCJBF', color='#f58518')
        ax.set_xticks(x, [r['scene'] for r in subset], rotation=25, ha='right')
        ax.set_ylabel(label); ax.set_title(f'Original JBU vs PCJBF | x{factor}')
        ax.legend(); fig.tight_layout()
        fig.savefig(charts / f'{metric}_x{factor}.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width/2, [r['jbu_seconds'] for r in subset], width, label='Original JBU', color='#4c78a8')
    ax.bar(x + width/2, [r['pcjbf_seconds'] for r in subset], width, label='PCJBF', color='#f58518')
    ax.set_xticks(x, [r['scene'] for r in subset], rotation=25, ha='right')
    ax.set_ylabel('Elapsed seconds'); ax.set_title(f'Runtime | x{factor}'); ax.legend(); fig.tight_layout()
    fig.savefig(charts / f'runtime_x{factor}.png', dpi=180); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, metric, ylabel in zip(axes, ('bpr', 'rmse'), ('BPR (%)', 'RMSE (px)')):
    ax.plot(factors, [a['jbu_'+metric] for a in averages], 'o-', label='Original JBU', color='#4c78a8')
    ax.plot(factors, [a['pcjbf_'+metric] for a in averages], 'o-', label='PCJBF', color='#f58518')
    ax.set_xticks(factors); ax.set_xlabel('Upsampling factor'); ax.set_ylabel(ylabel); ax.legend()
fig.savefig(charts / 'scale_trend.png', dpi=180); plt.close(fig)

primary = [r for r in rows if r['factor'] == 4]
ranked = sorted(primary, key=lambda r: r['jbu_bpr'] - r['pcjbf_bpr'], reverse=True)
best = ranked[0]; typical = ranked[len(ranked)//2]
degraded = [r for r in ranked if r['pcjbf_bpr'] > r['jbu_bpr']]
selection = {'best': best['scene'], 'typical': typical['scene'],
             'degradation': [r['scene'] for r in degraded], 'criterion': 'x4 absolute BPR reduction in percentage points'}
(RESULTS / 'case_selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
for label, row, directory in [('best', best, 'best_cases'), ('typical', typical, 'best_cases')]:
    dest = RESULTS / 'visualizations' / directory
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RESULTS / 'visualizations/full_comparison' / f"{row['scene']}_x4_comparison.png",
                 dest / f"{label}_{row['scene']}_x4_comparison.png")
for row in degraded:
    dest = RESULTS / 'visualizations/degradation_cases'; dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RESULTS / 'visualizations/full_comparison' / f"{row['scene']}_x4_comparison.png",
                 dest / f"degradation_{row['scene']}_x4_comparison.png")

def row_md(r):
    return f"| {r['scene']} | ×{int(r['factor'])} | {r['jbu_bpr']:.3f} | {r['pcjbf_bpr']:.3f} | {r['jbu_rmse']:.3f} | {r['pcjbf_rmse']:.3f} |"

table = '\n'.join(row_md(r) for r in sorted(rows, key=lambda r: (r['factor'], scenes.index(r['scene']))))
avg_table = '\n'.join(f"| ×{a['factor']} | {a['jbu_bpr']:.3f} | {a['pcjbf_bpr']:.3f} | {a['bpr_reduction_pp']:.3f} | {a['jbu_rmse']:.3f} | {a['pcjbf_rmse']:.3f} | {a['rmse_reduction']:.3f} |" for a in averages)
decline_text = '、'.join(f"{r['scene']} ×4（BPR +{r['pcjbf_bpr']-r['jbu_bpr']:.3f} 个百分点）" for r in degraded) or '×4 无 BPR 退化场景'
full_figures = '\n'.join(f"- {name}：![{name} ×4 对比](results/visualizations/full_comparison/{name}_x4_comparison.png)" for name in scenes)
report = f'''# 基于边缘优化的深度图超分辨率重建：实验报告

## 一、原始 GitHub 项目、背景与研究意义

原项目 [SaraFattouh/Upsampling-Bilateral-Filter](https://github.com/SaraFattouh/Upsampling-Bilateral-Filter) 的固定提交为 `e4b0b207c3da0b8772b94c89b1138c3fa956159d`。原仓库只有 `src/main.cpp` 作为算法文件，CMake 依赖 OpenCV，可执行程序还弹出窗口并固定进行 16 组双边滤波实验。本文以其上采样函数的真实逻辑作为 Baseline，并集成 Ren 等人在 2018 年发表的 [PCJBF](https://itiis.org/digital-library/21814)。PCJBF 是已有方法，本项目完成移植与对照评估。

原项目的 `joint_bilateral_filter` 在同尺寸图像上使用灰度引导，`upsample` 先通过 OpenCV 双线性 `resize` 扩大 8 位视差，然后滤波。论文式(1)则直接从 LR 视差邻域向 HR 像素进行加权，二者并不相同。为保证“原始算法 vs 改进算法”同源，本文实验中 PCJBF 的初始图始终是原仓库逻辑的输出，未用论文式(1)替换它。原始 C++ 文件完整保留。

## 二、数据集与实验设置

数据均来自 [Middlebury Stereo 2005/2006 官方数据集](https://vision.middlebury.edu/stereo/data/)。2005 年使用 Books、Dolls、Laundry、Moebius、Reindeer；2006 年将论文简写的 Baby、Bowling、Cloth、Wood 分别固定为 Baby1、Bowling1、Cloth1、Wood1。每组用同一场景的 `view1.png` 和 `disp1.png`，原始 FullSize PNG 视差数值按官方说明保持原样；0 是无效像素。图像与视差共同裁剪至宽高可被 8 整除。LR 输入按论文实验段落用最近邻将 HR 真值缩小 ×2/×4/×8，HR RGB 不缩放。评价统一只在 GT > 0 的位置进行。这里是视差像素值，不是米制真实深度。

Baseline 固定使用原仓库有效参数 `sigma_s=1, sigma_r=10`，因其函数采用整数参数和 8 位 [0,255] 灰度，无法将论文的 0.5、0.1 直接代入；窗口由原代码 `rmax=int(2.5*sigma_s)` 得到 5×5。PCJBF 式(7)采用 σs=0.2、σr=0.1；灰度差除以 255，LR 空间距离除以 2。公式(4)的“归一化局部方差”在论文中没有给出分母，本项目定义为 5×5 有效视差方差除以场景有效视差范围的平方，阈值 0.005；公式(5)的 5×5 灰度最大最小差归一化后阈值 0.10。论文 PDF 正文也未给出 BPR 数值阈值，本项目明确采用 `|估计视差-真值视差|>1` 像素为错误，BPR 单位为百分比。因此表中数字是本项目独立实验结果，不应与原论文表格直接逐数对齐。

## 三、原始 JBU 算法与 Baseline 复现

审查发现原仓库的 `bilateral_filter` 是单图双边滤波；`joint_bilateral_filter` 使用引导图像灰度差计算范围权重，使用待处理视差求加权平均；`upsample` 先 `resize` 再联合滤波。原始 SSD/MSE/RMSE 实现把 `CV_64FC1` 的结果按 `uchar` 读取，且 8 位相减会饱和，故不适合量化评价。本项目独立实现有效掩码下的浮点 BPR、RMSE。原仓库没有数据加载模块；命令行用 `imread(..., IMREAD_GRAYSCALE)` 读两张图。

本机初次 CMake 配置因缺少 OpenCV C++ 开发包而在 `find_package(OpenCV)` 中止。之后使用官方 OpenCV 4.10 源码构建依赖，原 `OpenCV_naive_stereo` 目标与新增 `pcjbf_experiment` 目标都成功编译。由于本机路径含汉字、MinGW 链接器无法打开该路径下的静态库，`scripts/build_windows.ps1` 通过临时 ASCII 盘符完成了构建。原程序含固定 16 组无关双边滤波及 `waitKey(0)` 交互界面，批量实验使用 Python OpenCV + Numba 严格照原 `upsample/joint_bilateral_filter` 控制流执行。新增原生目标在 Baby ×2、Wood ×4、Books ×8 的逐像素验证中，JBU、PCJBF 与分类图均与 Python 结果完全一致（差异像素数均为 0），验证记录见 `validation/native_validation.json`。基线在 Baby ×2 首先运行并独立保存 `jbu.npy`，之后才运行 PCJBF。

## 四、问题与 PCJBF 改进方法

论文公式(4)用 LR 视差局部方差生成 HR 深度边缘图，公式(5)用 HR 灰度局部最大最小差生成颜色边缘图。公式(6)分类为：P1 两种边缘同时存在；P2 仅深度边缘；P3 仅颜色边缘；P4 均无。公式(7)的 P1 在 LR 5×5 候选内，以空间距离、颜色差及候选视差的乘积评分选最大者；P2 在映射 LR 位置的 2×2 四候选内，取与初始 JBU 最接近的有效视差；P3/P4 在 LR 5×5 邻域执行同样的最近候选选择。所有类别在没有有效候选时保留初始 JBU。论文公式(7)只列三种优化情况，P3、P4 共用 case III，并非四套互不相同的公式。

## 五、代码修改内容

原 `src/main.cpp` 未修改。新增原生 C++ 的 `src/source_jbu.cpp`、`src/pcjbf.cpp`、`src/pcjbf_cli.cpp`，分别承载同源 JBU、分类/优化/评价和命令行入口。`src/experiment/dataset.py` 读取官方对齐图像并保留视差数值，且支持 PFM；`algorithms.py` 包含原仓库上采样逻辑移植、边缘检测、四类分类、式(7)精化；`metrics.py` 计算 BPR/RMSE；`visualize.py` 生成统一色标的完整对比、GT 边界自动选取的局部放大、共享误差色标及四类图；`scripts/download_middlebury.py` 下载并记录 SHA256；`scripts/run_experiment.py` 逐场景运行并立即保存 CSV 与 `results/run_manifest.json`（环境版本、参数、数据哈希）；`scripts/make_report.py` 从 CSV 生成本报告和统计图。

## 六、实际实验结果与分析

| 场景 | 倍率 | JBU BPR % | PCJBF BPR % | JBU RMSE | PCJBF RMSE |
|---|---:|---:|---:|---:|---:|
{table}

| 倍率 | JBU 平均 BPR | PCJBF 平均 BPR | 降低百分点 | JBU 平均 RMSE | PCJBF 平均 RMSE | 降低 |
|---|---:|---:|---:|---:|---:|---:|
{avg_table}

九场景简单算术平均（不是按像素汇总）显示 PCJBF 在三个倍率上均降低 BPR 和 RMSE。×4 平均 BPR 降低 {averages[1]['bpr_reduction_pp']:.3f} 个百分点（相对 {100*averages[1]['bpr_reduction_pp']/averages[1]['jbu_bpr']:.2f}%），平均 RMSE 降低 {averages[1]['rmse_reduction']:.3f}（相对 {100*averages[1]['rmse_reduction']/averages[1]['jbu_rmse']:.2f}%）。×4 绝对 BPR 改善最大的是 {best['scene']}（降低 {best['jbu_bpr']-best['pcjbf_bpr']:.3f} 个百分点）；中位改善案例为 {typical['scene']}。退化如实保留：{decline_text}。这提示按式(7)直接选择 LR 候选值，在颜色纹理或边界错位处可能增加超过 1 像素的错误点，即使整体 RMSE 下降。

![×4 BPR 对比](results/charts/bpr_x4.png)

![×4 RMSE 对比](results/charts/rmse_x4.png)

![倍率趋势](results/charts/scale_trend.png)

![×4 运行时间](results/charts/runtime_x4.png)

CSV 运行时间来自 Python 批量实验，且含 Numba 首次调用编译开销；首例时间与后续样例不可直接当作纯算法运行时。原生 C++ 样例用于逐像素等价验证，不混入此时间图。

## 七、案例展示

局部 ROI 由 GT Sobel 梯度的固定窗口和最大值自动选取，选择过程不看任何算法输出。所有 GT/JBU/PCJBF 使用同一场景真值的数值色标。误差两图共用第 99 百分位范围并明确标出截断，原始误差数组另存为 NPY。

### 改善最大案例：{best['scene']} ×4

![最佳案例](results/visualizations/best_cases/best_{best['scene']}_x4_comparison.png)

![最佳案例局部边缘](results/visualizations/edge_zoom/{best['scene']}_x4_edge_zoom.png)

![最佳案例误差图](results/visualizations/error_maps/{best['scene']}_x4_error_comparison.png)

![四类像素图](results/visualizations/classification_maps/{best['scene']}_x4_pixel_classes.png)

### 中位改善案例：{typical['scene']} ×4

![中位案例](results/visualizations/best_cases/typical_{typical['scene']}_x4_comparison.png)

### 退化案例：{degraded[0]['scene'] if degraded else '无'} ×4

{f"![退化案例](results/visualizations/degradation_cases/degradation_{degraded[0]['scene']}_x4_comparison.png)" if degraded else '本次实验无 ×4 退化案例。'}

### 九场景 ×4 完整对比图

{full_figures}

×2、×8 的同类图在 `results/visualizations/` 对应子目录；每个场景及倍率均保存数值矩阵。

## 八、总结

在指定的九组官方场景和统一输入条件下，本项目实现的 PCJBF 分类精化通常改善原仓库逻辑的上采样视差结果，但 Wood ×4 的 BPR 轻微退化。该结论只适用于上述明确的阈值、场景版本、裁剪和评价掩码。论文未给足边缘/BPR 阈值定义，原仓库 JBU 与论文式(1)也不同，因此本实验不能声称为论文的逐数复现。

参考：Ren et al., *Depth Up-Sampling via Pixel-Classifying and Joint Bilateral Filtering*, KSII TIIS 12(7), 2018，DOI: [10.3837/tiis.2018.07.013](https://doi.org/10.3837/tiis.2018.07.013)；[Middlebury 2005/2006 数据说明](https://vision.middlebury.edu/stereo/data/)。
'''
(RESULTS / '实验记录.md').write_text(report.replace('](results/', ']('), encoding='utf-8')
print('Generated report, summary CSV, charts, and selected cases.')
