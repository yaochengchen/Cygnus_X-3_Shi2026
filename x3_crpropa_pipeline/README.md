# X-3：Lucas 2021 周期 MHD + 用户指定扩散系数 + CRPropa

这版流程假设你已经拿到 Lucas et al. 2021 的无量纲 MHD 快照。正式计算采用原始 **10 pc 周期立方盒**，不会改成 4 pc，也不会建立一个巨大的 1 kpc MHD 网格。

主流程为：

1. 读取无量纲 `density`、`magnetic_field` 和可选 `velocity`；
2. 将数组统一为 `(x,y,z)` / `(x,y,z,component)`；
3. 把整个 10 pc 盒缩放到目标平均密度和磁场强度；
4. 在粒子所在的任意坐标对同一个 10 pc 盒做周期查询；
5. CRPropa 从 YAML 直接读取用户给定的扩散系数；
6. 用 `DiffusionSDE` 沿局部 MHD 磁场传播连续注入的质子；
7. 用周期 MHD 密度加权，计算 pp 伽马射线、6° 径向轮廓和到地球的吸收。

## 拿到 MHD 文件后改哪里

先查看文件结构：

```bash
python inspect_mhd_file.py data/your_lucas2021_snapshot.h5
```

然后通常只改 `x3_config.yaml` 中这两处。

第一处是 MHD 文件路径和数据集名称：

```yaml
mhd_input:
  path: data/your_lucas2021_snapshot.h5
  format: auto
  fields:
    density:
      dataset: density
    magnetic:
      dataset: magnetic_field
      vector_axis: 0
    velocity:
      dataset: velocity
      vector_axis: 0
      optional: true
  spatial_axis_order: zyx

  physical_scaling:
    box_size_pc: 10.0
    mean_target_density_cm3: 1.0
    magnetic_normalization: rms_strength
    target_magnetic_uG: 3.0
```

常见数组形状：

- `(3,Nz,Ny,Nx)`：`vector_axis: 0`，`spatial_axis_order: zyx`；
- `(Nz,Ny,Nx,3)`：`vector_axis: -1`，`spatial_axis_order: zyx`；
- 三个分量分开存储：`components: [Bx, By, Bz]`。

HDF5 内部路径可写成 `fields/density`、`snapshot/B`；`.npz` 使用数组名称。真实文件无需改名或人工转存。

第二处是 CRPropa 真正使用的扩散系数：

```yaml
transport:
  user_diffusion:
    d0_cm2_s_at_reference: 3.0e29
    reference_energy_pev: 1.0
    alpha: 0.3333333333333333
    sde_epsilon: 0.1
```

正式运行使用

\[
D_\parallel(E)=D_0\left(\frac{E}{E_0}\right)^{1/3},
\qquad D_\perp=\epsilon D_\parallel.
\]

默认值是 \(D_0=3\times10^{29}\,\mathrm{cm^2\,s^{-1}}\) at 1 PeV、`alpha=1/3`。以后你只需修改 `d0_cm2_s_at_reference`；代码会检查 `alpha` 是否仍为 \(1/3\)。

## 10 pc 周期盒如何覆盖约 1 kpc

`box_size_pc: 10.0` 表示 Lucas 2021 MHD 立方盒的真实物理边长。CRPropa 的粒子坐标不在 10 pc 边界处回卷，粒子可以传播到数百 pc 或约 1 kpc；只有磁场和气体查询采用

\[
\mathbf x_{\rm box}=\mathbf x\bmod 10\ {\rm pc}.
\]

因此内存中始终只有一个 MHD 小盒，不会创建 \(100^3\) 个副本。`injection_scale_fraction_of_box` 只是所选快照的元数据，必须按快照实际驱动尺度填写；它不会把 10 pc 盒重新缩放。

无量纲场的物理缩放为

\[
n(\mathbf x)=\hat n(\mathbf x)
\frac{\langle n\rangle_{\rm target}}{\langle\hat n\rangle},
\qquad
\mathbf B(\mathbf x)=\hat{\mathbf B}(\mathbf x)
\frac{B_{\rm target}}{\sqrt{\langle|\hat{\mathbf B}|^2\rangle}}.
\]

这保留了原快照的密度对比、磁场方向、间歇结构和周期性。如果论文中的 3 μG 指平均矢量强度而非 RMS，改为 `magnetic_normalization: mean_vector`。

## `estimate_transport.py` 的定位

`estimate_transport.py` 被保留为**独立的 test-particle 对照工具**。它可根据相匹配的完整磁场 test-particle 标定，生成 mirror + pitch-angle scattering 的 \(D_\parallel(E)\) 诊断表。

关键点是：

- `crpropa_run_x3.py` 不读取该脚本生成的系数；
- 即使没有 `outputs/mirror_scattering_transport.json`，CRPropa 仍可运行；
- 正式 CRPropa 系数只来自 `transport.user_diffusion`；
- 对照结果与用户输入系数的差异只作为诊断，不解释成预设物理结论。

如果以后拿到新的 test-particle 表，可配置：

```yaml
coefficient_model: external_csv
external_csv: data/lucas_diffusion_table.csv
```

CSV 格式：

```text
energy_TeV,D_parallel_cm2_s,D_perp_cm2_s
```

## 运行

安装：

```bash
mamba env create -f environment.yml
conda activate x3-crpropa
```

准备背景、生成可选 test-particle 对照并验证配置：

```bash
python run_pipeline.py --config x3_config.yaml --prepare-only
```

少量粒子冒烟测试：

```bash
OMP_NUM_THREADS=8 python run_pipeline.py --config x3_config.yaml \
  --age-kyr 300 --n-particles 512 --no-postprocess
```

生产运行：

```bash
OMP_NUM_THREADS=16 python run_pipeline.py --config x3_config.yaml
```

也可分步运行：

```bash
python prepare_mhd_background.py --config x3_config.yaml
python estimate_transport.py --config x3_config.yaml       # 可选诊断
python validate_x3.py --config x3_config.yaml
python crpropa_run_x3.py --config x3_config.yaml --engine crpropa_sde_mhd
python validate_x3.py --config x3_config.yaml \
  --input outputs/protons_age_300kyr_crpropa_sde_mhd.npz
python postprocess_to_earth.py --config x3_config.yaml \
  outputs/protons_age_300kyr_crpropa_sde_mhd.npz
```

## 不装 CRPropa 的接口测试

`make_demo_mhd.py` 只用于测试读取和缩放接口，不能用于科学结果：

```bash
python make_demo_mhd.py
python inspect_mhd_file.py data/demo_mhd.npz
python prepare_mhd_background.py --config x3_config.yaml --input data/demo_mhd.npz
python estimate_transport.py --config x3_config.yaml
python validate_x3.py --config x3_config.yaml
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `inspect_mhd_file.py` | 列出真实文件的数据集、shape、dtype 和属性 |
| `mhd_io.py` | HDF5/NPZ/NPY 适配、轴变换、周期插值和 CRPropa 网格导出 |
| `prepare_mhd_background.py` | 缩放 Lucas 2021 的 10 pc 周期背景 |
| `estimate_transport.py` | 可选 mirror + scattering test-particle 对照，不控制 CRPropa |
| `crpropa_run_x3.py` | 使用 YAML 中的 \(D_0\)、\(\alpha=1/3\) 和周期 MHD 场传播 |
| `postprocess_to_earth.py` | 周期气体加权、pp 伽马射线、孔径/径向轮廓和吸收 |
| `validate_x3.py` | 验证 10 pc MHD、用户扩散系数、步长、薄靶和端点扩散矩 |
| `run_pipeline.py` | 全流程入口 |

## 仍需在论文中明确的假设

- 同一个 10 pc 周期实现被视为约 1 kpc 区域的统计代表；建议用多个快照或平移/旋转实现评估系统误差。
- 当前不是逐回旋轨道积分，而是用用户指定的扩散张量进行 SDE 传播；test-particle 结果只是独立对照。
- MHD 密度被当作 pp 靶核子密度；若文件存的是质量密度或 \(n_{\rm H_2}\)，需要确认换算。
- 质子传播忽略 pp 能损，后处理采用薄靶加权；验证文件会报告保守碰撞概率。
- `proton_power_erg_s=1e38` 是线性归一化占位值，应按最终谱拟合修改。
