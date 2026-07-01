# MobileCN | Data and Workflow

This document describes the workflow for generating **urban mobile traffic** and **mobile user distributions** with ZoomDiff, including data preparation, pretrained inference, and model training.

Run all commands from the repository root directory, because the scripts use paths such as `ZoomDiff/datasets/...`.

## Contents

- [Data Assets](#data-assets)
- [Generation with Pretrained Checkpoints](#generation-with-pretrained-checkpoints)
- [Model Training](#model-training)
- [Key Arguments](#key-arguments)

<a id="data-assets"></a>

## 📦 Data Assets

The prepared open environment data has been uploaded to the [`v1.0` GitHub Release](https://github.com/tsinghua-fib-lab/MobileData/releases/tag/v1.0) as per-city `.tar.zst` assets.

Asset naming:

```text
MobileCN_data_mobile_<City>.tar.zst
MobileCN_data_mobile__shared_geographic_data.tar.zst
```

<a id="generation-with-pretrained-checkpoints"></a>

## 🚀 Generation with Pretrained Checkpoints

### 1️⃣ Prepare Geographic Data

Download the geographic data required by the generation pipeline:

- 🗺️ **Population raster**: China population distribution file (`.tif`) from [WorldPop](https://hub.worldpop.org/geodata/listing?id=135).
- 🗺️ **OSM data**: China provincial OSM data from [Geofabrik](https://download.geofabrik.de/asia/china.html).
- 📍 **POI data**: City-level point-of-interest data from any suitable source.

> **Important:** POI data used for training and inference must share the same set of `bigtype` categories and identical category encodings.

Geographic data of some major cities in China are available from [GitHub Releases](https://github.com/tsinghua-fib-lab/MobileData/releases/tag/v1.0). Organize `ZoomDiff/datasets` as follows. City and province directory names are expected to be in English, for example `Nanchang` and `Jiangxi`.

```text
ZoomDiff/datasets
├── cities
│   └── <City>
│       └── geographic_data
│           ├── OSM_<Province>
│           │   └── ...
│           └── POI_<City>
│               └── ...
└── _shared_geographic_data
    ├── China_city_boundaries
    │   ├── china_city_boundaries_2024.shp
    │   └── ...
    └── chn_ppp_2020_constrained.tif
```

<a id="build-condition-files"></a>

### 2️⃣ Build Condition Files

`Env_extra.py` creates environmental condition files, including population, road length, water area, building count, and POI features.

Run:

```bash
python ZoomDiff/datasets/_data_preparation/Env_extra.py \
  --cityname <City> \
  --province <Province>
```

By default, the script reads:

```text
ZoomDiff/datasets/cities/<City>/geographic_data/OSM_<Province>
ZoomDiff/datasets/cities/<City>/geographic_data/POI_<City>
```

and writes:

```text
ZoomDiff/datasets/cities/<City>/cond/<City>_cond.npz
```

You can also edit and run the example script:

```bash
bash ZoomDiff/datasets/_data_preparation/src/Env_gen.sh
```

### 3️⃣ Run Pretrained Inference

ZoomDiff is the generative model used for MobileCN. It is based on:

> X. Qi, H. Chai, S. Liu, L. Yue, R. Pan, Y. Wang, and Y. Li, "Denoising refinement diffusion models for simultaneous generation of multi-scale mobile network traffic," *arXiv preprint arXiv:2511.17532*, Oct. 2025, doi: 10.48550/arXiv.2511.17532.

The repository provides two pretrained checkpoints:

```text
ZoomDiff/save/pretrain/traffic_ckpt/model.pth
ZoomDiff/save/pretrain/user_ckpt/model.pth
```

Before inference, the target city must already have a condition file:

```text
ZoomDiff/datasets/cities/<City>/cond/<City>_cond.npz
```

Run traffic inference:

```bash
python ZoomDiff/ZoomDiff_infer.py \
  --dataset <City> \
  --datatype traffic \
  --device <Device> \
  --modelfolder pretrain/traffic_ckpt \
  --nsample 1
```

Run user inference:

```bash
python ZoomDiff/ZoomDiff_infer.py \
  --dataset <City> \
  --datatype user \
  --device <Device> \
  --modelfolder pretrain/user_ckpt \
  --nsample 1
```

Example devices include `cpu`, `cuda:0`, and `cuda:1`.

If `--modelfolder` is omitted, `ZoomDiff_infer.py` automatically uses:

```text
pretrain/<datatype>_ckpt
```

Inference outputs are saved as 500 m grid data masked by the city boundary shapefile:

```text
ZoomDiff/results/<City>_500m_<datatype>.npz
```

The output includes `data_500m`, condition features converted to 500 m where applicable, and masked grid coordinates `lat` and `lon`.

<a id="model-training"></a>

## 🧪 Model Training

### 1️⃣ Prepare Training Data

Training uses the same geographic data as pretrained inference. In addition, you need raw base-station-level measurements for mobile traffic and/or mobile user counts.

Organize `ZoomDiff/datasets` as follows:

```text
ZoomDiff/datasets
├── cities
│   └── <City>
│       ├── cond
│       │   └── <City>_cond.npz
│       ├── data
│       │   ├── <City>_traffic_data.npz
│       │   └── <City>_user_data.npz
│       ├── raw_data_for_train
│       │   ├── <City>_traffic.npz
│       │   └── <City>_user.npz
│       └── geographic_data
│           ├── OSM_<Province>
│           │   └── ...
│           └── POI_<City>
│               └── ...
└── _shared_geographic_data
    ├── China_city_boundaries
    │   ├── china_city_boundaries_2024.shp
    │   └── ...
    └── chn_ppp_2020_constrained.tif
```

### 2️⃣ Preprocess Training Inputs

The preprocessing pipeline has two parts:

- `Grid_meta.py` converts raw base-station-level measurements into gridded training data.
- `Env_extra.py` creates condition files, such as population, road length, water area, building count, and POI features.

#### 2.1 Create Gridded Training Data

Run:

```bash
python ZoomDiff/datasets/_data_preparation/Grid_meta.py \
  --cityname <City> \
  --province <Province> \
  --datatype <Datatype>
```

`<Datatype>` should be either `traffic` or `user`.

The script writes:

```text
ZoomDiff/datasets/cities/<City>/data/<City>_<datatype>_data.npz
```

You can also edit and run the example script:

```bash
bash ZoomDiff/datasets/_data_preparation/src/Grid_meta_gen.sh
```

#### 2.2 Create Condition Files

Run `Env_extra.py` as described in [Build Condition Files](#build-condition-files):

```bash
python ZoomDiff/datasets/_data_preparation/Env_extra.py \
  --cityname <City> \
  --province <Province>
```

### 3️⃣ Train ZoomDiff

Run:

```bash
python ZoomDiff/ZoomDiff_train.py \
  --dataset <Cities> \
  --datatype <Datatype> \
  --device <Device>
```

The script prints the model output directory, for example:

```text
model folder: ZoomDiff/save/traffic_<Cities>_t1s3_YYYYMMDD_HHMMSS
```

<a id="key-arguments"></a>

## ⚙️ Key Arguments

| Argument          | Used by                                                        | Description                                                                  |
| ----------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `--cityname`    | `Env_extra.py`, `Grid_meta.py`                             | English city name, such as`Nanchang`.                                      |
| `--province`    | `Env_extra.py`, `Grid_meta.py`                             | English province name used by`OSM_<Province>`, such as `Jiangxi`.        |
| `--dataset`     | `ZoomDiff_train.py`, `ZoomDiff_infer.py`                   | One city or multiple cities separated by`*`, such as `Nanchang*Nanjing`. |
| `--datatype`    | `Grid_meta.py`, `ZoomDiff_train.py`, `ZoomDiff_infer.py` | `traffic` or `user`.                                                     |
| `--device`      | `ZoomDiff_train.py`, `ZoomDiff_infer.py`                   | `cpu`, `cuda:0`, `cuda:1`, etc.                                        |
| `--nsample`     | `ZoomDiff_infer.py`                                          | Number of generated samples per inference run.                               |
| `--modelfolder` | `ZoomDiff_train.py`, `ZoomDiff_infer.py`                   | Checkpoint directory relative to`ZoomDiff/save`.                           |
| `--shp_path`    | `Env_extra.py`, `Grid_meta.py`, `ZoomDiff_infer.py`      | City boundary shapefile path.                                                |
| `--pop_path`    | `Env_extra.py`, `Grid_meta.py`                             | Population raster path.                                                      |
| `--osm_dir`     | `Env_extra.py`                                               | Optional custom OSM directory.                                               |
| `--poi_dir`     | `Env_extra.py`                                               | Optional custom POI directory.                                               |
| `--data_path`   | `Grid_meta.py`                                               | Optional custom raw`.npz` file for training preprocessing.                 |
