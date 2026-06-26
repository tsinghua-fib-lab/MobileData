# MobileCN Data and ZoomDiff Workflow

The dataset is available from [GitHub Releases](https://github.com/tsinghua-fib-lab/MobileData/releases/tag/v1.0).

This README describes the workflow for generating urban mobile traffic and mobile user distributions with ZoomDiff. Run all commands from the repository root directory, not from inside `ZoomDiff/`, because the scripts use paths such as `ZoomDiff/datasets/...`.

## 1. Dataset Layout

After downloading or preparing the data, organize `ZoomDiff/datasets` as follows:

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
│           └── POI_<City>
├── _data_preparation
├── _shared_geographic_data
│   ├── China_city_boundaries
│   │   └── china_city_boundaries_2024.shp
│   └── chn_ppp_2020_constrained.tif
├── template_scaler_traffic.pkl
├── template_scaler_user.pkl
├── template_traffic.npz
└── template_user.npz
```

Use English city directory names, for example `Nanjing` and `Nanchang`. The scripts also accept supported Chinese city names, such as `南京`, and map them to the corresponding English directory.

## 2. Prepare Geographic Inputs

### 2.1 Shared Geographic Data

Place shared geographic files under:

```text
ZoomDiff/datasets/_shared_geographic_data
```

Required files:

- `China_city_boundaries/china_city_boundaries_2024.shp`
- `chn_ppp_2020_constrained.tif`

The population raster can be downloaded from WorldPop:

```text
https://hub.worldpop.org/geodata/listing?id=135
```

### 2.2 City Geographic Data

For each target city, place OSM and POI data under:

```text
ZoomDiff/datasets/cities/<City>/geographic_data
```

For example:

```text
ZoomDiff/datasets/cities/Nanjing/geographic_data/OSM_Jiangsu
ZoomDiff/datasets/cities/Nanjing/geographic_data/POI_Nanjing
```

OSM data can be downloaded from:

```text
https://download.geofabrik.de/asia/china.html
```

POI data may come from any suitable source. For reproducible training and inference, the POI data used across cities should use the same POI category set and encoding.

## 3. Preprocess Data

Preprocessing has two parts:

1. `Env_extra.py` creates condition files, such as population, road length, water area, and POI features.
2. `Grid_meta.py` creates training data files from raw mobile traffic or user data.

For inference-only cities, run only Step 3.1. For training cities, run both Step 3.1 and Step 3.2.

### 3.1 Create Condition Files

Run:

```bash
python ZoomDiff/datasets/_data_preparation/Env_extra.py \
  --cityname "南京" \
  --province "江苏省" \
  --shp_path "ZoomDiff/datasets/_shared_geographic_data/China_city_boundaries/china_city_boundaries_2024.shp" \
  --pop_path "ZoomDiff/datasets/_shared_geographic_data/chn_ppp_2020_constrained.tif"
```

By default, this reads:

```text
ZoomDiff/datasets/cities/<City>/geographic_data/OSM_<Province>
ZoomDiff/datasets/cities/<City>/geographic_data/POI_<City>
```

and writes:

```text
ZoomDiff/datasets/cities/<City>/cond/<City>_cond.npz
```

If your OSM or POI directories use different paths, add:

```bash
--osm_dir "<OSM_DIR>" --poi_dir "<POI_DIR>"
```

You can also edit and run the example script:

```bash
bash ZoomDiff/datasets/_data_preparation/src/Env_gen.sh
```

### 3.2 Create Training Data Files

First place raw mobile data under:

```text
ZoomDiff/datasets/cities/<City>/raw_data_for_train/<City>_<datatype>.npz
```

Then run:

```bash
python ZoomDiff/datasets/_data_preparation/Grid_meta.py \
  --cityname "南京" \
  --province "江苏省" \
  --datatype traffic \
  --shp_path "ZoomDiff/datasets/_shared_geographic_data/China_city_boundaries/china_city_boundaries_2024.shp" \
  --pop_path "ZoomDiff/datasets/_shared_geographic_data/chn_ppp_2020_constrained.tif"
```

This writes:

```text
ZoomDiff/datasets/cities/<City>/data/<City>_<datatype>_data.npz
```

Use `--datatype user` to prepare mobile user data. If the raw data is not in the default location, add:

```bash
--data_path "<RAW_DATA_NPZ>"
```

You can also edit and run the example script:

```bash
bash ZoomDiff/datasets/_data_preparation/src/Grid_meta_gen.sh
```

## 4. Use Open Pretrained Models for Direct Inference

ZoomDiff is based on:

> X. Qi, H. Chai, S. Liu, L. Yue, R. Pan, Y. Wang, and Y. Li, "Denoising refinement diffusion models for simultaneous generation of multi-scale mobile network traffic," *arXiv preprint arXiv:2511.17532*, Oct. 2025, doi: 10.48550/arXiv.2511.17532.

The repository provides two pretrained checkpoints:

```text
ZoomDiff/save/pretrain/traffic_ckpt/model.pth
ZoomDiff/save/pretrain/user_ckpt/model.pth
```

Before inference, the target city must already have:

```text
ZoomDiff/datasets/cities/<City>/cond/<City>_cond.npz
```

Run traffic inference with the pretrained traffic model:

```bash
python ZoomDiff/ZoomDiff_infer.py \
  --dataset Nanjing \
  --datatype traffic \
  --device cuda:0 \
  --modelfolder pretrain/traffic_ckpt \
  --nsample 1
```

Run user inference with the pretrained user model:

```bash
python ZoomDiff/ZoomDiff_infer.py \
  --dataset Nanjing \
  --datatype user \
  --device cuda:0 \
  --modelfolder pretrain/user_ckpt \
  --nsample 1
```

If `--modelfolder` is omitted, `ZoomDiff_infer.py` automatically uses:

```text
pretrain/<datatype>_ckpt
```

Inference outputs are saved as 500 m grid data masked by the city boundary shapefile:

```text
ZoomDiff/results/<City>_500m_<datatype>.npz
```

The output includes `data_500m`, condition features converted to 500 m where applicable, and the masked grid coordinates `lat` and `lon`.

## 5. Train a Model, Then Run Inference

### 5.1 Train

Training cities must have both condition files and data files:

```text
ZoomDiff/datasets/cities/<City>/cond/<City>_cond.npz
ZoomDiff/datasets/cities/<City>/data/<City>_<datatype>_data.npz
```

Run:

```bash
python ZoomDiff/ZoomDiff_train.py \
  --dataset "Nanchang*Nanjing" \
  --datatype traffic \
  --device cuda:0
```

Use `--datatype user` to train a mobile user distribution model.

The script prints the model output directory, for example:

```text
model folder: ZoomDiff/save/traffic_Nanchang*Nanjing_t1s3_YYYYMMDD_HHMMSS
```

### 5.2 Infer with the Trained Checkpoint

Pass the trained folder name relative to `ZoomDiff/save`:

```bash
python ZoomDiff/ZoomDiff_infer.py \
  --dataset Nanjing \
  --datatype traffic \
  --device cuda:0 \
  --modelfolder "traffic_Nanchang*Nanjing_t1s3_YYYYMMDD_HHMMSS" \
  --nsample 1
```

The output is saved to:

```text
ZoomDiff/results/<City>_500m_<datatype>.npz
```

## 6. Key Arguments

- `--dataset`: one city or multiple cities separated by `*`, for example `Nanchang*Nanjing`
- `--datatype`: `traffic` or `user`
- `--device`: `cpu`, `cuda:0`, `cuda:1`, etc.
- `--nsample`: number of generated samples per inference run
- `--modelfolder`: checkpoint directory relative to `ZoomDiff/save`
- `--shp_path`: city boundary shapefile used to mask 500 m inference outputs
