# 📦 MobileCN data.
**The dataset is available for download in [GitHub Releases](https://github.com/yourname/repo/releases).**

# 📘 Workflow of Data Generation

This README describes the complete workflow of **urban mobile traffic and mobile user distribution generation**, including **dataset preprocessing**, **model training**, and **model inference**. Please execute all commands in the following directory:

```
./ZoomDiff/
```

Or execute directly in the current directory:

```
cd ./ZoomDiff/
```

------

## 🗂️ Dataset Preprocessing

### 1️⃣ Data Preparation

##### 1.1 Population Distribution Data

Please place the national population distribution data under the following directory:

```
./datasets/data_preparation/geographic_data/
```

Taking **China** as an example, the required geographic data includes:

- 🗺️ **China Population Distribution file (.tif)**
  Download from:
  https://hub.worldpop.org/geodata/listing?id=135

##### 1.2 Geographic Data of the Target City

Please place the geographic data of the target city under the following directory:

```
./datasets/data_preparation/geographic_data/
```

Taking **Chengdu** as an example, the required geographic data includes:

- 🗺️ **OSM – Sichuan Province**
  Download from:
  https://download.geofabrik.de/asia/china.html
- 📍 **POI – Chengdu**
  You may obtain POI data from any suitable source.
  ⚠️ **Important:** The POI data used for **training** and **inference** must share the **same set of `bigtype` categories and identical encodings**.

------

### 2️⃣ Preprocess City Data for Training

To preprocess city data for **model training**, execute the following scripts in order:

- 📦 **Grid-level Data Transformation**

  ```
  python ./datasets/data_preparation/Grid_meta.py \
    --cityname "${CITYNAME}" \
    --province "${PROVINCE}" \
    --data_path "${DATA_PATH}" \
    --shp_path "${SHP_PATH}" \
    --pop_path "${POP_PATH}"
  ```

  Please fill in the correct names or paths in `${...}`. Alternatively, you can directly run:

  ```
  ./datasets/data_preparation/src/Grid_meta_gen.sh
  ```

- ⚙️ **Environmental Feature Fusion**

  ```
  python ./datasets/data_preparation/Env_extra.py \
    --cityname "${CITYNAME}" \
    --province "${PROVINCE}" \
    --shp_path "${SHP_PATH}" \
    --pop_path "${POP_PATH}" \
    --osm_dir "${OSM_DIR}" \
    --poi_dir "${POI_DIR}"
  ```

  Please fill in the correct names or paths in `${...}`. Alternatively, you can directly run:

  ```
  ./datasets/data_preparation/src/Env_gen.sh
  ```

------

### 3️⃣ Preprocess City Data for Inference

For **model inference**, only the condition preparation step is required:

- ⚙️ **Environmental Feature Fusion**

  ```
  ./datasets/data_preparation/src/Env_gen.sh
  ```

------

## 🚀 ZoomDiff Model Training

For the detail of ZoomDiff model, please refer to **Denoising Refinement Diffusion Models for Simultaneous Generation of Multi-scale Mobile Traffic**：

> X. Qi, H. Chai, S. Liu, L. Yue, R. Pan, Y. Wang, and Y. Li, “Denoising refinement diffusion models for simultaneous generation of multi-scale mobile network traffic,” *arXiv preprint arXiv:2511.17532*, Oct. 2025, doi: 10.48550/arXiv.2511.17532.

Run the following command to train the ZoomDiff model:

```
python ZoomDiff_train.py \
  --dataset TrainCity1*TrainCity2*TrainCity3*... \
  --datatype traffic \
  --device cuda:0
```

**Arguments:**

- 🏙️ `--dataset`: Training cities, separated by `*`
- 📊 `--datatype`: Type of data (e.g., `traffic`, `user`)
- 💻 `--device`: Computing device (e.g., `cpu`, `cuda:0`)

------

## 🔍 ZoomDiff Model Inference

Run the following command to perform inference:

```
python ZoomDiff_infer.py \
  --dataset InfCity1*InfCity2*InfCity3*... \
  --datatype traffic \
  --device cuda:0 \
  --nsample 1
```

**Arguments:**

- 🏙️ `--dataset`: Target cities for inference, separated by `*`
- 📊 `--datatype`: Type of data (e.g., `traffic`, `user`)
- 💻 `--device`: Computing device (e.g., `cpu`, `cuda:0`)
- 🎯 `--nsample`: Number of samples generated per single inference run

------


✨ *This pipeline ensures consistent preprocessing, reliable training, and reproducible inference across multiple cities.*
