# 📘 Project Workflow

This README describes the complete workflow of the project, including **dataset preprocessing**, **model training**, and **model inference**.

------

## 🗂️ Dataset Preprocessing

### 1️⃣ Data Preparation

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

- ⚙️ **Condition preparation**

  ```
  ./datasets/data_preparation/src/cond_preparation.sh
  ```

- 📦 **Data preparation**

  ```
  ./datasets/data_preparation/src/data_preparation.sh
  ```

------

### 3️⃣ Preprocess City Data for Inference

For **model inference**, only the condition preparation step is required:

- ⚙️ **Condition preparation**

  ```
  ./datasets/data_preparation/src/cond_preparation.sh
  ```

------

## 🚀 Model Training

Run the following command to train the model:

```
python exe_train.py \
  --dataset TrainCity1*TrainCity2*TrainCity3*... \
  --datatype traffic \
  --device cuda:0
```

**Arguments:**

- 🏙️ `--dataset`: Training cities, separated by `*`
- 📊 `--datatype`: Type of data (e.g., `traffic`, `user`)
- 💻 `--device`: Computing device (e.g., `cuda:0`)

------

## 🔍 Model Inference

Run the following command to perform inference:

```
python exe_inference.py \
  --dataset InfCity1*InfCity2*InfCity3*... \
  --datatype traffic \
  --device cuda:0 \
  --nsample 1
```

**Arguments:**

- 🏙️ `--dataset`: Target cities for inference, separated by `*`
- 📊 `--datatype`: Type of data (e.g., `traffic`, `user`)
- 💻 `--device`: Computing device (e.g., `cuda:0`)
- 🎯 `--nsample`: Number of samples generated per condition

------

✨ *This pipeline ensures consistent preprocessing, reliable training, and reproducible inference across multiple cities.*