import os


ZOOMDIFF_DIR = "ZoomDiff"
DATASETS_DIR = os.path.join(ZOOMDIFF_DIR, "datasets")
CITIES_DIR = os.path.join(DATASETS_DIR, "cities")
SHARED_GEOGRAPHIC_DIR = os.path.join(DATASETS_DIR, "_shared_geographic_data")
DATA_PREPARATION_DIR = os.path.join(DATASETS_DIR, "_data_preparation")

CITY_EN = {
    "上海": "Shanghai",
    "北京": "Beijing",
    "南京": "Nanjing",
    "南宁": "Nanning",
    "南昌": "Nanchang",
    "富阳": "Fuyang",
    "杭州": "Hangzhou",
    "济南": "Jinan",
    "南阳": "Nanyang",
    "呼和浩特": "Hohhot",
    "唐山": "Tangshan",
    "成都": "Chengdu",
    "烟台": "Yantai",
    "珠海": "Zhuhai",
    "长春": "Changchun",
    "阳江": "Yangjiang",
    "驻马店": "Zhumadian",
    "乌鲁木齐": "Urumqi",
    "兰州": "Lanzhou",
    "台北": "Taipei",
    "合肥": "Hefei",
    "哈尔滨": "Harbin",
    "天津": "Tianjin",
    "太原": "Taiyuan",
    "广州": "Guangzhou",
    "拉萨": "Lhasa",
    "昆明": "Kunming",
    "武汉": "Wuhan",
    "沈阳": "Shenyang",
    "海口": "Haikou",
    "澳门": "Macau",
    "石家庄": "Shijiazhuang",
    "福州": "Fuzhou",
    "西宁": "Xining",
    "西安": "Xian",
    "贵阳": "Guiyang",
    "郑州": "Zhengzhou",
    "重庆": "Chongqing",
    "银川": "Yinchuan",
    "长沙": "Changsha",
    "香港": "HongKong",
}


def city_slug(city):
    return CITY_EN.get(city, city)


def city_dataset_dir(city):
    return os.path.join(CITIES_DIR, city_slug(city))


def city_data_file(city, datatype):
    slug = city_slug(city)
    return os.path.join(city_dataset_dir(city), "data", f"{slug}_{datatype}_data.npz")


def city_cond_file(city):
    slug = city_slug(city)
    return os.path.join(city_dataset_dir(city), "cond", f"{slug}_cond.npz")


def city_raw_train_file(city, datatype):
    slug = city_slug(city)
    return os.path.join(city_dataset_dir(city), "raw_data_for_train", f"{slug}_{datatype}.npz")
