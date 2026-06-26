from __future__ import annotations

import shutil
from pathlib import Path


DATASETS_DIR = Path("ZoomDiff") / "datasets"
CITIES_DIR = DATASETS_DIR / "cities"
DATA_PREP_DIR = DATASETS_DIR / "_data_preparation"
SOURCE_GROUPS = [
    DATA_PREP_DIR / "source_geographic_data" / "pretrain",
    DATA_PREP_DIR / "source_geographic_data" / "test1",
    DATA_PREP_DIR / "source_geographic_data" / "test2",
]

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

PROVINCE_EN = {
    "上海市": "Shanghai",
    "北京市": "Beijing",
    "江苏省": "Jiangsu",
    "广西壮族自治区": "Guangxi",
    "江西省": "Jiangxi",
    "浙江省": "Zhejiang",
    "山东省": "Shandong",
    "河南省": "Henan",
    "内蒙古自治区": "InnerMongolia",
    "河北省": "Hebei",
    "四川省": "Sichuan",
    "广东省": "Guangdong",
    "吉林省": "Jilin",
    "新疆维吾尔自治区": "Xinjiang",
    "甘肃省": "Gansu",
    "台湾省": "Taiwan",
    "安徽省": "Anhui",
    "黑龙江省": "Heilongjiang",
    "天津市": "Tianjin",
    "山西省": "Shanxi",
    "西藏自治区": "Tibet",
    "云南省": "Yunnan",
    "湖北省": "Hubei",
    "辽宁省": "Liaoning",
    "海南省": "Hainan",
    "澳门特别行政区": "Macau",
    "福建省": "Fujian",
    "青海省": "Qinghai",
    "陕西省": "Shaanxi",
    "贵州省": "Guizhou",
    "重庆市": "Chongqing",
    "宁夏回族自治区": "Ningxia",
    "湖南省": "Hunan",
    "香港特别行政区": "HongKong",
}

CATEGORY_EN = {
    "餐饮服务": "CateringServices",
    "购物服务": "ShoppingServices",
    "生活服务": "LifeServices",
    "体育休闲服务": "SportsLeisureServices",
    "医疗保健服务": "MedicalHealthcareServices",
    "住宿服务": "AccommodationServices",
    "风景名胜": "ScenicSpots",
    "商务住宅": "BusinessResidential",
    "政府机构及社会团体": "GovernmentAndOrganizations",
    "科教文化服务": "EducationCultureServices",
    "交通设施服务": "TransportationFacilities",
    "金融保险服务": "FinanceInsuranceServices",
    "公司企业": "Companies",
    "汽车服务": "AutomotiveServices",
    "汽车销售": "AutomotiveSales",
    "汽车维修": "AutomotiveRepair",
    "摩托车服务": "MotorcycleServices",
    "公共设施": "PublicFacilities",
    "地名地址信息": "PlaceNameAddressInfo",
    "道路附属设施": "RoadFacilities",
    "室内设施": "IndoorFacilities",
    "通行设施": "PassageFacilities",
    "事件活动": "EventsActivities",
    "虚拟数据": "VirtualData",
}


def find_city_sources() -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for group in SOURCE_GROUPS:
        if not group.exists():
            continue
        for city_dir in sorted(group.iterdir()):
            if city_dir.is_dir():
                sources.setdefault(city_dir.name, city_dir)
    return sources


def copy_osm(src_geo: Path, dst_city: Path) -> str:
    osm_dirs = sorted(p for p in src_geo.iterdir() if p.is_dir() and p.name.startswith("OSM_"))
    if len(osm_dirs) != 1:
        raise RuntimeError(f"Expected one OSM directory under {src_geo}, found {len(osm_dirs)}")

    province_cn = osm_dirs[0].name.removeprefix("OSM_")
    province_en = PROVINCE_EN.get(province_cn)
    if not province_en:
        raise KeyError(f"Missing English province mapping for {province_cn}")

    dst_osm = dst_city / f"OSM_{province_en}"
    if dst_osm.exists():
        shutil.rmtree(dst_osm)
    shutil.copytree(osm_dirs[0], dst_osm)
    return dst_osm.name


def poi_category_name(path: Path) -> str:
    stem = path.stem
    category_cn = stem.rsplit("_", 1)[-1]
    category_en = CATEGORY_EN.get(category_cn)
    if not category_en:
        raise KeyError(f"Missing English POI category mapping for {category_cn} in {path}")
    return category_en


def copy_poi(src_geo: Path, dst_city: Path, city_en: str) -> tuple[str, int]:
    poi_dirs = sorted(
        p for p in src_geo.iterdir()
        if p.is_dir() and p.name.startswith("POI")
    )
    if len(poi_dirs) != 1:
        raise RuntimeError(f"Expected one POI directory under {src_geo}, found {len(poi_dirs)}")

    dst_poi = dst_city / f"POI_{city_en}"
    if dst_poi.exists():
        shutil.rmtree(dst_poi)
    dst_poi.mkdir(parents=True)

    copied = 0
    used_names: set[str] = set()
    for src_file in sorted(poi_dirs[0].glob("*.csv")):
        if "暂时不使用的POI" in src_file.parts:
            continue
        category_en = poi_category_name(src_file)
        dst_name = f"{city_en}_{category_en}.csv"
        if dst_name in used_names:
            raise RuntimeError(f"Duplicate POI target filename {dst_name} from {src_file}")
        used_names.add(dst_name)
        shutil.copy2(src_file, dst_poi / dst_name)
        copied += 1
    return dst_poi.name, copied


def remove_old_city_dirs() -> None:
    for city_cn, city_en in CITY_EN.items():
        path = CITIES_DIR / city_en / "geographic_data"
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    sources = find_city_sources()
    missing = sorted(set(CITY_EN) - set(sources))
    if missing:
        raise RuntimeError(f"Missing source geographic_data for: {', '.join(missing)}")

    remove_old_city_dirs()

    for city_cn in sorted(CITY_EN, key=CITY_EN.get):
        city_en = CITY_EN[city_cn]
        src_geo = sources[city_cn]
        dst_city = CITIES_DIR / city_en / "geographic_data"
        dst_city.mkdir(parents=True, exist_ok=True)

        osm_name = copy_osm(src_geo, dst_city)
        poi_name, poi_count = copy_poi(src_geo, dst_city, city_en)
        print(f"{city_cn} -> {city_en}: {osm_name}, {poi_name}, {poi_count} POI files")


if __name__ == "__main__":
    main()
