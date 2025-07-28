import xml.etree.ElementTree as ET

def is_weather_mild(weather_elem):
    cloudiness = float(weather_elem.attrib.get("cloudiness", 0))
    fog_density = float(weather_elem.attrib.get("fog_density", 0))
    precipitation = float(weather_elem.attrib.get("precipitation", 0))
    wetness = float(weather_elem.attrib.get("wetness", 0))
    sun_altitude = float(weather_elem.attrib.get("sun_altitude_angle", 0))

    return (
        cloudiness < 70 and
        fog_density < 50 and
        precipitation < 50 and # 降水强度
        # wetness < 60 and   # 湿度
        sun_altitude > -20
    )

def filter_and_save_mild_routes(input_path, output_path):
    tree = ET.parse(input_path)
    root = tree.getroot()

    new_root = ET.Element("routes")  # 新的根元素

    for route in root.findall('route'):
        weathers = route.find('weathers')
        if weathers is not None:
            weather_list = weathers.findall('weather')
            if all(is_weather_mild(w) for w in weather_list):
                new_root.append(route)  # 满足条件则添加到新树

    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"已保存到 {output_path}")

if __name__ == "__main__":
    input_xml = '/home/featurize/Bench2Drive/leaderboard/data/bench2drive220.xml'
    output_xml = '/home/featurize/Bench2Drive/leaderboard/jcz_data/0_mild_routes.xml'  # 你也可以换成绝对路径

    filter_and_save_mild_routes(input_xml, output_xml)
