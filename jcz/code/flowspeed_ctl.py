import re
from pathlib import Path

src = Path('/home/featurize/Bench2Drive/leaderboard/data/1_jcz_bench2drive220_uniad_speedlimit.xml')
dst = Path('/home/featurize/Bench2Drive/leaderboard/data/4_jcz_bench2drive220_uniad_speedlimit.xml')

def fmt(num: float) -> str:
    """若结果是整数，转回 int；否则保留最多两位小数"""
    return str(int(num)) if num.is_integer() else f'{num:.2f}'.rstrip('0').rstrip('.')

def sub_flow_speed(m):
    return f'<flow_speed value="{fmt(float(m.group(1)) - 2)}" />'

def sub_bicycle_speed(m):
    return f'<bicycle_speed value="{fmt(float(m.group(1)) - 1)}" />'

def sub_speed(m):
    return f'<speed value="{fmt(float(m.group(1)) - 2*3.6)}" />'   # 2 m/s → 7.2 km/h

pattern_funcs = [
    (r'<flow_speed value="([\d.]+)"\s*/>',   sub_flow_speed),
    (r'<bicycle_speed value="([\d.]+)"\s*/>', sub_bicycle_speed),
    (r'<speed value="([\d.]+)"\s*/>',        sub_speed),
]

text = src.read_text(encoding='utf-8')
for pat, func in pattern_funcs:
    text = re.sub(pat, func, text)

dst.write_text(text, encoding='utf-8')
print(f'数值调整完成，结果已保存到 {dst}')