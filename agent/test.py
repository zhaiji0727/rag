import httpx
from typing import Any
import os
import urllib3
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
# Suppress warnings about Elasticsearch certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "db38fb6e13151188064989ee1794680c"  # 请替换为你自己的 OpenWeather API Key
USER_AGENT = "weather-app/1.0"

async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    从 OpenWeather API 获取天气信息。
    :param city: 城市名称（需使用英文，如 Beijing）
    :return: 天气数据字典；若出错返回包含 error 信息的字典
    """
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "zh_cn"
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.get(OPENWEATHER_API_BASE, params=params, headers=headers, timeout=120.0)
            response.raise_for_status()
            return response.json()  # 返回字典类型
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 错误: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}
        
response = await fetch_weather('Beijing')
print(response)

'''
{'coord': {'lon': 116.3972, 'lat': 39.9075}, 'weather': [{'id': 800, 'main': 'Clear', 'description': '晴', 'icon': '01d'}], 'base': 'stations', 'main': {'temp': 23.94, 'feels_like': 22.65, 'temp_min': 23.94, 'temp_max': 23.94, 'pressure': 1005, 'humidity': 10, 'sea_level': 1005, 'grnd_level': 1000}, 'visibility': 10000, 'wind': {'speed': 4.63, 'deg': 343, 'gust': 9.31}, 'clouds': {'all': 6}, 'dt': 1744612286, 'sys': {'type': 1, 'id': 9609, 'country': 'CN', 'sunrise': 1744580304, 'sunset': 1744627839}, 'timezone': 28800, 'id': 1816670, 'name': 'Beijing', 'cod': 200}
'''