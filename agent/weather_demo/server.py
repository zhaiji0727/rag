import json
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP

import os
import urllib3

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
# Suppress warnings about Elasticsearch certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 初始化 MCP 服务器
mcp = FastMCP("WeatherServer")

# OpenWeather API 配置
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "db38fb6e13151188064989ee1794680c"  # 请替换为你自己的 OpenWeather API Key
USER_AGENT = "weather-app/1.0"


async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    从 OpenWeather API 获取天气信息。
    :param city: 城市名称（需使用英文，如 Beijing）
    :return: 天气数据字典；若出错返回包含 error 信息的字典
    """
    # weather = '{"coord": {"lon": 116.3972, "lat": 39.9075}, "weather": [{"id": 800, "main": "Clear", "description": "晴", "icon": "01d"}], "base": "stations", "main": {"temp": 23.94, "feels_like": 22.65, "temp_min": 23.94, "temp_max": 23.94, "pressure": 1005, "humidity": 10, "sea_level": 1005, "grnd_level": 1000}, "visibility": 10000, "wind": {"speed": 4.63, "deg": 343, "gust": 9.31}, "clouds": {"all": 6}, "dt": 1744612286, "sys": {"type": 1, "id": 9609, "country": "CN", "sunrise": 1744580304, "sunset": 1744627839}, "timezone": 28800, "id": 1816670, "name": "Beijing", "cod": 200}'
    # return json.loads(weather)
    params = {"q": city, "appid": API_KEY, "units": "metric", "lang": "zh_cn"}
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.get(
                OPENWEATHER_API_BASE, params=params, headers=headers, timeout=30.0
            )
            response.raise_for_status()
            return response.json()  # 返回字典类型
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 错误: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}


def format_weather(data: dict[str, Any] | str) -> str:
    """
    将天气数据格式化为易读文本。
    :param data: 天气数据（可以是字典或 JSON 字符串）
    :return: 格式化后的天气信息字符串
    """
    # 如果传入的是字符串，则先转换为字典
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return f"无法解析天气数据: {e}"

    # 如果数据中包含错误信息，直接返回错误提示
    if "error" in data:
        return f"⚠️ {data['error']}"

    # 提取数据时做容错处理
    city = data.get("name", "未知")
    country = data.get("sys", {}).get("country", "未知")
    temp = data.get("main", {}).get("temp", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    # weather 可能为空列表，因此用 [0] 前先提供默认字典
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "未知")

    return (
        f"🌍 {city}, {country}\n"
        f"🌡 温度: {temp}°C\n"
        f"💧 湿度: {humidity}%\n"
        f"🌬 风速: {wind_speed} m/s\n"
        f"🌤 天气: {description}\n"
    )


@mcp.tool()
async def query_weather(city: str) -> str:
    """
    输入指定城市的英文名称，返回今日天气查询结果。
    :param city: 城市名称（需使用英文）
    :return: 格式化后的天气信息
    """
    data = await fetch_weather(city)
    return format_weather(data)


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport="stdio")
