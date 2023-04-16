import requests
from datetime import datetime
import numpy as np

def getDailyInfo(timeReq):
    time_list=timeReq.split(':')
    float_hour=float(time_list[0])
    float_minut=float(time_list[1])
    current_date = datetime.now().strftime("%Y-%m-%d")
    date_string = current_date+' '+timeReq
    date_object = datetime.strptime(date_string, '%Y-%m-%d %H:%M')
    current_time_seconds = str(int(date_object.timestamp()))
    api_key = 'f5f0abf1ae708ee51b889d207f4c2b6d'
    lat = '41.15'
    lon = '-8.61024'
    url = f'http://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&dt={current_time_seconds}&appid={api_key}&units=metric'

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        try:
            temperature = float(data['current']['temp'])
        except:
            temperature=0.0

        try:
            dewpoint = float(data['current']['dew_point'])
        except:
            dewpoint=0.0

        try:
            pressure = float(data['current']['pressure'])
        except:
            pressure=0.0
        
        try:
            wind_direction = float(data['current']['wind_deg'])
        except:
            wind_direction=0.0

        try:
            wind_speed = float(data['current']['wind_speed']* 1.60934)#miles/hour-KM/H 
        except:
            wind_speed=0.0

        try:
            wind_gust = float(data['current']['wind_gust']* 3.6)#metre/sec-KM/H 
        except:
            len_=len(data['hourly'])
            for i in range(0, len_):
                try:
                    wind_gust=float(data['hourly'][len_-i]['wind_gust']* 3.6)
                except:
                    wind_gust=None
                if wind_gust is not None:
                    break
        if wind_gust is None:
            wind_gust=0.0   
        humidity = float(data['current']['humidity'])
        try:
            hourly_precip = float(data['hourly'][0]['rain']['1h'])
        except:
            hourly_precip=0.0
        try:
            daily_rain = float(data['daily'][0]['rain'])
        except:
            daily_rain=0.0
        try:
            solar_radiation = solar_radiation = data['current']['uvi']* 40 
            solar_radiation=float(solar_radiation)
        except:
            solar_radiation=0.0
        info=[[float_hour, float_minut,temperature, dewpoint, pressure, wind_direction,wind_speed,wind_gust,humidity,hourly_precip,daily_rain,solar_radiation]]
        info=np.array(info)
        return info

