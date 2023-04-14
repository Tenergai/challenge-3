import streamlit as sl
import requests

sl.write("""
# Tenergito
""")


url = 'http://192.168.1.1'

def fetch_forecast(url):
    response = requests.get(url+'/forecast')
    data = response.json()
    return data['forecast']

def fetch_report(url):
    response = requests.get(url+'/report')
    data = response.json()
    return data['report']

sl.write("""
### Generation Forecast
""")
forecast = fetch_forecast(url)
sl.line_chart(forecast)


devices = ["Washing Machine", "Fridge", "TV", "Heater"]
option = sl.selectbox(
        "What model do you want to use?",
        (devices),
    )

for i, device in enumerate(devices):
    if option == device:
        with sl.spinner('Wait for it...'):
            report = fetch_report()
        sl.write(report)
