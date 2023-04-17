import streamlit as sl
import requests
import pandas as pd
import time
from shapModel.shapModel import getContributions

def fetch_forecast(url):
    return [0,0,0,0,0,0,50,100,350,800,1100,1500,2000,2300,2350,2123,1800,1304,754,300,213,0,0,0]

def fetch_report(prediction, all_contribution, mean_contribution,listDevices):
    time.sleep(5)
    return 'dear carlos , we hope this message finds you well . as your solar power provider , we wanted to provide you with a report on the predicted solar power generation from your solar panels on october 15th , 2024 at 1800 . based on our analysis , we predict that the solar power generated will be medium , with a total of 5 . 0 kilo watts . this is due to the negative impact of the solar radiation and hourly precipitation predicted value . based on this prediction , we recommend that you avoid using your appliances heater , as it might exceed the generated power . however , we advise against using your washing machine and dishwasher as they will definitely exceed the generated power . we hope this information is helpful and please do not hesitate to contact us if you have any further questions or concerns . best regards , tenergito'


sl.set_page_config(page_title="Tenergito",layout="wide")
sl.header("""Tenergito""")

url = 'http://192.168.1.1'


left, right = sl.columns(2)

left.write("""
### generation forecast
""")

forecast = fetch_forecast(url)
forecast = pd.DataFrame(forecast, columns = ['generation'])
left.line_chart(forecast)

name = right.text_input("Name:")

devices = ["AC", "DishWasher", "WashingMachine", "Dryer", "WaterHeater" , "TV", "Microwave", "Kettle", "Lighting", "Refrigerator"]
used_devices = right.multiselect('Devices to use (first has the most priority and the last the least priority):',devices)

ordered_devices = []
for i, device in enumerate(used_devices):
    ordered_devices.append({"name": device, "priority": i+1})

datetime = right.date_input('What time?')
hour = right.time_input('To what hour do you want advice?')

if right.button("Generate Report"):
    prediction, all_contribution, mean_contribution,listDevices = getContributions(ordered_devices, str(hour)[:-3])
    sl.write(prediction, all_contribution, mean_contribution,listDevices)

sl.write('### report')
with sl.spinner(text='Generating report..'):
    sl.write(fetch_report(prediction, all_contribution, mean_contribution,listDevices))

