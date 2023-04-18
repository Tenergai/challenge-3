import streamlit as sl
import requests
import pandas as pd
import time
import random
from shapModel.shapModel import getContributions
import shapModel.nlp_serialized as nlp

devices = ["AC", "DishWasher", "WashingMachine", "WaterHeater", "Heater", "Dryer", "TV", "Microwave", "Kettle", "Lighting", "Refrigerator"]
months = ["January","February","March","April","May","June","July","August","September","October","November","December"]

def fetch_forecast():
    hours = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    result = []
    for hour in hours:
        prediction, all_contribution, mean_contribution,listDevices = getContributions(ordered_devices, hour+':00')
        result.append([prediction, all_contribution, mean_contribution,listDevices])
    return result

def order_devices(used_devices):
    ordered_devices = []
    for i, device in enumerate(used_devices):
        ordered_devices.append({"name": device, "priority": i+1})
    return ordered_devices

def fetch_report(datetime, hour, name, prediction, all_contribution, mean_contribution,ordered_devices):
    priorities = ''
    for device in devices[:5]:
        added = False
        for ordered_device in ordered_devices:
            if ordered_device['name'] == device:
                priorities += ' ' + str(ordered_device['priority'])+'.0'
                added = True
        if not added:
            priorities += ' 0.0'
    #solar_power_cat,solar_power_num,feat1,feat2,contri1,contri2,air conditioner,washing machine,dishwasher,water heater,heater
    text = str(datetime.day) + ' ' + months[datetime.month-1] + ' ' + str(datetime.year) + ' ' + str(hour.hour) + ' ' + name
    text = text + ' ' + 'medium 8.0 hourly precipitation temperature negative impact negative impact' + ' ' +priorities
    #sl.write(text)
    return 'lalalla'
    #loaded_model = nlp.load_model()
    #return nlp.translate(loaded_model, [text])


sl.set_page_config(page_title="Tenergito",layout="wide")
sl.header("""Tenergito""")
left, right = sl.columns(2)

left.write("""
### generation forecast
""")
left.write(fetch_forecast())

name = right.text_input("Name:")
used_devices = right.multiselect('Devices to use (first has the most priority and the last the least priority):',devices)
datetime = right.date_input('What time?')
hour = right.time_input('To what hour do you want advice?')


if right.button("Generate Report"):
    ordered_devices = order_devices(used_devices)
    prediction, all_contribution, mean_contribution,listDevices = getContributions(ordered_devices, str(hour)[:-3])
    sl.write(prediction, all_contribution, mean_contribution,listDevices)
    sl.write('### report')
    with sl.spinner(text='Generating report..'):
        report = fetch_report(datetime, hour, name, prediction, all_contribution, mean_contribution,ordered_devices)
    text_element = sl.empty()
    
    wr = ''
    for char in report:
        wr = wr + char
        time.sleep(random.randrange(0,5)/100)
        text_element.write(wr)        
else:
    sl.write('### report')
    sl.write('no report to be generated')

