from get_completion import gen_question, gen_completion

import random

months = {
    "January": [i for i in range(1,31+1)],
    "February": [i for i in range(1,28+1)],
    "March": [i for i in range(1,31+1)],
    "April": [i for i in range(1,30+1)],
    "May": [i for i in range(1,31+1)],
    "June": [i for i in range(1,30+1)],
    "July": [i for i in range(1,31+1)],
    "August": [i for i in range(1,31+1)],
    "September": [i for i in range(1,30+1)],
    "October": [i for i in range(1,31+1)],
    "November": [i for i in range(1,30+1)],
    "December": [i for i in range(1,31+1)]
}

years = [i for i in range(2022, 2023+1)]

category = {
    "small": 35,
    "medium": 30,
    "big": 25
}

devices = ["air conditioner", "washing machine", "dishwasher", "water heater", "heater"]

features = "TemperatureC,DewpointC,PressurehPa,WindDirectionDegrees,WindSpeedKMH,WindSpeedGustKMH,Humidity,HourlyPrecipMM,dailyrainMM,SolarRadiationWatts_m2"

def gen_date():
    items = list(months.items())
    selected_month_day  = random.choice(items)
    selected_month = selected_month_day[0]
    selected_day = random.choice(selected_month_day[1])
    selected_year = random.choice(years)

    suffix = ""
    if str(selected_day)[-1] == "1":
        suffix = "st"
    elif str(selected_day)[-1] == "2":
        suffix = "nd"
    elif str(selected_day)[-1] == "3":
        suffix = "rd"
    else: 
        suffix = "th"

    return selected_month + ' ' + str(selected_day) + suffix + ' ' + str(selected_year)

def select_devices(ny, ns, no):
    # maximum values accepted
    n_acceptance = {"accept": ny, "uncertain": ns, "unaccept": no}
    devices_prc = devices.copy()
    accepted_devices = {}
    smooth_param = 1/10
    for key, value in n_acceptance.items():
        accepted_devices[key] = []
        for v in range(value):
            if random.random() <= 0.5 + v*smooth_param:
                try:
                    accepted_devices[key].append(devices_prc.pop(devices_prc.index(random.choice(devices_prc))))
                except IndexError:
                    print("No more to be poped.")
    
    return accepted_devices

def gen_explanation(solar_power_cat):
    if solar_power_cat == "very low":
        pass
    elif solar_power_cat == "low":
        pass
    elif solar_power_cat == "medium":
        pass
    elif solar_power_cat == "high":
        pass
    elif solar_power_cat == "very high":
        pass


def gen_text_formats():
    date = gen_date()
    name = random.choice(["Pedro", "Luís", "Franciso", "Constantino", "Rafael", "Letícia"])
    
    coef = random.choice(list(category.values()))
    func_rescale = lambda x: round(x/coef, 3)
    solar_power = {
        "very low": list(map(func_rescale, [i for i in range(1, 20+1)])),
        "low": list(map(func_rescale, [i for i in range(21, 40+1)])),
        "medium": list(map(func_rescale, [i for i in range(41, 60+1)])),
        "high": list(map(func_rescale, [i for i in range(61, 80+1)])),
        "very high": list(map(func_rescale, [i for i in range(81, 100+1)]))
    }
    solar_power_cat, solar_power_nums = random.choice(list(solar_power.items()))
    solar_power_num = random.choice(solar_power_nums)
    
    selected_devices = {}
    if solar_power_cat == "very low":
        # accept 0, 1
        # uncertain 0, 1
        # unaccept 3, 4, 5
        selected_devices = select_devices(0,1,4)
    elif solar_power_cat == "low":
        # accept: 0, 1
        # uncertain 1, 2
        # unaccept: 2, 3, 4
        selected_devices = select_devices(1,2,3)
    elif solar_power_cat == "medium":
        # accept: 1, 2
        # uncertain: 1, 2
        # unaccept: 1, 2, 3
        selected_devices = select_devices(2,1,2)
    elif solar_power_cat == "high":
        # accept: 2, 3 
        # uncertain: 1, 2
        # unaccept: 0, 1, 2
        selected_devices = select_devices(3,1,1)
    elif solar_power_cat == "very high":
        # accept: 3, 4
        # uncertain: 1, 2
        # unaccept: 0, 1
        selected_devices = select_devices(4,1,0)

    select_devices_text = {}
    for cat in selected_devices:
        select_devices_text[cat] = ''
        count = 0
        for d in selected_devices[cat]:
            count += 1
            if count > 1 and selected_devices[cat].index(d) == len(selected_devices[cat]) - 1:
                select_devices_text[cat] += 'and ' + d
            elif count == 1 and selected_devices[cat].index(d) == len(selected_devices[cat]) - 1:
                select_devices_text[cat] += d
            elif selected_devices[cat].index(d) == len(selected_devices[cat]) - 2:
                select_devices_text[cat] += d + ' '
            else:
                select_devices_text[cat] += d + ', '

    format_text = {
        "date": date,
        "client": name,
        "solar_power_cat": solar_power_cat,
        "solar_power_num": str(solar_power_num),
        "explanation": "",
        "use_devices": select_devices_text["accept"],
        "uncertain_devices": select_devices_text["uncertain"],
        "nouse_devices": select_devices_text["unaccept"]
    }

    return format_text

def save_file(filename, body):
    with open(filename, "w") as file:
        file.write(body)

def del_sure(body):
    if body[2:6] == "Sure":
        body = body.replace(body[0:2], "", 1)
        id_final_phrase = body[:].find("\n")
        body = body.replace(body[0:id_final_phrase], "", 1)
        body = body.replace(body[0:2], "", 1)
    else:
        body = body.replace(body[0:2], "", 1)

    return body

def main():
    format_text = gen_text_formats()
    question = gen_question(format_text)
    print(question)
    completion = gen_completion(question)

    loc = "./GenerationCorpus/corpus/"
    body = completion["choices"][0]["message"]["content"]
    print(body)

    body = del_sure(body)
    print(body)

    save_file(loc + "example1.txt", body)

format_text = gen_text_formats()
question = gen_question(format_text=format_text)

print(format_text)
print(question)