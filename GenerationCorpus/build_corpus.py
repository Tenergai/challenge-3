from get_completion import gen_question, gen_completion

import random

import pandas as pd
import numpy as np

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

years = [i for i in range(2023, 2024+1)]

category = {
    "small": 35,
    "medium": 30,
    "big": 25
}

devices = ["air conditioner", "washing machine", "dishwasher", "water heater", "heater"]

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

    hour = random.randint(6, 20)

    return selected_month + ' ' + str(selected_day) + suffix + ' ' + str(selected_year) + ', ' + str(hour) + "o\'clock", hour

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
        possible_explations = [
            ("solar radiation", "significant negative impact"),
        ]
    elif solar_power_cat == "low":
        possible_explations = [
            ("solar radiation", "negative impact"),
            ("hourly precipitation", "significant negative impact")
        ]
    elif solar_power_cat == "medium":
        possible_explations = [
            ("hourly precipitation", "negative impact"),
            ("temperature", "negative impact")
        ]
    elif solar_power_cat == "high":
        possible_explations = [
            ("solar radiation", "positive contribution"),
            ("temperature", "positive contribution"),
            ("dewpoint", "positive contribution")
        ]
    elif solar_power_cat == "very high":
        possible_explations = [
            ("solar radiation", "significant positive contribution"),
            ("temperature", "positive contribution")
        ]
    
    explanation_tokens = [random.choice(possible_explations)]
    possible_explations.pop(possible_explations.index(explanation_tokens[0]))
    if random.randint(0,1) == 1 and len(possible_explations) != 0:
        explanation_tokens.append(random.choice(possible_explations))

    if len(explanation_tokens) == 2:
        explanation = f" due to {explanation_tokens[0][0]} predicted value having a {explanation_tokens[0][1]} and {explanation_tokens[1][0]} predicted value having a {explanation_tokens[1][1]}"
    else:
        explanation = f" due to {explanation_tokens[0][0]} predicted value having a {explanation_tokens[0][1]}"

    return explanation, explanation_tokens


def gen_text_formats():
    date, hour = gen_date()
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

    if hour >= 6 and hour < 10:
        solar_power_cat, solar_power_nums = random.choice(list(solar_power.items())[0:4])
    elif hour >= 10 and hour < 17:
        solar_power_cat, solar_power_nums = random.choice(list(solar_power.items())[2:5])
    elif hour >= 17 and hour < 19:
        solar_power_cat, solar_power_nums = random.choice(list(solar_power.items())[1:4])
    elif hour >= 19 and hour <= 20:
        solar_power_cat, solar_power_nums = random.choice(list(solar_power.items())[0:1])

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

    explanation, explanation_tokens = gen_explanation(solar_power_cat)

    format_text = {
        "date": date,
        "client": name,
        "solar_power_cat": solar_power_cat,
        "solar_power_num": str(solar_power_num),
        "explanation": explanation,
        "use_devices": select_devices_text["accept"],
        "uncertain_devices": select_devices_text["uncertain"],
        "nouse_devices": select_devices_text["unaccept"]
    }

    return format_text, selected_devices, explanation_tokens

def save_file(filename, body):
    with open(filename, "w") as file:
        file.write(body)

def del_sure(body):
    if body[:].find("\n") == 0:
            body = body.replace(body[0:2], "", 1)

    if body[0:4] == "Sure":
        id_final_phrase = body[:].find("\n")
        body = body.replace(body[0:id_final_phrase], "", 1)
        body = body.replace(body[0:2], "", 1)

    return body

def init_meta_data():
    df = pd.DataFrame(
        columns=[
            "date", 
            "client", 
            "solar_power_cat", 
            "solar_power_num",
            "explanation",
            "use_devices",
            "uncertain_devices",
            "nouse_devices",
            "filename"
        ]
    )
    df.to_csv("./GenerationCorpus/metadata.csv")

def gen_reports_batch(batch, start_point, batch_size, df):
    print(f"Progress on batch {batch+1}: ", end="")
    pipes = "|||"
    for b in range(start_point-1, batch_size):
        format_text, raw_devices, explanation_tokens = gen_text_formats()
        use_devices = "#" + "&".join(raw_devices["accept"])
        uncertain_devices = "#" + "&".join(raw_devices["uncertain"])
        nouse_devices = "#" + "&".join(raw_devices["unaccept"])
        exp_tokens = "#" + "&".join(map(lambda x: str(x), explanation_tokens))

        question = gen_question(format_text)
        
        raw_completion = gen_completion(question)
        body = raw_completion["choices"][0]["message"]["content"]
        completion = del_sure(body)

        loc = "./GenerationCorpus/corpus/"
        filename = f"report_{batch+1}_{b+1}" + ".txt"
        save_file(loc + filename, completion)
        s_add = pd.DataFrame(np.array([[
            format_text["date"],
            format_text["client"],
            format_text["solar_power_cat"],
            format_text["solar_power_num"],
            exp_tokens,
            use_devices,
            uncertain_devices,
            nouse_devices,
            filename
        ]]), columns=[
            "date", 
            "client", 
            "solar_power_cat", 
            "solar_power_num",
            "explanation",
            "use_devices",
            "uncertain_devices",
            "nouse_devices",
            "filename"
        ])

        if "Unnamed: 0" in list(df.columns):
            del df["Unnamed: 0"]

        if "Unnamed: 0.1" in list(df.columns):
            del df["Unnamed: 0.1"]

        if "Unnamed: 0.2" in list(df.columns):
            del df["Unnamed: 0.2"]

        if "Unnamed: 0.3" in list(df.columns):
            del df["Unnamed: 0.3"]

        if "Unnamed: 0.4" in list(df.columns):
            del df["Unnamed: 0.4"]

        if "Unnamed: 0.5" in list(df.columns):
            del df["Unnamed: 0.5"]

        df = pd.concat([df, s_add], ignore_index=True)
        df.to_csv("./GenerationCorpus/metadata.csv")

        print(f"{pipes}", end="")
    
    print()

def main():
    n_batches = 100
    batch_size = 16
    start_batch = int(input("Batch you want to start: "))
    start_point = int(input("Point in the batch to start: "))
    if start_batch == 1 and start_point == 1:
        init_meta_data()

    df = pd.read_csv("./GenerationCorpus/metadata.csv")
    for batch in range(start_batch-1, n_batches):
        proceed = input(f"Do you want to proceed to the next batch (number {batch+1})? (y/(any key)): ")
        if proceed == "y":
            gen_reports_batch(batch, start_point, batch_size, df)
        else:
            break

        start_point = 1


if __name__ == "__main__":
    main()