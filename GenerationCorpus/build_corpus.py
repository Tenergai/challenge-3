from get_completion import gen_question, gen_completion

format_text = {
            "date": "November 22nd of 2020",
            "client": "Pedro",
            "solar_power_cat": "low",
            "solar_power_num": "0.5 kW",
            "explanation": "temperature is moderate and solar irradiantion is low",
            "use_devices": "air conditioner and water heater",
            "uncertain_devices": "dishwasher, washing machine and television",
            "nouse_devices": ""
        }


def save_file(filename, body):
    with open(filename, "w") as file:
        file.write(body)

def del_sure(body):
    if body[2:6] == "Sure":
        id_final_phrase = body[2:].index("\n")
        body[2:id_final_phrase] = ""

    print(body)
    return body

def main():
    question = gen_question(format_text)
    print(question)
    completion = gen_completion(question)

    loc = "./GenerationCorpus/corpus/"
    body = completion["choices"][0]["message"]["content"]
    print(body)

    save_file(loc + "example1.txt", body)

main()