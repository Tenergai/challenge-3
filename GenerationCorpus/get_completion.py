import openai

"""
    format_text structure:
        {
            "date": xnd of month of year, hour 
            "client": name,
            "solar_power_cat": verylow/low/moderate/high/veryhigh,
            "solar_power_num": number,
            "explanation": text,
            "use_devices": text,
            "uncertain_devices": text,
            "nouse_devices": text
        }

"""

def gen_question(format_text):
    raw_question = ""
    with open("./GenerationCorpus/resources/question_template.txt", "r") as f:
        raw_question = f.read()

    section_divided = raw_question.split("$")

    if format_text["use_devices"] == "":
        section_divided[1] = ""
    if format_text["uncertain_devices"] == "":
        section_divided[2] = ""
    if format_text["nouse_devices"] == "":
        section_divided[3] = ""

    token_divided = "".join(section_divided).split("#")

    for i in range(1, len(token_divided), 2):
        token_divided[i] = format_text[token_divided[i]]

    return "".join(token_divided)

def gen_completion(question):
    openai.api_key = "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"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )

    return completion