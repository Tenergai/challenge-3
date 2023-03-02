import openai

"""
    format_text structure:
        {
            "client": name,
            "solar_power_cat": low/moderate/high,
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
        raw_question = f.read().split("#")

    for i in range(1, len(raw_question), 2):
        raw_question[i] = format_text[raw_question[i]]

    return "".join(raw_question)


def gen_completion(question):
    openai.api_key = "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"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )

    return completion

format_text = {
            "client": "Pedro",
            "solar_power_cat": "high",
            "solar_power_num": "1",
            "explanation": "moderate ambient temperature (20 degrees celsius)",
            "use_devices": "washing machine",
            "uncertain_devices": "dishwasher and air conditioner",
            "nouse_devices": "none"
        }

question = gen_question(format_text)
completion = gen_completion(question)
print(completion)