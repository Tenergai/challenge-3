import openai

def gen_completion(question):
    openai.api_key = "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"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )

    return completion

question = """
    I am trying to build small reports for my clients relative to generated solar power prediction from their solar panels.
    Take for instance that, for my client Luís Faria the solar power generated will be moderate (1 kilo Watts) due to the forecast of high temperature (30 degrees celsius) and high solar irradiance (200 Watts per meter square).
    He can use his washing machine and air conditioner, his dishwasher might exceed the generated power, and his water heater will exceed the generated power. 
    Can you build me a small report for my client with this information?
"""

completion = gen_completion(question)
print(completion)