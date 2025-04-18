import openai

openai.api_key = "enter open ai key here"

def generate_guidelines(aqi_value: int) -> str:
    if aqi_value <= 1:
        return "Air quality is safe. No precautions needed. Enjoy your day!"

    prompt = (
        f"The air quality index (AQI) is currently {aqi_value}, which is above the safe limit. "
        "Please provide simple and effective safety guidelines for the public during this high pollution level."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an environmental safety advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Failed to generate guidelines: {e}"
