from openai import OpenAI

from app.config import settings


# OpenAI client
client = OpenAI(
    api_key=settings.OPENAI_API_KEY
)


# Standard answer generation
def generate_answer(
    query,
    context
):

    response = client.chat.completions.create(

        model="gpt-3.5-turbo",

        messages=[
            {
                "role": "system",
                "content": f"""
You are a legal AI assistant.

Answer ONLY using the provided legal context.

If the answer is not present in the context,
say you could not find it in the document.

Context:
{context}
"""
            },
            {
                "role": "user",
                "content": query
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# Streaming answer generation
def stream_answer(
    query,
    context
):

    response = client.chat.completions.create(

        model="gpt-3.5-turbo",

        messages=[
            {
                "role": "system",
                "content": f"""
You are a legal AI assistant.

Answer ONLY using the provided legal context.

If the answer is not present in the context,
say you could not find it in the document.

Context:
{context}
"""
            },
            {
                "role": "user",
                "content": query
            }
        ],

        temperature=0.2,

        stream=True
    )

    for chunk in response:

        delta = chunk.choices[0].delta

        if delta.content:

            yield delta.content