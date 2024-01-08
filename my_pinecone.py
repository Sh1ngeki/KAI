import uuid
import openai
from openai import OpenAI
import pinecone
import firebase_admin
from firebase_admin import db, credentials
import json
from uuid import uuid4

openai_api_key = 'api_key'
pinecone_api_key = 'pinecone_key'
chatbot_index_name = 'test'

client = OpenAI(api_key=openai_api_key)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def gpttt_embedding(content, model='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = client.embeddings.create(
        model = model,
        input = content
    )
    vector = response['choices'][0]['text'].strip()

    return vector


def gpt_completion(prompt, model='gpt-3.5-turbo', stop=['USER:', 'KOTIKO:']):
    retry = 0
    max_retry = 5
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    while True:
        try:
            response = client.Completion.create(
                prompt=prompt,
                model=model,
                stop=stop
            )
            text = response['choices'][0]['text'].strip()
            return text
        except Exception as error:
            retry += 1
            if retry >= max_retry:
                return f"GPT3 error: {error}"
            print('Error communicating with OpenAI:', error)


def load_conversation(result):
    result_list = []
    for m in result['matches']:
        info = load_json(f'idek/{m["id"]}.json')  # Changed formatting for better readability
        result_list.append(info)
    ordered = sorted(result_list, key=lambda d: d['time'], reverse=False)
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()


if __name__ == '__main__':
    openai.api_key = openai_api_key
    pinecone.init(
        api_key=pinecone_api_key,
        environment='gcp-starter'
    )
    index = pinecone.Index(chatbot_index_name)
    while True:
        payload = []
        question = input('Studentis kitxva: ')
        message = question
        vector = gpttt_embedding(message)
        unique_id = str(uuid4())
        metadata = {'speaker': 'USER', 'message': message, 'uuid': unique_id}
        save_json(f'idek/{unique_id}.json', metadata)  # Changed formatting for better readability
        payload.append({'id': unique_id, 'embedding': vector})

        results = index.query(vector=vector, top_k=40)
        conversation = load_conversation(results)
        answer = open_file('conversation.txt').replace('<<Answer>>', conversation).replace('<<Question>>', question)

        output = gpt_completion(answer)
        message = output
        vector = gpttt_embedding(message)
        unique_id = str(uuid4())
        metadata = {'speaker': 'Kotiko', 'message': message, 'uuid': unique_id}
        save_json(f'idek/{unique_id}.json', metadata)  # Changed formatting for better readability
        payload.append({'id': unique_id, 'embedding': vector})
        index.upsert(payload)
        print(f'\n\nKotiko: {output}')
