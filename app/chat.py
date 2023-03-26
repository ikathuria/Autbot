from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

chat_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/blenderbot-400M-distill"
)
chat_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/blenderbot-400M-distill"
)


def generate_response(text, emotion):
    prompt = f"{text}. My emotion is {emotion}."
    input_ids = chat_tokenizer.encode([prompt], return_tensors="pt")

    chat_response_ids = chat_model.generate(**input_ids)

    response = chat_tokenizer.decode(chat_response_ids[0], skip_special_tokens=True)
    return response


