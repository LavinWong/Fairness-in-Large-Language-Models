from openai import OpenAI
from api_key import TOGETHERAPI_KEY

class LLAMA2_MODEL:
  def __init__ (self):
      self.client = OpenAI(
        api_key=TOGETHERAPI_KEY,
        base_url='https://api.together.xyz/v1',
      )

  def ask(self, chat, model = "meta-llama/Llama-2-70b-chat-hf"):
      chat_response = self.client.chat.completions.create(
          model = model,
          messages=chat,
          top_p = 0.8,
          max_tokens = 200,
          stream = False,
      )

      return chat_response.choices[0].message.content