import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def main():
    messages = [
      {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
      {"role": "system", "name":"example_user", "content": "New synergies will help drive top-line growth."},
      {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
      {"role": "system", "name":"example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
      {"role": "system", "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
      {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
    ]

    model = "gpt-3.5-turbo-0301"

    print(f"{num_tokens_from_messages(messages, model)} prompt tokens counted.")
# Should show ~126 total_tokens
