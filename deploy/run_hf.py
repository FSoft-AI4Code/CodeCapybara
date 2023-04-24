from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import gradio as gr
import torch
import fire

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
total_queries = 0
def load_model(model_path: str, tokenizer_path: str, load_8bit: bool = False):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    generator = LlamaForCausalLM.from_pretrained(model_path).to(device).eval()


    tokenizer.pad_token_id = generator.config.pad_token_id = 0
    tokenizer.bos_token_id = generator.config.bos_token_id = 1
    tokenizer.eos_token_id = generator.config.eos_token_id = 2
    return tokenizer, generator

def main(
        model_path: str = '/datadrive/minhnh46/codecmal/cp1/cp1/cp1/llama-7b/',
        tokenizer_path: str = '/datadrive/minhnh46/codecmal/cp1/cp1/cp1/tokenizer/',
        load_8bit: bool = False
):
    global total_queries
    tokenizer, generator = load_model(model_path, tokenizer_path, load_8bit = load_8bit)

    def generate(input:str, model_name, temperature: float, top_p: float, top_k: int, max_new_tokens: int):
        global total_queries
        generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        #print('input', input)

        input_ids = tokenizer.encode(input, return_tensors = 'pt').to(device)
        total_queries += 1
        print('NUM QUERY:', total_queries)
        #print(input_ids.shape, input_ids)
        with torch.inference_mode():
            generation_output = generator.generate(
                input_ids = input_ids,
                generation_config = generation_config,
                max_new_tokens = max_new_tokens,
                top_k = top_k,
                top_p = top_p
            )
        result = generation_output[0].tolist()[1:]
        try:
            result = result[: result.index(tokenizer.eos_token_id)]
        except: pass
        yield tokenizer.decode(result)



    gr.Interface(
        fn = generate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Write an instruction...",
            ),
            gr.Radio(["Fully Fine-tuning"], value = 'Fully Fine-tuning', llabel = 'Model'),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=512, step=1, value=128, label="Max New Tokens"
            )
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title= "CodeCamel",
        description="Our demonstration for CodeCamel. CodeCamel is fine-tuned from LLaMA on our curated dataset.",  # noqa: E501
    ).queue().launch(server_name="localhost", share = True)
if __name__ == "__main__":
    fire.Fire(main)