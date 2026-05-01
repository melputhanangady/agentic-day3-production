from prompt_manager import PromptManager

prompt_manager = PromptManager()

# prompt_data = prompt_manager.load_prompt("customer_support", "v1.0.0")
# print("Prompt Data: ", prompt_data)
print("--------------------------------")
prompt_data = prompt_manager.load_prompt("prompts","support_agent_v1")
prompt = prompt_manager.compile_prompt(prompt_data, "I have a problem with my mobile phone")
print("Prompt: ", prompt)
print("--------------------------------")