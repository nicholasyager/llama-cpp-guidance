from pathlib import Path
import guidance
from guidance._program import Program
from llama_guidance.llm import LlamaCpp

# set the default language model used to execute guidance programs
guidance.llm = LlamaCpp(
    model_path=Path("../../llm/models/gguf/llama-2-13b.Q4_K_M.gguf"),
    n_gpu_layers=1,
    n_threads=8,
)

# guidance.llm = guidance.llms.OpenAI("text-davinci-003")

# define a guidance program that adapts a proverb
# program: Program = guidance(
#     """Tweak this proverb to apply to model instructions instead.

# {{proverb}}
# - {{book}} {{chapter}}:{{verse}}

# UPDATED
# Where there is no guidance{{gen 'rewrite' stop="\\n-"}}
# - GPT {{#select 'chapter'}}9{{or}}10{{or}}11{{/select}}:{{gen 'verse'}}""",
# )

# # execute the program on a specific proverb
# executed_program = program(
#     proverb=(
#         "Where there is no guidance, a people falls,\nbut in an abundance of counselors"
#         " there is safety."
#     ),
#     book="Proverbs",
#     chapter=11,
#     verse=14,
# )

# print(executed_program["chapter"])

# prompt = guidance(
#     """Is the following sentence offensive? Please answer with a single word, either "Yes", "No", or "Maybe".
# Sentence: {{example}}
# Answer:{{#select "answer" logprobs='logprobs'}} Yes{{or}} No{{or}} Maybe{{/select}}"""
# )
# prompt = prompt(example="I hate tacos")


# print(prompt)


# we can use the {{gen}} command to generate text from the language model
# note that we used a ~ at the start of the command tag to remove the whitespace before it (just like in Handlebars)

# we can pre-define valid option sets
valid_weapons = ["sword", "axe", "mace", "spear", "bow", "crossbow"]
valid_armor = ["leather", "chainmail", "plate"]
#    "items": [{{#geneach 'items' num_iterations=3}}
#         "{{gen 'this'}}",{{/geneach}}
#     ]

# define the prompt
program = guidance(
    """The following is a character profile for an RPG game in JSON format.
```json
{
    "description": "{{description}}",
    "name": "{{gen 'name' temperature=0.9 max_tokens=12 stop='"'}}",
    "age": {{gen 'age' pattern='[0-9]+' stop=','}},
    "armor": "{{select 'armor' options=valid_armor}}",
    "weapon": "{{select 'weapon' options=valid_weapons}}",
    "class": "{{gen 'class' stop='"'}}",
    "mantra": "{{gen 'mantra' stop='"'}}",
    "strength": {{gen 'strength' pattern='[0-9]+' stop=','}},
 
}```""",
    logging=True,
)

# execute the prompt
output = program(
    description="A quick and nimble fighter.",
    valid_weapons=valid_weapons,
    valid_armor=valid_armor,
)
print(output)
print(output.variables())
