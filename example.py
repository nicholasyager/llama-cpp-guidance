import random
from pathlib import Path

import guidance
from loguru import logger

from llama_cpp_guidance.llm import LlamaCpp

logger.enable("llama_cpp_guidance")

# set the default language model used to execute guidance programs
guidance.llm = LlamaCpp(
    model_path=Path("../../llm/models/gguf/pygmalion-2-13b.Q4_K_M.gguf"),
    n_gpu_layers=1,
    n_threads=8,
    seed=random.randint(0, 1000000),
)

# we can use the {{gen}} command to generate text from the language model
# note that we used a ~ at the start of the command tag to remove the whitespace before
#  it (just like in Handlebars)

# we can pre-define valid option sets
valid_weapons = ["sword", "axe", "mace", "spear", "bow", "crossbow"]
valid_armor = ["leather", "chainmail", "plate"]

name_program = guidance(
    """The following is a character profile for an RPG game in JSON format.
```json
{
    "description": "{{description}}",
    "first_name": "{{gen 'first_name' temperature=0.8 max_tokens=12 stop=[' ', '"']}}",
    "last_name": "{{gen 'last_name' temperature=0.8 max_tokens=12 stop=[' ', '"']}}",
}```""",
    logging=True,
)

name_output = name_program(
    description="A quick and nimble fighter.",
)


# define the prompt
program = guidance(
    """The following is a character profile for an RPG game in JSON format.
```json
{
    "description": "{{description}}",
    "name": "{{ name }}",
    "age": {{gen 'age' pattern='[0-9]+' stop=',' temperature=1}},
    "armor": "{{select 'armor' logprobs='logprobs' options=valid_armor}}",
    "weapon": "{{select 'weapon' options=valid_weapons}}",
    "class": "{{gen 'class' stop='"'}}",
    "mantra": "{{gen 'mantra' temperature=0.8 stop='"'}}",
    "strength": {{gen 'strength' pattern='[0-9]+' stop=','}},
    "items": [{{#geneach 'character_items' num_iterations=3}}
        "{{gen 'this' stop='"' temperature=0.95}}",{{/geneach}}
    ]
}```""",
    logging=True,
)

# execute the prompt
output = program(
    description="A quick and nimble rouge that murdered a lich using a crossbow",
    valid_weapons=valid_weapons,
    valid_armor=valid_armor,
    name=name_output["first_name"] + " " + name_output["last_name"],
)
print(output)
