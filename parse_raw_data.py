from loadData import clean_dialogue
from create_dialogue_dataset import parse_dialogue
from create_gamestate_dataset import parse_game_state

# Cleans the dialogue data and adds player tokens if player_token is True
clean_dialogue(player_token=True)

# Creates complete dialogue dataset with or w/o dialogue history, addes EOS token if eos_tag is True
parse_dialogue(history=True, eos_tag=True)
parse_dialogue(history=False, eos_tag=True)

# Creates complete gamestate dataset synced with dialogue data
parse_game_state()
