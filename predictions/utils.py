import pandas as pd
import numpy as np


def match_teams(team, abbr_list):
    # Try direct match
    if team in abbr_list.values:
        return team
    # Try lower-case match
    for abbr in abbr_list:
        if team.lower() == abbr.lower():
            return abbr
    # Try partial match
    for abbr in abbr_list:
        if team.lower() in abbr.lower():
            return abbr
    return team


def american_odds_to_probability(odds):
    odds = int(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

