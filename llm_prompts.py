##### LLM Predictions Prompting #####
# Incoming text is.... Coordinates XXX, Places of Interest, <TASK> (On a Scale from 0.0 to 9.9): 
# basic prompting 
only_coords_prompt = """
Given coordinates predict the likelihood of finding a {SPECIES} (On a Scale from 0.0 to 9.9). Provide a single number as a likelihood estimate.

{COORDINATES}

Predict the likelihood of finding a {SPECIES} (On a Scale from 0.0 to 9.9):

"""  

basic_prompt = """
Given the coordinates, address, and nearby places, predict the likelihood of finding a {SPECIES} (On a Scale from 0.0 to 9.9). Provide a single number as a likelihood estimate.

{NEW_LOCATION}

Predict the likelihood of finding a {SPECIES} (On a Scale from 0.0 to 9.9):
"""  

############# Similarity Prompting ########

coordinate_prompt = """
Given the coordinates, address, and nearby places, predict the types of species found at this location.

{NEW_LOCATION}
"""

# TODO
coordinate_temporal_prompt = """
Given the coordinates, date, address, and nearby places, predict the types of species found at this location.

{NEW_LOCATION}
"""

species_prompt = "Predict the locations where you would find {SPECIES}."



#########
# In context prompting
# incontext_prompt = """
# Given the coordinates, address, and nearby places, predict the likelihood of finding a {SPECIES} (On a Scale from 0.0 to 9.9).

# Example:  

# Coordinates: 38.6270, -90.1994 
# Address: "Gateway Arch, St. Louis, Missouri, United States"  
# Nearby Places:  
# "  
# 2.0 km West: Downtown St. Louis  
# 3.2 km North: Hyde Park  
# 4.5 km East: East St. Louis  
# 5.1 km South: Lafayette Square  
# "  
# How likely are you to find a {SPECIES} here (On a Scale from 0.0 to 9.9): 2.3  

# Now, predict the likelihood for the following location:  


# {NEW_LOCATION}

# Predict the likelihood of finding a {SPECIES} (On a Scale from 0.0 to 9.9):

# """  


# Adding expert prompting @ beginning
# expert_prompting = "You are an expert biologist estimating the likelihood of finding a {SPECIES} at various locations.  "