##### LLM Predictions Prompting #####
# Incoming text is.... Coordinates XXX, Places of Interest, <TASK> (On a Scale from 0.0 to 9.9): 
# basic prompting 
basic_prompt = """
Given the coordinates, address, and nearby places, predict the likelihood of finding a {SPECIES} on a scale from 0.0 to 9.9. 

Predict the likelihood for the following location:  

{NEW_LOCATION}

"""  

# Adding expert prompting @ beginning
expert_prompting = "You are an expert biologist estimating the likelihood of finding a {SPECIES} at various locations.  "

# In context prompting
incontext_prompt = """
Given the coordinates, address, and nearby places, predict the likelihood of finding a {SPECIES} on a scale from 0.0 to 9.9.  

Examples:  

Coordinates: (38.6270, -90.1994)  
Address: "Gateway Arch, St. Louis, Missouri, United States"  
Nearby Places:  
"  
2.0 km West: Downtown St. Louis  
3.2 km North: Hyde Park  
4.5 km East: East St. Louis  
5.1 km South: Lafayette Square  
"  
How likely are you to find a {SPECIES} here (On a Scale from 0.0 to 9.9): 2.3  

---

Coordinates: (39.0997, -94.5786)  
Address: "Kansas City, Jackson County, Missouri, United States"  
Nearby Places:  
"  
3.5 km North: North Kansas City  
4.8 km East: Independence Avenue  
6.0 km South-West: Westside  
7.2 km West: Kansas River  
"  
How likely are you to find a {SPECIES} here (On a Scale from 0.0 to 9.9): 5.7  

---

Now, predict the likelihood for the following location:  


{NEW_LOCATION}

"""  

# In context prompting with date
incontext_temporal_prompt = """
Given the coordinates, date, address, and nearby places, predict the likelihood of finding a {SPECIES} on a scale from 0.0 to 9.9.  

Examples:  

Coordinates: (38.6270, -90.1994)  
Date: 09/07/2022
Address: "Gateway Arch, St. Louis, Missouri, United States"  
Nearby Places:  
"  
2.0 km West: Downtown St. Louis  
3.2 km North: Hyde Park  
4.5 km East: East St. Louis  
5.1 km South: Lafayette Square  
"  
How likely are you to find a {SPECIES} here (On a Scale from 0.0 to 9.9): 2.3

---

Coordinates: (39.0997, -94.5786)  
Date: 03/10/2024
Address: "Kansas City, Jackson County, Missouri, United States"  
Nearby Places:  
"  
3.5 km North: North Kansas City  
4.8 km East: Independence Avenue  
6.0 km South-West: Westside  
7.2 km West: Kansas River  
"  
How likely are you to find a {SPECIES} here (On a Scale from 0.0 to 9.9): 5.7  

---

Now, predict the likelihood for the following location:  


{NEW_LOCATION}

"""  

############# Similarity Prompting ########

coordinate_prompt = """
You are an expert biologist estimating the likelihood of finding different species at a given locations.  Given the coordinates, address, and nearby places, predict the types of species found at this location.

{NEW_LOCATION}
"""

species_prompt = """
You are an expert biologist. Think about what locations you would find {SPECIES}.
"""