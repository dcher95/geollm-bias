squirrel_incontext_prompt = """
You are an expert biologist estimating the likelihood of finding a Eastern Gray Squirrel at various locations.  
Given the coordinates, address, and nearby places, predict the likelihood on a scale from 0.0 to 9.9.  

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
How likely are you to find a Eastern Gray Squirrel here (On a Scale from 0.0 to 9.9): 2.3  

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
How likely are you to find a Eastern Gray Squirrel here (On a Scale from 0.0 to 9.9): 5.7  

---

Now, predict the likelihood for the following location:  


{NEW_LOCATION}

"""  

squirrel_prompt_expert = """
You are an expert wildlife biologist estimating the likelihood of finding an Eastern Gray Squirrel at various locations.  
Given the coordinates, address, and nearby places, predict the likelihood on a scale from 0.0 to 9.9. 

Predict the likelihood for the following location:  

{NEW_LOCATION}

"""  

coordinate_prompt = """
You are an expert biologist estimating the likelihood of finding different species at various locations.  Given the coordinates, address, and nearby places, predict the types of species found at this location.

{NEW_LOCATION}
"""