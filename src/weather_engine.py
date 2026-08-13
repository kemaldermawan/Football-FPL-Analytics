import requests
import pandas as pd
import streamlit as st

# Matriks kordinat stadion Liga Primer (Garis Lintang, Garis Bujur)
STADIUM_COORDS = {
    "Arsenal": (51.5549, -0.1084),
    "Aston Villa": (52.5091, -1.8848),
    "Bournemouth": (50.7352, -1.8385),
    "Brentford": (51.4907, -0.3016),
    "Brighton": (50.8616, -0.0837),
    "Chelsea": (51.4817, -0.1910),
    "Crystal Palace": (51.3983, -0.0855),
    "Everton": (53.4388, -2.9663),
    "Fulham": (51.4750, -0.2216),
    "Ipswich Town": (52.0535, 1.1448),
    "Leicester": (52.6204, -1.1422),
    "Liverpool": (53.4308, -2.9608),
    "Man City": (53.4831, -2.2004),
    "Man Utd": (53.4631, -2.2913),
    "Newcastle": (54.9756, -1.6216),
    "Nott'm Forest": (52.9400, -1.1328),
    "Southampton": (50.9058, -1.3911),
    "Spurs": (51.6042, -0.0664),
    "Sunderland": (54.9146, -1.3883),
    "West Ham": (51.5387, -0.0166),
    "Wolves": (52.5902, -2.1304)
}

@st.cache_data(ttl=10800)
def fetch_matchday_weather(home_team):
    """
    Mengekstrak data telemetri cuaca langsung dari kordinat stadion.
    """
    if home_team not in STADIUM_COORDS:
        return None
        
    lat, lon = STADIUM_COORDS[home_team]
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        weather = data.get("current_weather", {})
        return {
            "Temperature_C": weather.get("temperature", 0),
            "Wind_Speed_kmh": weather.get("windspeed", 0),
            "Weather_Code": weather.get("weathercode", 0)
        }
    except requests.exceptions.RequestException:
        return None

def analyze_weather_impact(weather_data):
    """
    Mengalkulasi koefisien penalti taktikal berdasarkan metrik cuaca.
    """
    impact = {"xG_Multiplier": 1.0, "xGA_Multiplier": 1.0, "Risk_Flag": "Optimal Conditions"}
    
    if not weather_data:
        return impact
        
    temp = weather_data["Temperature_C"]
    wind = weather_data["Wind_Speed_kmh"]
    w_code = weather_data["Weather_Code"]
    
    # Kode interpretasi WMO (World Meteorological Organization)
    heavy_rain_codes = [63, 65, 81, 82]
    snow_codes = [71, 73, 75, 85, 86]
    
    if w_code in heavy_rain_codes or w_code in snow_codes:
        impact["xG_Multiplier"] -= 0.15 
        impact["xGA_Multiplier"] += 0.20 
        impact["Risk_Flag"] = "Slippery Pitch (Defensive Errors Likely)"
        
    if wind > 35.0:
        impact["xG_Multiplier"] -= 0.10
        if "Slippery" in impact["Risk_Flag"]:
            impact["Risk_Flag"] += " | High Wind (Crosses Affected)"
        else:
            impact["Risk_Flag"] = "High Wind (Crosses Affected)"
            
    if temp < 2.0:
        impact["Risk_Flag"] += " | Freezing"
        
    return impact