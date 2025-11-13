import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from openai import OpenAI

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def llm_call(prompt, system_message="You are a helpful travel planning assistant.", temperature=0.3):
    """
    Connect to OpenAI API and get response
    """
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"LLM API Error: {str(e)}")
        return None

def get_weather_data(city, date_str):
    """
    Fetch weather data from OpenWeatherMap API
    For dates >2 days ahead, use LLM to predict weather
    """
    api_key = st.secrets["OPENWEATHERMAP_API_KEY"]
    
    try:
        # Get current weather data
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        
        if response.status_code != 200:
            return {"error": f"Could not fetch weather for {city}. Please check city name."}
        
        data = response.json()
        
        # Calculate days difference
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        today = datetime.now()
        days_diff = (target_date - today).days
        
        current_weather = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }
        
        # If date is more than 2 days away, use LLM to predict
        if days_diff > 2:
            prompt = f"""Based on current weather data for {city}:
Temperature: {current_weather['temperature']}°C
Humidity: {current_weather['humidity']}%
Description: {current_weather['description']}

Predict the likely weather conditions for {days_diff} days from now ({date_str}).
Consider seasonal patterns and typical weather variations.

Return ONLY a JSON object with this exact format:
{{
    "temperature": <predicted temp in celsius>,
    "description": "<brief weather description>",
    "humidity": <predicted humidity %>,
    "conditions": "<overall conditions>"
}}"""
            
            llm_response = llm_call(prompt, temperature=0.3)
            
            try:
                # Extract JSON from response
                if "```json" in llm_response:
                    llm_response = llm_response.split("```json")[1].split("```")[0]
                elif "```" in llm_response:
                    llm_response = llm_response.split("```")[1].split("```")[0]
                
                predicted_weather = json.loads(llm_response.strip())
                predicted_weather["city"] = city
                predicted_weather["predicted"] = True
                return predicted_weather
            except:
                # Fallback to current weather if parsing fails
                current_weather["predicted"] = False
                return current_weather
        else:
            current_weather["predicted"] = False
            return current_weather
            
    except Exception as e:
        return {"error": str(e)}

def calculate_travel_info(origin, destination):
    """
    Use LLM to estimate distance and travel time
    """
    prompt = f"""Calculate the approximate travel information between {origin} and {destination}.

Consider:
1. Approximate distance in kilometers and miles
2. Estimated driving time (if reasonable to drive)
3. Estimated flight time (if applicable)
4. Whether this is a domestic or international trip
5. Recommended primary mode of transport

Return ONLY a JSON object with this exact format:
{{
    "distance_km": <distance in km>,
    "distance_miles": <distance in miles>,
    "drive_time_hours": <driving hours or null if not recommended>,
    "flight_time_hours": <flight hours>,
    "is_international": <true/false>,
    "primary_transport": "<car/flight/train>"
}}"""
    
    response = llm_call(prompt, temperature=0.3)
    
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        return json.loads(response.strip())
    except:
        # Fallback values
        return {
            "distance_km": 500,
            "distance_miles": 310,
            "drive_time_hours": 6,
            "flight_time_hours": 1.5,
            "is_international": False,
            "primary_transport": "flight"
        }

def weather_agent(origin_weather, destination_weather):
    """
    Compare weather at origin and destination
    """
    prompt = f"""As a weather analysis agent, compare the weather conditions:

Origin Weather:
- City: {origin_weather.get('city', 'Unknown')}
- Temperature: {origin_weather.get('temperature', 'N/A')}°C
- Description: {origin_weather.get('description', 'N/A')}
- Humidity: {origin_weather.get('humidity', 'N/A')}%

Destination Weather:
- City: {destination_weather.get('city', 'Unknown')}
- Temperature: {destination_weather.get('temperature', 'N/A')}°C
- Description: {destination_weather.get('description', 'N/A')}
- Humidity: {destination_weather.get('humidity', 'N/A')}%

Provide a concise comparison highlighting:
1. Temperature differences
2. Weather condition differences
3. Any notable changes travelers should prepare for
4. Overall comfort level at destination

Keep response under 150 words."""
    
    return llm_call(prompt)

def logistics_agent(origin, destination, departure_date, duration, travel_info, origin_weather, dest_weather):
    """
    Recommend travel mode, timing, and tips
    """
    prompt = f"""As a logistics planning agent, provide travel recommendations:

Trip Details:
- Origin: {origin}
- Destination: {destination}
- Departure: {departure_date}
- Duration: {duration} days
- Distance: {travel_info.get('distance_km', 'Unknown')} km
- International: {travel_info.get('is_international', False)}

Weather Context:
- Origin: {origin_weather.get('temperature', 'N/A')}°C, {origin_weather.get('description', 'N/A')}
- Destination: {dest_weather.get('temperature', 'N/A')}°C, {dest_weather.get('description', 'N/A')}

Provide recommendations for:
1. Best mode of transportation and why
2. Optimal departure time/timing considerations
3. Travel duration estimate
4. Important logistics tips (documentation, booking advice, etc.)
5. Local transportation options at destination

{"NOTE: This is an international trip. Do NOT recommend driving." if travel_info.get('is_international') else ""}

Keep response organized and under 200 words."""
    
    return llm_call(prompt)

def packing_agent(destination_weather, duration, destination):
    """
    Suggest clothing and accessories based on weather and trip length
    """
    prompt = f"""As a packing advisor agent, create a packing list:

Trip Details:
- Destination: {destination}
- Duration: {duration} days
- Weather: {destination_weather.get('temperature', 'N/A')}°C, {destination_weather.get('description', 'N/A')}
- Humidity: {destination_weather.get('humidity', 'N/A')}%

Suggest:
1. Clothing items (appropriate for weather and duration)
2. Accessories and weather-specific gear
3. Essential travel items
4. Any special considerations for this destination

Organize by categories. Keep response under 200 words."""
    
    return llm_call(prompt)

def activity_agent(destination, duration, departure_date, dest_weather):
    """
    Create day-wise itinerary with local suggestions
    """
    prompt = f"""As an activity planning agent, create a {duration}-day itinerary:

Destination: {destination}
Start Date: {departure_date}
Weather: {dest_weather.get('temperature', 'N/A')}°C, {dest_weather.get('description', 'N/A')}

Create a day-by-day itinerary including:
1. Morning, afternoon, and evening activities
2. Local attractions and must-see places
3. Restaurant/dining suggestions
4. Weather-appropriate activities
5. Mix of popular sights and hidden gems

Format as:
Day 1 (Date):
- Morning: ...
- Afternoon: ...
- Evening: ...

Keep each day concise but informative."""
    
    return llm_call(prompt)

# Streamlit App
def main():
    st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")
    
    st.title("✈️ Multi-Agent Travel Planning System")
    st.markdown("Plan your perfect trip with AI-powered agents and real-time weather data!")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Trip Details")
        
        origin = st.text_input("Origin City", placeholder="e.g., New York")
        destination = st.text_input("Destination City", placeholder="e.g., Paris")
        
        departure_date = st.date_input(
            "Departure Date",
            min_value=datetime.now().date(),
            value=datetime.now().date() + timedelta(days=7)
        )
        
        duration = st.number_input("Trip Duration (days)", min_value=1, max_value=30, value=5)
        
        plan_button = st.button("🚀 Generate Travel Plan", type="primary", use_container_width=True)
    
    # Main content area
    if plan_button:
        if not origin or not destination:
            st.error("Please enter both origin and destination cities!")
            return
        
        with st.spinner("Planning your trip... This may take a moment ⏳"):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Fetch weather data
            status_text.text("🌤️ Fetching weather data...")
            progress_bar.progress(20)
            
            origin_weather = get_weather_data(origin, departure_date.strftime("%Y-%m-%d"))
            destination_weather = get_weather_data(destination, departure_date.strftime("%Y-%m-%d"))
            
            if "error" in origin_weather or "error" in destination_weather:
                st.error("Could not fetch weather data. Please check city names and try again.")
                return
            
            # Step 2: Calculate travel info
            status_text.text("🗺️ Calculating travel information...")
            progress_bar.progress(40)
            
            travel_info = calculate_travel_info(origin, destination)
            
            # Step 3: Run agents
            status_text.text("🤖 Weather agent analyzing conditions...")
            progress_bar.progress(50)
            weather_analysis = weather_agent(origin_weather, destination_weather)
            
            status_text.text("🚗 Logistics agent planning transportation...")
            progress_bar.progress(60)
            logistics_plan = logistics_agent(
                origin, destination, departure_date.strftime("%Y-%m-%d"), 
                duration, travel_info, origin_weather, destination_weather
            )
            
            status_text.text("🎒 Packing agent preparing your list...")
            progress_bar.progress(75)
            packing_list = packing_agent(destination_weather, duration, destination)
            
            status_text.text("📅 Activity agent creating your itinerary...")
            progress_bar.progress(90)
            itinerary = activity_agent(
                destination, duration, departure_date.strftime("%Y-%m-%d"), 
                destination_weather
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Travel plan complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        # Display results
        st.success("🎉 Your travel plan is ready!")
        
        # Overview Section
        st.header("📋 Trip Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Origin", origin)
        with col2:
            st.metric("Destination", destination)
        with col3:
            st.metric("Duration", f"{duration} days")
        with col4:
            st.metric("Distance", f"{travel_info.get('distance_km', 'N/A')} km")
        
        st.divider()
        
        # Weather Section
        st.header("🌤️ Weather Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📍 {origin}")
            st.metric("Temperature", f"{origin_weather.get('temperature', 'N/A')}°C")
            st.write(f"**Conditions:** {origin_weather.get('description', 'N/A').title()}")
            st.write(f"**Humidity:** {origin_weather.get('humidity', 'N/A')}%")
            if origin_weather.get('predicted'):
                st.info("⚠️ Predicted weather (>2 days ahead)")
        
        with col2:
            st.subheader(f"📍 {destination}")
            st.metric("Temperature", f"{destination_weather.get('temperature', 'N/A')}°C")
            st.write(f"**Conditions:** {destination_weather.get('description', 'N/A').title()}")
            st.write(f"**Humidity:** {destination_weather.get('humidity', 'N/A')}%")
            if destination_weather.get('predicted'):
                st.info("⚠️ Predicted weather (>2 days ahead)")
        
        st.write("**Analysis:**")
        st.info(weather_analysis)
        
        st.divider()
        
        # Logistics Section
        st.header("🚗 Logistics & Transportation")
        st.write(logistics_plan)
        
        st.divider()
        
        # Packing Section
        st.header("🎒 Packing Recommendations")
        st.write(packing_list)
        
        st.divider()
        
        # Itinerary Section
        st.header("📅 Day-by-Day Itinerary")
        st.write(itinerary)
        
        # Footer
        st.divider()
        st.caption("Powered by OpenAI GPT-3.5 and OpenWeatherMap API")

if __name__ == "__main__":
    main()