import streamlit as st
import requests
import openai
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="What to Wear Bot",
    page_icon="🤖",
    layout="centered",
)

# --- Title and Description ---
st.title("What to Wear Bot 👕")
st.markdown("""
Welcome! Enter a city and I'll give you advice on what to wear and whether it's a good day for a picnic.
""")

# --- API Key Configuration ---
try:
    # Attempt to load API keys from Streamlit secrets
    openweathermap_api_key = st.secrets["OPENWEATHERMAP_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = openai_api_key
except KeyError:
    st.error("API keys not found! Please add your OpenWeatherMap and OpenAI API keys to the Streamlit secrets.", icon="🚨")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading API keys: {e}", icon="🚨")
    st.stop()


# --- Weather Function (Lab 5a) ---
def get_current_weather(location, api_key):
    """
    Fetches current weather information for a given location using the OpenWeatherMap API.
    """
    if not isinstance(location, str):
        return {"error": "Location must be a string."}

    # Clean up the location string
    if "," in location:
        location = location.split(",")[0].strip()

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"  # Get temperature in Celsius directly
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        # Extract relevant weather data
        weather_info = {
            "location": data.get("name", location),
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "temp_min": data["main"]["temp_min"],
            "temp_max": data["main"]["temp_max"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"]
        }
        return json.dumps(weather_info)

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return json.dumps({"error": f"City '{location}' not found. Please check the spelling."})
        return json.dumps({"error": f"An HTTP error occurred: {http_err}"})
    except requests.exceptions.RequestException as req_err:
        return json.dumps({"error": f"A request error occurred: {req_err}"})
    except (KeyError, IndexError) as e:
        return json.dumps({"error": f"Could not parse weather data. Invalid API response. Details: {e}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred: {e}"})


# --- Streamlit User Interface ---
location_input = st.text_input("Enter a city name:", placeholder="e.g., Syracuse, NY or London")

if st.button("Get Suggestion", type="primary"):
    if not location_input:
        st.warning("Please enter a city name.", icon="⚠️")
    else:
        with st.spinner(f"Checking the weather and thinking of suggestions for {location_input}..."):
            try:
                # --- OpenAI Integration (Lab 5b) ---
                client = openai.OpenAI(api_key=openai.api_key)
                messages = [{"role": "user", "content": f"What should I wear in {location_input} today?"}]
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. Syracuse, NY",
                                    },
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ]

                # First call to OpenAI to determine if a function call is needed
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                # Check if the model wants to call a function
                if tool_calls:
                    available_functions = {
                        "get_current_weather": get_current_weather,
                    }
                    function_name = tool_calls[0].function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_calls[0].function.arguments)

                    # Call the weather function
                    function_response = function_to_call(
                        location=function_args.get("location", "Syracuse, NY"),
                        api_key=openweathermap_api_key,
                    )

                    # Check if the weather function returned an error
                    weather_data = json.loads(function_response)
                    if "error" in weather_data:
                         st.error(f"Error fetching weather: {weather_data['error']}", icon="🌦️")
                         st.stop()


                    # Append the function call info and response to messages
                    messages.append(response_message)
                    messages.append(
                        {
                            "tool_call_id": tool_calls[0].id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )

                    # --- Second call to OpenAI with weather context ---
                    system_prompt = (
                        "You are a helpful assistant that provides clothing and activity suggestions based on weather data. "
                        "Based on the provided weather information, give advice on what to wear. "
                        "Also, state whether it's a good day for a picnic and explain why. "
                        "Format the output nicely using markdown."
                    )
                    messages.insert(0, {"role": "system", "content": system_prompt})


                    second_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                    )
                    suggestion = second_response.choices[0].message.content
                    
                    st.success("Here are your suggestions!", icon="✅")
                    st.markdown(suggestion)

                    # Display the raw weather data for context
                    with st.expander("See Raw Weather Data"):
                        st.json(weather_data)

                else:
                    st.info("The model didn't need to fetch weather data. Here is its direct response:", icon="💡")
                    st.markdown(response_message.content)

            except openai.APIError as e:
                st.error(f"OpenAI API Error: {e}", icon="🤖")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}", icon="🚨")

# --- Testing the weather function directly (for Lab 5a requirements) ---
st.markdown("---")
if st.checkbox("Test get_current_weather() function"):
    st.subheader("Function Test")
    test_location = st.text_input("Enter a location to test:", "Syracuse, NY")
    if st.button("Run Test"):
        weather_result = get_current_weather(test_location, openweathermap_api_key)
        st.json(weather_result)
