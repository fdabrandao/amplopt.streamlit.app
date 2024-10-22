import streamlit as st
import googlemaps
import pydeck as pdk
import pandas as pd
import numpy as np
import random
from amplpy import AMPL
import json
import os

# Initialize Google Maps client with your API key
API_KEY = os.environ.get("GOOGLE_API_KEY", None)

# List of Disney World theme parks
parks = [
    "EPCOT, Orlando, FL",
    "Magic Kingdom, Orlando, FL",
    "Disney's Hollywood Studios, Orlando, FL",
    "Disney's Animal Kingdom, Orlando, FL",
]


# Function to fetch coordinates
@st.cache_data
def fetch_coordinates(park_name):
    gmaps = googlemaps.Client(key=API_KEY)
    geocode_result = gmaps.geocode(park_name)
    if geocode_result:
        location = geocode_result[0]["geometry"]["location"]
        return location["lat"], location["lng"]
    else:
        return None, None


@st.cache_data
def find_place_near_location(location, place_type="restaurant", radius=1000):
    gmaps = googlemaps.Client(key=API_KEY)
    print(f"find_place_near_location({location})")
    # Perform a nearby search for restaurants within the specified radius
    result = gmaps.places_nearby(location=location, radius=radius, type=place_type)
    restaurants = []
    for restaurant in result.get("results", []):
        name = restaurant["name"]
        lat = restaurant["geometry"]["location"]["lat"]
        lng = restaurant["geometry"]["location"]["lng"]
        place_id = restaurant["place_id"]
        details = gmaps.place(place_id=place_id)
        rating = details["result"].get("rating")
        reviews = details["result"].get("user_ratings_total")
        restaurants.append(
            {
                "name": name,
                "coordinates": (lat, lng),
                "rating": rating,
                "reviews": reviews,
            }
        )
    return restaurants


@st.cache_data
def fetch_data(parks):
    data = {}
    for park_name in parks:
        data[park_name] = {}
        park_location = fetch_coordinates(park_name)
        data[park_name]["coordinates"] = park_location
        data[park_name]["restaurants"] = find_place_near_location(
            location=park_location, place_type="restaurant"
        )
        data[park_name]["cafes"] = find_place_near_location(
            location=park_location, place_type="cafe"
        )

    return data


MODEL = r"""
    set RESTAURANTS;  # Set of restaurants

    param cost {RESTAURANTS};  # Cost to buy each restaurant
    param rating {RESTAURANTS};  # Rating of each restaurant
    param reviews {RESTAURANTS};  # Number of reviews for each restaurant

    param Budget;  # Total budget available to buy restaurants

    var Buy {RESTAURANTS} binary;  # Decision variable: 1 if a restaurant is bought, 0 otherwise

    # Objective: Maximize the total number of reviews
    maximize TotalReviews: 
        sum {r in RESTAURANTS} reviews[r] * Buy[r];

    # Constraint: Total cost of selected restaurants must not exceed the budget
    subject to BudgetConstraint:
        sum {r in RESTAURANTS} cost[r] * Buy[r] <= Budget;

    # Constraint: Ensure the average rating is at least 4 (simplified linear form)
    subject to AverageRatingConstraint:
        sum {r in RESTAURANTS} rating[r] * reviews[r] * Buy[r] >= 4 * sum {r in RESTAURANTS} reviews[r] * Buy[r];
    """


# Streamlit application
def main():
    st.title("🍽️ Bistro Game")

    st.markdown(
        r"""
    This app was generated using ChatGPT ([ChatGPT Session](https://chatgpt.com/share/da0f42a5-1b94-4787-8772-554864811d6b),
                [Webinar on 🚀 AMPL + 🐍 Python + 🧠 AI 🤖](https://www.youtube.com/watch?v=Yw_Usvea8jM))
    """
    )

    with st.expander("Problem description and 🚀 AMPL model by 🤖 ChatGPT"):
        st.markdown(
            r"""
        ## Problem Description: Optimal Restaurant Selection

        ### Objective
        The primary goal of this AMPL model is to select a subset of restaurants to maximize the total number of customer reviews while adhering to a specified budget. The selection process also aims to ensure that the average rating of the chosen restaurants remains above a certain threshold, reflecting a commitment to quality.

        ### Decision Variables
        - **`Buy[r]`**: A binary variable for each restaurant \( r \) in the set of possible restaurants. It takes a value of 1 if the restaurant is selected, and 0 otherwise.

        ### Parameters
        - **`cost[r]`**: The cost associated with purchasing or investing in restaurant \( r \).
        - **`rating[r]`**: The average customer rating of restaurant \( r \).
        - **`reviews[r]`**: The number of reviews for restaurant \( r \).
        - **`Budget`**: The total budget available for purchasing restaurants.

        ### Constraints
        1. **Budget Constraint**: The total cost of the selected restaurants must not exceed the available budget. This ensures that the selections remain financially viable.
        $$
        \sum_{r \in \text{RESTAURANTS}} \text{cost}[r] \times \text{Buy}[r] \leq \text{Budget}
        $$
        2. **Average Rating Constraint**: The weighted average rating of the selected restaurants must be at least 4. This constraint ensures that the chosen establishments maintain a high standard of quality. The average rating is calculated as the sum of the product of ratings and the corresponding binary decision variables, normalized by the number of reviews. This constraint requires careful linearization to handle in a linear programming framework.
        $$
        \sum_{r \in \text{RESTAURANTS}} \text{rating}[r] \times \text{reviews}[r] \times \text{Buy}[r] \geq 4 \times \sum_{r \in \text{RESTAURANTS}} \text{reviews}[r] \times \text{Buy}[r]
        $$

        ### Objective Function
        The objective function is designed to maximize the sum of reviews from the selected restaurants, thereby favoring restaurants that are more popular or have greater visibility among customers.
        $$
        \text{Maximize} \sum_{r \in \text{RESTAURANTS}} \text{reviews}[r] \times \text{Buy}[r]
        $$

        ### Model Usage and Expected Output
        - **Usage**: The model is used by decision-makers considering investments in a portfolio of restaurants, with constraints on budget and a minimum quality threshold. It is suitable for scenarios in corporate strategy, franchise decisions, or portfolio management in the hospitality sector.
        - **Output**: The model will output the binary decisions for each restaurant, indicating whether it should be included in the portfolio. Additionally, the total cost and the average rating of the selected group will be computed to ensure they meet the stipulated constraints.
        """
        )

        st.markdown(
            f"""
        ### AMPL Model
        ```python
        {MODEL}
        ```
        """
        )

    # Fetch data
    data_file = os.path.join(os.path.dirname(__file__), "data.json")
    if not os.path.exists(data_file):
        data = fetch_data(parks)
        open(data_file, "w", newline="\n", encoding="utf-8").write(
            json.dumps(data, separators=(",", ": "), indent="  ", ensure_ascii=False)
            + "\n"
        )
    data = json.load(open(data_file, "r", encoding="utf-8"))

    park_name = st.selectbox("Choose a Disney World park:", parks, index=0)

    place_type = st.selectbox(
        "Choose the type of place to display:",
        ["restaurants", "cafes"],
        index=0,
    )

    park_location = data[park_name]["coordinates"]
    places = data[park_name][place_type]

    @st.cache_data
    def generate_random_numbers(n):
        return np.random.randint(1, 5, size=n) * 100000

    @st.cache_data
    def generate_random_bools(n, num_trues):
        bool_list = [True] * num_trues + [False] * (n - num_trues)
        random.shuffle(bool_list)
        return bool_list

    df = pd.DataFrame(places)[["name", "rating", "reviews"]]
    df["name"] = [f"{index}-{name}" for index, name in enumerate(df["name"])]
    df.set_index(["name"], inplace=True)
    df["rent_estimate"] = generate_random_numbers(len(df))
    df["Rent?"] = generate_random_bools(len(df), min(3, len(df)))
    df = st.data_editor(
        df,
        disabled=["name"],
        column_config={
            "Buy?": st.column_config.CheckboxColumn(
                "Would you rent?",
                help="Select your **favorite** restaurants",
                default=False,
            )
        },
    )
    selected = df["Rent?"] == 1

    default_budget = df["rent_estimate"].sum() / 5
    if selected.sum() > 0:
        default_budget = df[selected]["rent_estimate"].sum()
        st.write(f"Total cost of selected locations: {default_budget:,}")

    # Prepare data for mapping
    data = [
        {
            "name": location["name"],
            "lat": location["coordinates"][0],
            "lon": location["coordinates"][1],
        }
        for location in places
    ]

    # Define map
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=park_location[0],
                longitude=park_location[1],
                zoom=14,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=data,
                    get_position="[lon, lat]",
                    get_color="[200, 30, 0, 160]",
                    get_radius=25,
                ),
                pdk.Layer(
                    "TextLayer",
                    data=data,
                    get_position="[lon, lat]",
                    get_text="name",
                    get_color=[0, 0, 0, 200],
                    get_size=8,
                    get_alignment_baseline="'bottom'",
                ),
            ],
        )
    )

    ampl = AMPL()
    ampl.eval(MODEL)
    ampl.set["RESTAURANTS"] = df.index
    ampl.param["cost"] = df["rent_estimate"]
    ampl.param["rating"] = df["rating"]
    ampl.param["reviews"] = df["reviews"]

    ampl.param["Budget"] = st.slider(
        "Budget?",
        min_value=float(df["rent_estimate"].min()),
        max_value=float(df["rent_estimate"].sum()),
        value=float(default_budget),
        step=float(df["rent_estimate"].sum() / 10),
    )

    output = ampl.solve(solver="gurobi", gurobi_options="outlev=1", return_output=True)
    solution = ampl.var["Buy"].to_pandas()
    df = pd.concat([df, solution], axis=1)
    to_buy = df["Buy.val"] == 1

    total_cost = df[to_buy]["rent_estimate"].sum()
    total_reviews = df[to_buy]["reviews"].sum()
    average_rating = (
        df[to_buy]["rating"] * df[to_buy]["reviews"]
    ).sum() / total_reviews
    st.write(
        f"""
    Optimal solution:
    - Total cost: {total_cost:,}
    - Total reviews: {total_reviews:,}
    - Average rating: {average_rating:.2f}
    """
    )
    st.write(df[to_buy])

    if selected.sum() > 0:
        total_cost = df[selected]["rent_estimate"].sum()
        total_reviews = df[selected]["reviews"].sum()
        average_rating = (
            df[selected]["rating"] * df[selected]["reviews"]
        ).sum() / total_reviews
        st.write(
            f"""
        Your solution:
        - Total cost: {total_cost:,}
        - Total reviews: {total_reviews:,}
        - Average rating: {average_rating:.2f}
        """
        )
        st.write(df[selected])

    st.write(f"```\n{output}\n```")


if __name__ == "__main__":
    main()
