import streamlit as st

# Loading the trained model
try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    st.error("Error: 'random_forest_model.pkl' not found. Please ensure the trained model file is in the same directory.")
    st.stop()

# Loading the original data to get the list of regions for one-hot encoding
try:
    df_original = pd.read_csv('Life_Expectancy_Data.csv')
    regions = sorted(df_original['Region'].unique())
except FileNotFoundError:
    st.error("Error: 'Life_Expectancy_Data.csv' not found. This is needed for the dashboard to function.")
    st.stop()

st.set_page_config(page_title="Life Expectancy Predictor", layout="wide")

# Dashboard Title and Description
st.title("Life Expectancy Predictor")
st.markdown("### Predict Life Expectancy based on various health and economic factors.")
st.write("Use the sidebar on the left to input the values for the different features.")

# Sidebar for user input
with st.sidebar:
    st.header("Input Features")
    st.markdown("Enter the values for the following predictors:")

    year = st.number_input("Year", min_value=2000, max_value=2015, value=2010)
    infant_deaths = st.number_input("Infant Deaths (per 1000 births)", min_value=0.0, value=20.0, step=0.1)
    under_five_deaths = st.number_input("Under Five Deaths (per 1000 births)", min_value=0.0, value=25.0, step=0.1)
    adult_mortality = st.number_input("Adult Mortality (per 1000 population)", min_value=0.0, value=150.0, step=0.1)
    alcohol_consumption = st.number_input("Alcohol Consumption (litres)", min_value=0.0, value=5.0, step=0.1)
    hepatitis_b = st.number_input("Hepatitis B (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    measles = st.number_input("Measles (per 1000 population)", min_value=0.0, value=50.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, value=22.0, step=0.1)
    polio = st.number_input("Polio (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    diphtheria = st.number_input("Diphtheria (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    incidents_hiv = st.number_input("Incidents HIV (per 1000 population)", min_value=0.0, value=0.5, step=0.01)
    gdp_per_capita = st.number_input("GDP per Capita ($)", min_value=0.0, value=5000.0, step=100.0)
    population_mln = st.number_input("Population (millions)", min_value=0.0, value=50.0, step=0.1)
    thinness_ten_nineteen_years = st.number_input("Thinness 10-19 Years (%)", min_value=0.0, value=5.0, step=0.1)
    thinness_five_nine_years = st.number_input("Thinness 5-9 Years (%)", min_value=0.0, value=5.0, step=0.1)
    schooling = st.number_input("Schooling (years)", min_value=0.0, value=10.0, step=0.1)
    economy_status = st.radio("Economy Status", ["Developed", "Developing"])
    region = st.selectbox("Region", regions)

    predict_button = st.button("Predict Life Expectancy")

# Prediction logic
if predict_button:
    # Prepare the input data
    input_data = {
        'Year': year,
        'Infant_deaths': infant_deaths,
        'Under_five_deaths': under_five_deaths,
        'Adult_mortality': adult_mortality,
        'Alcohol_consumption': alcohol_consumption,
        'Hepatitis_B': hepatitis_b,
        'Measles': measles,
        'BMI': bmi,
        'Polio': polio,
        'Diphtheria': diphtheria,
        'Incidents_HIV': incidents_hiv,
        'GDP_per_capita': gdp_per_capita,
        'Population_mln': population_mln,
        'Thinness_ten_nineteen_years': thinness_ten_nineteen_years,
        'Thinness_five_nine_years': thinness_five_nine_years,
        'Schooling': schooling,
        'Economy_status_Developed': 1 if economy_status == "Developed" else 0,
        'Economy_status_Developing': 1 if economy_status == "Developing" else 0
    }
    
    # Creating DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Handling one-hot encoding for 'Region'
    for r in regions:
        input_df[f'Region_{r}'] = 1 if r == region else 0
    
    # Ensuring column order matches the training data
    ## This is critical for the model to make correct predictions
    feature_order = [
        'Year', 'Infant_deaths', 'Under_five_deaths', 'Adult_mortality',
        'Alcohol_consumption', 'Hepatitis_B', 'Measles', 'BMI', 'Polio',
        'Diphtheria', 'Incidents_HIV', 'GDP_per_capita', 'Population_mln',
        'Thinness_ten_nineteen_years', 'Thinness_five_nine_years', 'Schooling',
        'Economy_status_Developed', 'Economy_status_Developing'
    ] + [f'Region_{r}' for r in regions]

    input_df = input_df[feature_order]
    
    # Making the prediction
    prediction = model.predict(input_df)[0]
    
    # Displaying the prediction
    st.markdown("---")
    st.subheader("Prediction Result")
    st.metric(label="Predicted Life Expectancy", value=f"{prediction:.2f} years")