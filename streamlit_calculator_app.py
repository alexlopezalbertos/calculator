import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
from geopy.geocoders import Nominatim
import requests


# Function to render the Supply Chain Resilience Calculator page
def render_calculator_page():
    st.title("Supply Chain Resilience Simulator")
    st.write("")
    # Get user inputs
    input1 = st.number_input("__Lead Time (:blue[days])__", value=None, placeholder="Introduce Lead Time", step=1)
    if input1 is not None and type(input1) != int:
        st.warning("Please enter a number in integer format")
    st.write("")
    input2 = st.number_input("__Distance from Supplier to CM (:blue[km])__", value=None, placeholder="Introduce Distance", step=.1, format="%.1f")
    if input2 is not None and type(input2) != float:
        st.warning("Please enter a number format")
    st.write("")
    input3 = st.selectbox("__BCP Risk__", ["LOW", "MEDIUM", "HIGH"], index=None, placeholder="Choose an option")
    with st.expander("Help"):
        st.write(":green[LOW]: A backup supplier is identified.")
        st.write(":orange[MEDIUM]: No backup supplier is identified but either:")
        st.write("                 1. The primary supplier has at least 1 plant in another location.")
        st.write("                 2. There is another material either used and qualified for CM internal production or qualified by another P&G plant.")
        st.write(":red[HIGH]: No backup supplier is identified.")
    st.write("")
    # input4 = st.number_input("Fragility Index of the country of the supplier", 0.0)
    # input5 = st.number_input("Natural Disaster Risk of the country of the supplier (%)", 0.0)
    input6 = st.selectbox("__Supplier Country__", ['Albania', 'Algeria', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czechia', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'DR Congo', 'East Timor', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'São Tomé and Príncipe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'], index=None, placeholder="Choose a country", help="The selected country will determine the _Fragility Index_ and _Natural Disaster Risk_ KPIs")
    st.write("")

    # Button to trigger the script
    if st.button("Simulate", type="primary"):
    # Check if any of the required inputs are None
        if input1 is None or input2 is None or input3 is None or input6 is None:
            # Identify which inputs are missing
            missing_inputs = []
            if input1 is None:
                missing_inputs.append(":red[Lead Time (days)]")
            if input2 is None:
                missing_inputs.append(":red[Distance from Supplier to CM (km)]")
            if input3 is None:
                missing_inputs.append(":red[BCP Risk]")
            if input6 is None:
                missing_inputs.append(":red[Supplier Country]")

            # Display a warning message
            st.warning(f"Please introduce: {', '.join(missing_inputs)}.", icon="⚠️")

        else:
            # Call the Python script with the user inputs
            try:
                run_script(input1, input2, input3, input6)
            except PermissionError as e:
                st.error(f"Please close the file in the following path: {e}")
            except IndexError:
                st.warning("Selected country is not currently in the database so _Fragility Index_ and _Natural Disaster Risk_ cannot be retrieved, please select a different one.")

def run_script(input1, input2, input3, input6):
    # Your Python script logic here

    
    # ======================================== COUNTRY RISKS ======================================================= #
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import os

    # Replace with the URL of the Wikipedia page containing the table
    url1 = "https://en.wikipedia.org/wiki/List_of_countries_by_Fragile_States_Index"
    url2 = "https://en.wikipedia.org/wiki/List_of_countries_by_natural_disaster_risk"

    response1 = requests.get(url1, verify=False)
    soup1 = BeautifulSoup(response1.text, "html.parser")
    response2 = requests.get(url2, verify=False)
    soup2 = BeautifulSoup(response2.text, "html.parser")
 
    # # Find and extract the table containing the country data
    table1 = soup1.find("table", {"class": "wikitable sortable"})
    table2 = soup2.find("table", {"class": "wikitable sortable"})
   
    # # # Initialize empty dictionaries to store data
    country_data1 = {}
    country_data2 = {}
    
    # # Loop through rows in the table and extract data
    for row in table1.find_all("tr")[1:]:      #<tr> is table row, <td> is table cell and <th> is table header
        columns1 = row.find_all("td")
        country1 = columns1[1].text.strip()
        score1 = columns1[2].text.strip()
        country_data1[country1] = score1
    for row in table2.find_all("tr")[1:]:      #<tr> is table row, <td> is table cell and <th> is table header
        columns2 = row.find_all("td")
        country2 = columns2[1].text.strip()
        score2 = columns2[2].text.strip()
        country_data2[country2] = score2
 
    # # # Create a DataFrame from the dictionary
    df_fragility = pd.DataFrame(list(country_data1.items()), columns=["Manufacturer Country", "Fragility Index"])
    df_naturaldisaster = pd.DataFrame(list(country_data2.items()), columns=["Manufacturer Country", "Natural Disaster Risk"])
    df_naturaldisaster['Natural Disaster Risk'] = df_naturaldisaster['Natural Disaster Risk'].str.replace('%', '').astype(float)

    # Get Fragility Index and Natural Disaster Risk for the selected country
    input9 = df_fragility[df_fragility["Manufacturer Country"] == input6]["Fragility Index"].values[0]
    input10 = df_naturaldisaster[df_naturaldisaster["Manufacturer Country"] == input6]["Natural Disaster Risk"].values[0]


    st.metric("","")

    with st.expander("See Explanation"):
        st.divider()
        st.write("__Current KPIs:__ Current values for all 5 KPIs.")
        st.divider()
        st.write("__Current Supply Chain Strength:__ :green[HIGH], :orange[MEDIUM] or :red[LOW] - How strong the current supply chain is, with a percentage orientation that compares the score against all scores in the current portfolio.", unsafe_allow_html=True)
        st.divider()
        st.write("__Target KPIs:__ Target KPI values to go up to the next level of supply chain strength. For example, :orange[MEDIUM] > :green[HIGH].")
        st.divider()
        st.write("__Target Supply Chain Strength:__ How strong the supply chain for that supplier would be if you made one of the suggested changes.")
        st.write("")
    st.metric("","")

    st.subheader("Current KPIs", anchor=None, help=None, divider="grey")

    c1, c2, c3 = st.columns(3)
    c1.metric("Lead Time", f"{input1} days")
    c2.metric("Distance", f"{input2:.1f} km")
    c3.metric("BCP Risk", input3)

    c4, c5 = st.columns(2)
    c4.metric(f"Fragility Index for :blue[{input6}]", input9)
    c5.metric(f"Natural Disaster Risk for :blue[{input6}]", f"{input10:.1f}%")

    st.metric("","")



    # Convert BCP_risk to numerical values
    bcp_mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    df0 = pd.DataFrame({
        'Lead Time': [input1],
        'Distance (km)': [input2],
        'BCP_risk': [input3],
        'Fragility Index': [input9],
        'Natural Disaster Risk': [input10]
    })

    df0['BCP_risk'] = df0['BCP_risk'].map(bcp_mapping)

    

    # df1 = pd.read_excel(r"C:\Users\lopez.a.83\OneDrive - Procter and Gamble\P&G Internship\0_PROJECTS\Project 3\3_1_Supply_Chain_Resilience\SCR_tool\0_SCR_Weights_from_comparison_matrix\streamlit_data_for_scaler.xlsx")
    df1 = pd.read_csv("streamlit_data_for_scaler.csv")

    # # GitHub raw URL for your Excel file
    # github_raw_url = "https://github.com/alexlopezalbertos/calculator/blob/main/streamlit_data_for_scaler.xlsx"

    # # Function to fetch Excel file from GitHub
    # def get_data():
    #     response = requests.get(github_raw_url)
    #     return pd.read_excel(pd.BytesIO(response.content))

    # # Load the data
    # df1 = get_data()


    df = pd.concat([df0, df1], ignore_index=True)


    scaler = StandardScaler()
    df[["Lead_Time_T", "Distance_T", "Fragility_Index_T", "Natural_Disaster_Risk_T", "BCP_Risk_T"]] = scaler.fit_transform(df[["Lead Time", "Distance (km)", "Fragility Index", "Natural Disaster Risk", "BCP_risk"]])

    #============== ACTUAL TOOL ==============#
    # Weights
    leadtime_weight = 0.205
    distance_weight = 0.117
    fragilityindex_weight = 0.042
    naturaldisasterrisk_weight = 0.042
    bcprisk_weight = 0.595

    df["SCR_score"] = -(leadtime_weight * df['Lead_Time_T']) - (distance_weight * df['Distance_T']) - (fragilityindex_weight * df['Fragility_Index_T']) - (naturaldisasterrisk_weight * df['Natural_Disaster_Risk_T']) - (bcprisk_weight * df['BCP_Risk_T'])

    # Scores 0, 1 or 2 for SCR
    df['SCR_Strength'] = df['SCR_score'].apply(lambda x: 0 if x <= -0.75 else (1 if -0.75 < x < -0.19 else 2))

    # selected_columns = df.loc[:, ['Lead Time', 'Distance (km)', 'Fragility Index', 'Natural Disaster Risk', 'BCP_risk', 'Lead_Time_T', 'Distance_T', 'Fragility_Index_T', 'Natural_Disaster_Risk_T', 'BCP_Risk_T', 'SCR_score', 'SCR_Strength']]
    # st.dataframe(selected_columns)



    # Display the value of "SCR_Strength" for the first row
    # st.write("SCR_Strength for the first row:", df.loc[0, 'SCR_Strength'])

    # Display the corresponding label for "SCR_Strength" for the first row
    strength_labels = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}

    st.subheader("Current Supply Chain Strength", anchor=None, help=None, divider="grey")

    c6, c7 = st.columns(2)
    if df.loc[0, 'SCR_Strength'] == 0:
        c6.metric(":red_circle: :red[Supply Chain Strength]", strength_labels[0])
    elif df.loc[0, 'SCR_Strength'] == 1:
        c6.metric(":large_orange_circle: :orange[Supply Chain Strength]", strength_labels[1])
    else:
        c6.metric(":large_green_circle: :green[Supply Chain Strength]", strength_labels[2])







    df_score = df['SCR_score']
    # Progress bar based on SCR_score
    progress_value = df.loc[0, 'SCR_score']
    progress_min = df1['SCR_score'].min()
    progress_max = df1['SCR_score'].max()
    if progress_max > progress_value and progress_min < progress_value:
        supply_chain_strength_percentage = abs(progress_value - progress_min) / abs(progress_max - progress_min)
    else:
        # Set supply_chain_strength_percentage to 0 if progress_value < progress_min
        # Set supply_chain_strength_percentage to 1 if progress_value > progress_max
        supply_chain_strength_percentage = 0 if progress_value < progress_min else 1

    if df.loc[0, 'SCR_Strength'] == 0:
        c7.metric(":red_circle: :red[Supply Chain Strength as a Percentage]", f"{supply_chain_strength_percentage:.1%}")
    elif df.loc[0, 'SCR_Strength'] == 1:
        c7.metric(":large_orange_circle: :orange[Supply Chain Strength as a Percentage]", f"{supply_chain_strength_percentage:.1%}")
    else:
        c7.metric(":large_green_circle: :green[Supply Chain Strength as a Percentage]", f"{supply_chain_strength_percentage:.1%}")  
    
    st.metric("","")
    


    # st.data_editor(
    #     df_score,
    #     column_config={
    #         "SCR_score": st.column_config.ProgressColumn(
    #             "Supply Chain Strength",
    #             help="Shows how strong the supply chain is for the simulated material as opposed to all the other supply chain in the portfolio.",
    #             format="%.2f",
    #             min_value=progress_min,
    #             max_value=progress_max,
    #         ),
    #     },
    #     hide_index=True,
    # )
   
   

   
    # ======= Change in criteria required ======== #
    # LeadTime
    df['Lead Time Required Scaled'] = df.apply(lambda row: ((-0.75) - (-distance_weight * row['Distance_T'] - fragilityindex_weight * row['Fragility_Index_T'] - naturaldisasterrisk_weight * row['Natural_Disaster_Risk_T'] - bcprisk_weight * row['BCP_Risk_T'])) / (-leadtime_weight) if row['SCR_Strength'] == 0 else (((-0.19) - (-distance_weight * row['Distance_T'] - fragilityindex_weight * row['Fragility_Index_T'] - naturaldisasterrisk_weight * row['Natural_Disaster_Risk_T'] - bcprisk_weight * row['BCP_Risk_T'])) / (-leadtime_weight) if row['SCR_Strength'] == 1 else ""), axis=1)
    # Distance
    df['Distance Required Scaled'] = df.apply(lambda row: ((-0.75) - (-leadtime_weight * row['Lead_Time_T'] - fragilityindex_weight * row['Fragility_Index_T'] - naturaldisasterrisk_weight * row['Natural_Disaster_Risk_T'] - bcprisk_weight * row['BCP_Risk_T'])) / (-distance_weight) if row['SCR_Strength'] == 0 else (((-0.19) - (-leadtime_weight * row['Lead_Time_T'] - fragilityindex_weight * row['Fragility_Index_T'] - naturaldisasterrisk_weight * row['Natural_Disaster_Risk_T'] - bcprisk_weight * row['BCP_Risk_T'])) / (-distance_weight) if row['SCR_Strength'] == 1 else ""), axis=1)
    # Fragility
    df['Fragility Required Scaled'] = df.apply(lambda row: ((-0.75) - (-distance_weight * row['Distance_T'] - leadtime_weight * row['Lead_Time_T'] - naturaldisasterrisk_weight * row['Natural_Disaster_Risk_T'] - bcprisk_weight * row['BCP_Risk_T'])) / (-fragilityindex_weight) if row['SCR_Strength'] == 0 else (((-0.19) - (-distance_weight * row['Distance_T'] - leadtime_weight * row['Lead_Time_T'] - naturaldisasterrisk_weight * row['Natural_Disaster_Risk_T'] - bcprisk_weight * row['BCP_Risk_T'])) / (-fragilityindex_weight) if row['SCR_Strength'] == 1 else ""), axis=1)
    # Natural Disaster Risk
    df['Natural Disaster Required Scaled'] = df.apply(lambda row: ((-0.75) - (-distance_weight * row['Distance_T'] - fragilityindex_weight * row['Fragility_Index_T'] - leadtime_weight * row['Lead_Time_T'] - bcprisk_weight * row['BCP_Risk_T'])) / (-naturaldisasterrisk_weight) if row['SCR_Strength'] == 0 else (((-0.19) - (-distance_weight * row['Distance_T'] - fragilityindex_weight * row['Fragility_Index_T'] - leadtime_weight * row['Lead_Time_T'] - bcprisk_weight * row['BCP_Risk_T'])) / (-naturaldisasterrisk_weight) if row['SCR_Strength'] == 1 else ""), axis=1)
    # BCP Risk
    df['BCP Risk Required Scaled'] = df.apply(lambda row: ((-0.75) - (-distance_weight * row['Distance_T'] - fragilityindex_weight * row['Fragility_Index_T'] - naturaldisasterrisk_weight * row['Natural_Disaster_Risk_T'] - leadtime_weight * row['Lead_Time_T'])) / (-bcprisk_weight) if row['SCR_Strength'] == 0 else (((-0.19) - (-distance_weight * row['Distance_T'] - fragilityindex_weight * row['Fragility_Index_T'] - naturaldisasterrisk_weight * row['Natural_Disaster_Risk_T'] - leadtime_weight * row['Lead_Time_T'])) / (-bcprisk_weight) if row['SCR_Strength'] == 1 else ""), axis=1)

    # df1 = df[df['Lead Time Required Scaled'] != ""]

    # Convert empty strings to NaN in the required scaled columns
    df['Lead Time Required Scaled'] = pd.to_numeric(df['Lead Time Required Scaled'], errors='coerce')
    df['Distance Required Scaled'] = pd.to_numeric(df['Distance Required Scaled'], errors='coerce')
    df['Fragility Required Scaled'] = pd.to_numeric(df['Fragility Required Scaled'], errors='coerce')
    df['Natural Disaster Required Scaled'] = pd.to_numeric(df['Natural Disaster Required Scaled'], errors='coerce')
    df['BCP Risk Required Scaled'] = pd.to_numeric(df['BCP Risk Required Scaled'], errors='coerce')

    # Reverse the scaling for all criteria
    df[['Lead Time Required', 'Distance Required', 'Fragility Required', 'Natural Disaster Required', 'BCP Risk Required']] = scaler.inverse_transform(df[['Lead Time Required Scaled', 'Distance Required Scaled', 'Fragility Required Scaled', 'Natural Disaster Required Scaled', 'BCP Risk Required Scaled']])

    # Pone un "0" si es negativo y deja el valor como esta (x) si es positivo.
    df['Distance Required'][df['Distance Required'] < 0] = 0
    df['Lead Time Required'][df['Lead Time Required'] < 0] = 0
    df['Fragility Required'][df['Fragility Required'] < 0] = 0
    df['Natural Disaster Required'][df['Natural Disaster Required'] < 0] = 0
    # df['BCP Risk Required'][df['BCP Risk Required'] < 0] = -1

    # Round down BCP Risk Required to nearest whole number.
    df['BCP Risk Required'] = df['BCP Risk Required'].apply(lambda x: math.floor(x) if x >= 0 else 0)

    # Assuming AB2 is a variable 'bcprisk_required_scaled' and AG2 is a variable 'bcprisk_required'.  -0.956438593999987 is the scaled value for BCP_Risk = 0.
    df['BCP Risk Required with negative'] = np.where(df['BCP Risk Required Scaled'] < -0.956438593999987, -1, df['BCP Risk Required'])

    # st.dataframe(df)


    def display_kpi_cards(df):
        # Assuming df is the DataFrame from your run_script function
        # Display KPI cards for Lead Time Required, Distance Required, Fragility Required, Natural Disaster Required, and BCP Risk Required with negative

        lead_time_value = df.loc[0, 'Lead Time Required']
        distance_value = df.loc[0, 'Distance Required']
        fragility_value = df.loc[0, 'Fragility Required']
        natural_disaster_value = df.loc[0, 'Natural Disaster Required']
        bcp_risk_value = df.loc[0, 'BCP Risk Required with negative']


        if df.loc[0, 'SCR_Strength'] != 2:
            st.subheader("Target KPIs", anchor=None, help=None, divider="grey")
            
            c8, c9, c10 = st.columns(3)
            # Display "-" if the value is 0 for the first 4 metrics
            c8.metric("Lead Time Required", "-" if lead_time_value == 0 else f"{lead_time_value:.1f} days", delta=f"{(lead_time_value - input1):.1f} days" if lead_time_value != 0 else None, delta_color="inverse")
            c9.metric("Distance Required", "-" if distance_value == 0 else f"{distance_value:.1f} km", delta=f"{(distance_value - input2):.1f} km" if distance_value != 0 else None, delta_color="inverse")

            # Map the values to labels for "BCP Risk Required with negative"
            bcp_risk_labels = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
            # Display "-" if the value is -1, otherwise display the label
            bcp_risk_display = "-" if bcp_risk_value == -1 else bcp_risk_labels.get(bcp_risk_value, f"{bcp_risk_value:.1f}")
            c10.metric("BCP Risk Required", bcp_risk_display)


            c11, c12 = st.columns(2)
            c11.metric("Fragility Index Required", "-" if fragility_value == 0 else f"{fragility_value:.1f}", delta=f"{(fragility_value - float(input9)):.1f}" if fragility_value != 0 else None, delta_color="inverse")
            c12.metric("Natural Disaster Risk Required", "-" if natural_disaster_value == 0 else f"{natural_disaster_value:.1f} %", delta=f"{(natural_disaster_value - float(input10)):.1f} %" if natural_disaster_value != 0 else None, delta_color="inverse")

            st.metric("","")

        # Map the values to labels for "Supply Chain Strength"
        supply_chain_strength_labels = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
        supply_chain_strength_value = df.loc[0, 'SCR_Strength']

        # Map the "Supply Chain Strength" value to "New Supply Chain Strength" label
        new_strength_label = {
            0: 'MEDIUM',
            1: 'HIGH',
            2: 'ALREADY HIGH'
        }

        st.subheader("Target Supply Chain Strength", anchor=None, help=None, divider="grey")
        # Assuming supply_chain_strength_value is the variable containing the strength value
        if supply_chain_strength_value == 0:
            st.metric(":large_orange_circle: :orange[New Supply Chain Strength]", new_strength_label.get(0, "UNKNOWN"))
        elif supply_chain_strength_value == 1:
            st.metric(":large_green_circle: :green[New Supply Chain Strength]", new_strength_label.get(1, "UNKNOWN"))
        else:
            st.metric(":heavy_check_mark: :green[New Supply Chain Strength]", new_strength_label.get(2, "UNKNOWN"))




    # Call the display_kpi_cards function with the DataFrame
    display_kpi_cards(df)


def render_ahp_page():
    st.title("Weight Calculation")
    st.write("This section describes the process followed to calculate the weights asigned to each KPI prior to calculating the Supply Chain Strength for a given supplier.")
    st.header("AHP Matrix", anchor=None, help="For more information about the matrix construction process, check out the Wikipedia article below.", divider="grey")
    st.image('AHP.png')

    st.header("Analytic hierarchy process", anchor=None, help=None, divider="grey")
    # Add more code specific to the Natural Disaster Risk page if needed
    st.markdown(
    """
    <iframe src="https://en.wikipedia.org/wiki/Analytic_hierarchy_process" width="100%" height="800"></iframe>
    """,
    unsafe_allow_html=True
    )

# Function to render the Fragility Index page
def render_fragility_index_page():
    st.title("Fragility Index")
    st.write("In the below Wikipedia article there is a table containing latest Fragility Indexes for every country. The tool accesses the website and retrieves the values for the selected supplier country.")
    st.header("", anchor=None, help=None, divider="grey")
    # Add more code specific to the Fragility Index page if needed
    st.markdown(
    """
    <iframe src="https://en.wikipedia.org/wiki/List_of_countries_by_Fragile_States_Index" width="100%" height="800"></iframe>
    """,
    unsafe_allow_html=True
    )

# Function to render the Natural Disaster Risk page
def render_natural_disaster_page():
    st.title("Natural Disaster Risk")
    st.write("In the below Wikipedia article there is a table containing the latest scores for Natural Disaster Risk for every country. The tool accesses the website and retrieves the values for the selected supplier country.")
    st.header("", anchor=None, help=None, divider="grey")
    # Add more code specific to the Natural Disaster Risk page if needed
    st.markdown(
    """
    <iframe src="https://en.wikipedia.org/wiki/List_of_countries_by_natural_disaster_risk" width="100%" height="800"></iframe>
    """,
    unsafe_allow_html=True
    )



# Main function
def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Simulator", "Weight Calculation", "Fragility Index", "Natural Disaster Risk"],
            icons=["calculator", "123", "wikipedia", "wikipedia"],
            menu_icon="list",
            default_index=0,
        )

    # Render the selected page
    if selected == "Simulator":
        render_calculator_page()
    elif selected == "Fragility Index":
        render_fragility_index_page()
    elif selected == "Natural Disaster Risk":
        render_natural_disaster_page()
    elif selected == "Weight Calculation":
        render_ahp_page()

if __name__ == "__main__":
    main()
