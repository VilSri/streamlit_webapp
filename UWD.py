import random
import datetime
import pandas as pd
from faker import Faker
import numpy as np
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import streamlit as st
import plotly.express as px
from numerize.numerize import numerize
import plotly.graph_objects as go

# Function to generate loan deviations
def generate_loan_deviation(credit_score, dti_ratio, ltv_ratio, property_type):
    deviation_reasons = ['null','Insufficient credit score or credit history.', 
                         'Inadequate or unstable income.', 
                         'High debt-to-income ratio.', 
                         'Limited assets or insufficient down payment.', 
                         'Property appraisal below the purchase price.', 
                         'Undisclosed debts or liabilities.', 
                         'Incomplete or inaccurate loan application.']
    deviation_probability = 0.25
    
    deviation = ''
    if credit_score < 600:
        deviation = deviation_reasons[0]
    elif dti_ratio > 0.43:
        deviation = deviation_reasons[2]
    elif ltv_ratio > 0.8:
        deviation = deviation_reasons[3]
    elif property_type == 'Condominium':
        deviation = deviation_reasons[4]
    elif random.random() < deviation_probability:
        deviation = random.choice(deviation_reasons)
    
    return deviation

# Setting up the loan origination dates range
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date.today()

# Creating dictionary for each record
loan_data = []
for i in range(1500):
    record = {}
    record['Application ID'] = i + 1
    record['Existing customer'] = random.choice([True, False])
    record['Product Name'] = random.choice(['30 Year Fixed', '15 Year Fixed', 'Adjustable Rate Mortgage', 'FHA Loan'])
    record['Borrower age'] = random.randint(21, 80)
    record['Loan-to-value ratio (LTV)'] = random.randint(50, 100)
    record['Debt-to-income ratio (DTI)'] = random.randint(20, 50)
    record['Credit score'] = random.randint(500, 850)
    record['Employment history'] = random.choice(['1-2 years', '2-5 years', '5-10 years', '10+ years'])
    record['Property type'] = random.choice(['Single Family', 'Condominium', 'Townhouse', 'Multi-Family'])
    record['Loan purpose'] = random.choice(['Purchase', 'Refinance', 'Cash-out Refinance'])
    record['Loan amount'] = round(random.uniform(50000, 1000000), 2)
    record['Loan term'] = random.choice([15, 20, 30, 40])
    record['Interest rate'] = round(random.uniform(2, 8), 2)
    record['Loan origination date'] = start_date + datetime.timedelta(days=random.randint(0, (end_date - start_date).days))
    record['Loan deviation'] = generate_loan_deviation(record['Credit score'], record['Debt-to-income ratio (DTI)'], record['Loan-to-value ratio (LTV)'], record['Property type'])
    record['Requested Amount'] = record['Loan amount']
    record['Approved Amount'] = 0
    record['App to Account Booking (AHT)'] = 0
    record['Days to Decide'] = random.randint(1, 30)
    loan_data.append(record)

# Loan decision based on loan deviation, loan amount, and credit score
for record in loan_data:
    if record['Loan deviation'] in ['null']:
        record['Loan decision'] = 'Approved'
        record['Approved Amount'] = record['Loan amount']
    elif record['Loan deviation'] in ['Insufficient credit score or credit history.', 'Incomplete or inaccurate loan application.']:
        record['Loan decision'] = 'Rejected'
        record['Approved Amount'] = 0
    elif record['Loan deviation'] in ['Inadequate or unstable income.', 'High debt-to-income ratio.', 'Limited assets or insufficient down payment.', 'Property appraisal below the purchase price.', 'Undisclosed debts or liabilities.']:
        if record['Credit score'] >= 700 and record['Loan amount'] <= 1000000:
            record['Loan decision'] = 'Conditional Approval'
            record['Approved Amount'] = round(random.uniform(0.85, 0.95) * record['Loan amount'], 2)
        else:
            record['Loan decision'] = 'Rejected'
            record['Approved Amount'] = 0
    elif record['Loan deviation'] == 'Multiple loan deviations':
        if record['Credit score'] >= 720 and record['Loan amount'] <= 1500000:
            record['Loan decision'] = 'Conditional Approval'
            record['Approved Amount'] = round(random.uniform(0.85, 0.95) * record['Loan amount'], 2)
        else:
            record['Loan decision'] = 'Rejected'
            record['Approved Amount'] = 0
            
    # Adding new columns for requested amount and AHT
    record['Requested Amount'] = record['Loan amount']
    record['App to Account Booking (AHT)'] = random.randint(14, 30)
    
# Creating a pandas DataFrame from the loan data
df_loan_data = pd.DataFrame(loan_data, columns=['Application ID', 'Existing customer', 'Product Name', 'Borrower age', 
                                                 'Loan-to-value ratio (LTV)', 'Debt-to-income ratio (DTI)', 'Credit score', 
                                                 'Employment history', 'Property type', 'Loan purpose', 'Loan amount', 
                                                 'Loan term', 'Interest rate', 'Loan origination date', 'Loan deviation', 
                                                 'Loan decision', 'Requested Amount', 'Approved Amount', 'App to Account Booking (AHT)'])


# Create list of underwriters using faker
fake = Faker()
underwriters = [{'Employee ID': fake.uuid4(), 'Employee Name': fake.name()} for _ in range(15)]

# Repeat underwriters data 100 times
underwriters_data = [underwriters[i % len(underwriters)] for i in range(1500)]

# Create list of addresses using faker
addresses = [{'Address': fake.street_address(), 'Street': fake.street_name(),
              'City': fake.city(), 'State': fake.state()} for _ in range(1500)]

# Function to generate exposure types based on location
def generate_exposure_type(state):
    if state in ['OK', 'KS', 'NE']:
        return 'Tornadoes and Severe Storms'
    elif state in ['FL', 'LA', 'TX']:
        return 'Hurricanes and Tropical Storms'
    elif state in ['TX', 'LA', 'MO']:
        return 'Floods'
    elif state in ['CA', 'OR', 'WA']:
        return 'Wildfires'
    elif state in ['CA']:
        return 'Earthquakes'
    elif state in ['AZ', 'CA']:
        return 'Drought'
    else:
        return 'Incidents of Mass Violence'
    
# Generate exposure type for each address state
exposures = [generate_exposure_type(address['State']) for address in addresses]

# Add new columns to dataframe
df_loan_data['Employee ID'] = [underwriter['Employee ID'] for underwriter in underwriters_data]
df_loan_data['Employee Name'] = [underwriter['Employee Name'] for underwriter in underwriters_data]
df_loan_data['Address'] = [address['Address'] for address in addresses]
df_loan_data['Street'] = [address['Street'] for address in addresses]
df_loan_data['City'] = [address['City'] for address in addresses]
df_loan_data['State'] = [address['State'] for address in addresses]
df_loan_data['Exposure Type'] = exposures

# Add exposure type data to dataframe
df_loan_data['Exposure'] = [generate_exposure_type(state) for state in df_loan_data['State']]



# Generate 50 records for the focus_queue DataFrame
num_records = 17
application_ids = range(15000, 19001)
product_names = ['30 Year Fixed', '15 Year Fixed', 'Adjustable Rate Mortgage', 'FHA Loan']
time_to_decide = range(1, 15)
credit_scores = range(600, 991)

# Generate the list of dictionaries
records = []
for _ in range(num_records):
    record = {
        'Application ID': random.choice(application_ids),
        'Product Name': random.choice(product_names),
        'Time to Decide (days)': random.choice(time_to_decide),
        'Loan Decision (Predicted)': None,  # Set the initial value to None
        'Credit Score': random.choice(credit_scores),
        'Loan-to-value ratio (LTV)': random.randint(50, 100),
        'Debt-to-income ratio (DTI)': random.randint(20, 50),
        'Reason1': '',
        'Reason2': '',
        'Reason3': '',
        
    }

    # Set loan decision based on product, credit score, loan-to-value ratio, and debt-to-income ratio
    if record['Product Name'] == '30 Year Fixed':
        if record['Credit Score'] >= 700 and record['Loan-to-value ratio (LTV)'] <= 80 and record['Debt-to-income ratio (DTI)'] <= 45:
            record['Loan Decision (Predicted)'] = 'Approved'
            record['Reason1'] = 'Strong credit score and credit history'
            record['Reason2'] = 'Stable and sufficient income'
            record['Reason3'] = 'Low debt-to-income ratio'
        elif record['Credit Score'] >= 600 and record['Loan-to-value ratio (LTV)'] <= 90 and record['Debt-to-income ratio (DTI)'] <= 50:
            record['Loan Decision (Predicted)'] = 'Conditional Approval'
            record['Reason1'] = 'Incomplete or inaccurate loan application'
            record['Reason2'] = 'Insufficient down payment'
            record['Reason3'] = 'Undisclosed debts or liabilities'
        else:
            record['Loan Decision (Predicted)'] = 'Rejected'
            record['Reason1'] = 'Inadequate or unstable income'
            record['Reason2'] = 'High debt-to-income ratio'
            record['Reason3'] = 'Limited assets or insufficient down payment'

    elif record['Product Name'] == '15 Year Fixed':
        if record['Credit Score'] >= 680 and record['Loan-to-value ratio (LTV)'] <= 80 and record['Debt-to-income ratio (DTI)'] <= 45:
            record['Loan Decision (Predicted)'] = 'Approved'
            record['Reason1'] = 'Good credit score and credit history'
            record['Reason2'] = 'Stable and sufficient income'
            record['Reason3'] = 'Low debt-to-income ratio'
        else:
            record['Loan Decision (Predicted)'] = 'Rejected'
            record['Reason1'] = 'Inadequate credit score'
            record['Reason2'] = 'High debt-to-income ratio'
            record['Reason3'] = 'Limited assets or insufficient down payment'

    elif record['Product Name'] == 'Adjustable Rate Mortgage':
        if record['Credit Score'] >= 650 and record['Loan-to-value ratio (LTV)'] <= 85 and record['Debt-to-income ratio (DTI)'] <= 50:
            record['Loan Decision (Predicted)'] = 'Approved'
            record['Reason1'] = 'Decent credit score and credit history'
            record['Reason2'] = 'Stable income'
            record['Reason3'] = 'Reasonable debt-to-income ratio'
        else:
            record['Loan Decision (Predicted)'] = 'Rejected'
            record['Reason1'] = 'Insufficient credit score'
            record['Reason2'] = 'High debt-to-income ratio'
            record['Reason3'] = 'Limited assets or insufficient down payment'

    elif record['Product Name'] == 'FHA Loan':
        if record['Credit Score'] >= 580 and record['Loan-to-value ratio (LTV)'] <= 90 and record['Debt-to-income ratio (DTI)'] <= 50:
            record['Loan Decision (Predicted)'] = 'Approved'
            record['Reason1'] = 'Fair credit score and credit history'
            record['Reason2'] = 'Stable income'
            record['Reason3'] = 'Reasonable debt-to-income ratio'
        else:
            record['Loan Decision (Predicted)'] = 'Rejected'
            record['Reason1'] = 'Low credit score'
            record['Reason2'] = 'High debt-to-income ratio'
            record['Reason3'] = 'Limited assets or insufficient down payment'

    record['Loan Decision (Pred Confid)'] = random.randint(85, 95)
    record['Loan amount'] = round(random.uniform(50000, 1000000), 2)
    record['Borrower age'] = random.randint(21, 80)

    records.append(record)

# Create the DataFrame from the list of dictionaries
focus_queue = pd.DataFrame(records)

# Fill missing values in "Loan Decision (Predicted)" column with an empty string
focus_queue["Loan Decision (Predicted)"].fillna("", inplace=True)




###################Streamlit dashboard Presentation layer#####################3
# Set page configuration
st.set_page_config(
    page_title='Assisted Underwriting Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Define the SessionState class
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Function to get the data with caching
@st.cache_data
def get_data():
    return df_loan_data

# Get the data using the caching function
df = get_data()

# Create a SessionState object
session_state = SessionState(
    existing_customer=[],
    product_name=[],
    selected_min_age=int(df['Borrower age'].min()),
    selected_max_age=int(df['Borrower age'].max()),
    selected_LTV=[int(df['Loan-to-value ratio (LTV)'].min()), int(df['Loan-to-value ratio (LTV)'].max())],
    selected_DTI=[int(df['Debt-to-income ratio (DTI)'].min()), int(df['Debt-to-income ratio (DTI)'].max())],
    selected_Creditscore=[int(df['Credit score'].min()), int(df['Credit score'].max())],
    selected_Loanamount=[int(df['Loan amount'].min()), int(df['Loan amount'].max())],
    property_type=[],
    loan_purpose=[],
    loan_decision=[],
    city=[],
    state=[]
)

header_left, header_mid, header_right = st.columns([1, 2, 1], gap='large')

with header_mid:
    st.markdown("<h1 style='font-size: 23px; padding-top: 5px; padding-bottom: 5px;'>AI-Underwrite Dashboard</h1>", unsafe_allow_html=True)
st.divider()

# Create the sidebar filters with cached values
with st.sidebar:
    existing_customer = st.multiselect(
        label='Select Customer Type',
        options=df['Existing customer'].unique(),
        default=df['Existing customer'].unique(),
    )
    session_state.existing_customer = existing_customer

    product_name = st.multiselect(
        label='Select Product',
        options=df['Product Name'].unique(),
        default=df['Product Name'].unique(),
    )
    session_state.product_name = product_name

    selected_min_age, selected_max_age = st.slider(
        label='Select Customer Age',
        value=[session_state.selected_min_age, session_state.selected_max_age],
        min_value=int(df['Borrower age'].min()),
        max_value=int(df['Borrower age'].max())
    )
    session_state.selected_min_age = selected_min_age
    session_state.selected_max_age = selected_max_age

    selected_LTV = st.slider(
        label='Select Loan to Value (%)',
        value=session_state.selected_LTV,
        min_value=int(df['Loan-to-value ratio (LTV)'].min()),
        max_value=int(df['Loan-to-value ratio (LTV)'].max())
    )
    session_state.selected_LTV = selected_LTV

    selected_DTI = st.slider(
        label='Select Debt to Income value (%)',
        value=session_state.selected_DTI,
        min_value=int(df['Debt-to-income ratio (DTI)'].min()),
        max_value=int(df['Debt-to-income ratio (DTI)'].max())
    )
    session_state.selected_DTI = selected_DTI

    selected_Creditscore = st.slider(
        label='Select Credit Score',
        value=session_state.selected_Creditscore,
        min_value=int(df['Credit score'].min()),
        max_value=int(df['Credit score'].max())
    )
    session_state.selected_Creditscore = selected_Creditscore

    selected_Loanamount = st.slider(
        label='Select Loan Amount',
        value=session_state.selected_Loanamount,
        min_value=int(df['Loan amount'].min()),
        max_value=int(df['Loan amount'].max())
    )
    session_state.selected_Loanamount = selected_Loanamount

    property_type = st.multiselect(
        label='Select Property Type',
        options=df['Property type'].unique(),
        default=df['Property type'].unique(),
    )
    session_state.property_type = property_type

    loan_purpose = st.multiselect(
        label='Select Loan Purpose',
        options=df['Loan purpose'].unique(),
        default=df['Loan purpose'].unique(),
    )
    session_state.loan_purpose = loan_purpose

    loan_decision = st.multiselect(
        label='Select Loan Decision',
        options=df['Loan decision'].unique(),
        default=df['Loan decision'].unique(),
    )
    session_state.loan_decision = loan_decision

    #city = st.multiselect(
    #    label='Select City',
    #    options=df['City'].unique(),
    #    default=df['City'].unique(),
    #)
    #session_state.city = city

    #state = st.multiselect(
    #    label='Select State',
    #    options=df['State'].unique(),
    #    default=df['State'].unique(),
    #)
    #session_state.state = state

# Apply the filters to the dataframe
filtered_data = df.query('`Product Name` in @product_name and `Existing customer` in @existing_customer and `Property type` in @property_type and \
                          `Borrower age` >= @selected_min_age and `Borrower age` <= @selected_max_age and \
                          `Loan-to-value ratio (LTV)` >= @selected_LTV[0] and `Loan-to-value ratio (LTV)` <= @selected_LTV[1] and \
                          `Debt-to-income ratio (DTI)` >= @selected_DTI[0] and `Debt-to-income ratio (DTI)` <= @selected_DTI[1] and \
                          `Credit score` >= @selected_Creditscore[0] and `Credit score` <= @selected_Creditscore[1] and \
                          `Loan amount` >= @selected_Loanamount[0] and `Loan amount` <= @selected_Loanamount[1] and \
                          `Loan purpose` in @loan_purpose and \
                          `Loan decision` in @loan_decision')

#st.write(filtered_data)

# Metrics Definition
Total_App_Vol = float(filtered_data["Application ID"].count())
Avg_Credit_score = float(filtered_data["Credit score"].mean())
Avg_Loan_Value = float(filtered_data["Loan amount"].mean())/1000

# Calculate the count of each loan decision within the filtered data
approval_count = float(filtered_data["Loan decision"].eq("Approved").sum())
rejection_count = float(filtered_data["Loan decision"].eq("Rejected").sum())
conditional_approval_count = float(filtered_data["Loan decision"].eq("Conditional Approval").sum())

# Calculate the average rates by dividing the counts by the total application volume
Approval_Rate = approval_count / Total_App_Vol
Rejection_Rate = rejection_count / Total_App_Vol
cond_Approval_Rate = conditional_approval_count / Total_App_Vol

total1, total2, total3, total4, total5, total6 = st.columns(6, gap='small')

with total1:
    st.markdown("<style>.metric-label, .metric-value { font-size: 14px !important; }</style>", unsafe_allow_html=True)
    st.image('images/appvol.png', width=45)
    st.metric(label="Total Application", value=numerize(Total_App_Vol))

with total2:
    st.markdown("<style>.metric-label, .metric-value { font-size: 14px !important; }</style>", unsafe_allow_html=True)
    st.image('images/creditscore.png', width=45)
    value = round(Avg_Credit_score)
    st.metric(label="Credit Score", value="{:.0f}".format(value), delta=None)

with total3:
    st.markdown("<style>.metric-label, .metric-value { font-size: 14px !important; }</style>", unsafe_allow_html=True)
    st.image('images/Loanval2.png', width=45)
    value3 = round(Avg_Loan_Value)
    st.metric(label="Loan Value", value="{:.0f}k".format(value3), delta=None)  # Displayed in thousands ($k)

with total4:
    st.markdown("<style>.metric-label, .metric-value { font-size: 14px !important; }</style>", unsafe_allow_html=True)
    st.image('images/Approved.png', width=45)
    value4 = round(Approval_Rate * 100, 2)  # Round to 2 decimal places
    st.metric(label="Approval Rate", value="{:.0f}%".format(value4), delta=None)

with total5:
    st.markdown("<style>.metric-label, .metric-value { font-size: 14px !important; }</style>", unsafe_allow_html=True)
    st.image('images/Rejected.png', width=45)
    value5 = round(Rejection_Rate * 100, 2)  # Round to 2 decimal places
    st.metric(label="Rejection Rate", value="{:.0f}%".format(value5), delta=None)

with total6:
    st.markdown("<style>.metric-label, .metric-value { font-size: 14px !important; }</style>", unsafe_allow_html=True)
    st.image('images/Conditionalapproval.png', width=45)
    value6 = round(cond_Approval_Rate * 100, 2)  # Round to 2 decimal places
    st.metric(label="Conditional Approval Rate", value="{:.0f}%".format(value6), delta=None)

# Define the metrics
approval_count = float(filtered_data["Loan decision"].eq("Approved").sum())
rejection_count = float(filtered_data["Loan decision"].eq("Rejected").sum())
conditional_approval_count = float(filtered_data["Loan decision"].eq("Conditional Approval").sum())

# Extract month and year from "Loan origination date" column
df_loan_data["Loan origination date"] = pd.to_datetime(df_loan_data["Loan origination date"])
df_loan_data["Month_Year"] = df_loan_data["Loan origination date"].dt.strftime("%b%Y")

# Group the data by 'Product Name' and 'Loan Decision' and calculate the count
grouped_data = filtered_data.groupby(['Product Name', 'State', 'Loan decision']).size().reset_index(name='Count')

# Create a treemap chart using Plotly Express
fig1 = px.treemap(grouped_data, path=['Product Name', 'State', 'Loan decision'], values='Count')

# Define the metrics
approval_count = float(filtered_data["Loan decision"].eq("Approved").sum())
rejection_count = float(filtered_data["Loan decision"].eq("Rejected").sum())
conditional_approval_count = float(filtered_data["Loan decision"].eq("Conditional Approval").sum())

# Extract month and year from "Loan origination date" column
df_loan_data["Loan origination date"] = pd.to_datetime(df_loan_data["Loan origination date"])
df_loan_data["Month_Year"] = df_loan_data["Loan origination date"].dt.strftime("%b%Y")

# Group the data by 'Product Name' and 'Loan Decision' and calculate the count
grouped_data = filtered_data.groupby(['Product Name', 'State', 'Loan decision']).size().reset_index(name='Count')

# Create a treemap chart using Plotly Express
fig1 = px.treemap(grouped_data, path=['Product Name', 'State', 'Loan decision'], values='Count')

#########Trend chart
def get_chart_35532020(df_loan_data):
    # Define the metrics
    approval_count = float(df_loan_data["Loan decision"].eq("Approved").sum())
    rejection_count = float(df_loan_data["Loan decision"].eq("Rejected").sum())
    conditional_approval_count = float(df_loan_data["Loan decision"].eq("Conditional Approval").sum())

    # Extract quarter and year from "Loan origination date" column
    df_loan_data["Loan origination date"] = pd.to_datetime(df_loan_data["Loan origination date"])
    df_loan_data["Quarter_Year"] = df_loan_data["Loan origination date"].dt.to_period("Q").astype(str)

    # Group by "Quarter_Year" and "Loan decision" and count occurrences
    grouped_data = df_loan_data.groupby(["Quarter_Year", "Loan decision"]).size().reset_index(name="Count")

    # Create the figure using Plotly
    fig_trend = go.Figure()

    # Add filled area plot for each loan decision category
    colors = ["skyblue", "blueviolet", "red"]
    categories = ["Approved", "Conditional Approval", "Rejected"]
    for category, color in zip(categories, colors):
        data = grouped_data[grouped_data["Loan decision"] == category]
        fig_trend.add_trace(go.Scatter(
            x=data["Quarter_Year"],
            y=data["Count"],
            mode="lines",
            name=category,
            fill='tozeroy',
            line=dict(color=color)
        ))

    # Set x-axis labels as Quarter and Year
    fig_trend.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=df_loan_data["Quarter_Year"].unique(),
            ticktext=df_loan_data["Loan origination date"].dt.strftime("Q%q'%y").unique()
        ),
        yaxis=dict(title="Count"),
        showlegend=True
    )

    return fig_trend



@st.cache_data
def get_chart_12007130(filtered_data):
    # Map loan decision categories to colors
    color_map = {"Approved": "green", "Rejected": "red", "Conditional Approval": "yellow"}

    # Convert loan amount to thousands USD
    filtered_data["Loan amount (in thousands USD)"] = filtered_data["Loan amount"] / 1000

    # Create the plot using Plotly Express
    fig_scp = px.scatter(filtered_data, x="Loan amount (in thousands USD)", y="City", color="Loan decision",
                     color_discrete_map=color_map, title="Loan Decisions by City",
                     labels={"Loan amount (in thousands USD)": "Loan Amount (in thousands USD)", "City": "City"})

    return fig_scp



#Tab views

tab1, tab2, tab3,tab4,tab5 = st.tabs(["Decision Trend", "Decision by City", "Decision by Product","Focus 'Q'","AI-Assist"])
# Render the tabs
with tab1:
    fig_trend = get_chart_35532020(df_loan_data)
    st.plotly_chart(fig_trend, use_container_width=True, theme="streamlit")

with tab2:
    fig_scp = get_chart_12007130(filtered_data)
    st.plotly_chart(fig_scp, use_container_width=True)

with tab3:
    st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

with tab4:
    uwpipeline,gap, AIbox = st.columns([1.75,0.25, 1], gap='small')
    with uwpipeline:
        @st.cache_data
        def get_data():
            return focus_queue

        # Get the data using the caching function
        fq_data = get_data()

        gd = GridOptionsBuilder.from_dataframe(fq_data)
        gd.configure_pagination(enabled=True, paginationAutoPageSize=True, paginationPageSize=15)
        gd.configure_default_column(editable=True, groupable=True)
        gd.configure_selection(selection_mode='single', use_checkbox=True)

        cellstyle_jscode = JsCode("""
        function(params) {
            var value = params.value;
            if (params.value === 'Approved') {
                return {
                    'color': 'white',
                    'backgroundColor': 'green'
                };
            }
            if (params.value === 'Conditional Approval') {
                return {
                    'color': 'black',
                    'backgroundColor': 'yellow'
                };
            }
            if (params.value === 'Rejected') {
                return {
                    'color': 'white',
                    'backgroundColor': 'red'
                };
            }
            if (value > 10 && value <=30) {
                return {
                    'color': 'white',
                    'backgroundColor': 'blue'
                };
            }
            if (value >= 4 && value <= 10) {
                return {
                    'color': 'black',
                    'backgroundColor': 'grey'
                };
            }
            if (value >= 1 && value <= 3) {
                return {
                    'color': 'white',
                    'backgroundColor': 'red'
                };
            }
            return null;
        }
        """)

        gd.configure_column('Loan Decision (Predicted)', cellStyle=cellstyle_jscode)
        gd.configure_column('Time to Decide (days)', cellStyle=cellstyle_jscode)
        gd.configure_column('Loan Decision (Pred Confid)', cellStyle=cellstyle_jscode)

        # Build the grid options
        grid_options = gd.build()


        # Display the AgGrid table
        grid_table = AgGrid(fq_data,
            gridOptions=grid_options,
            update_mode="SELECTION_CHANGED",
            fit_columns_on_grid_load=True,
            #height=700,
            allow_unsafe_jscode=True,
            theme='streamlit'
        )

        # Get the selected rows
        sel_row = grid_table["selected_rows"]

        # Create a dataframe with the selected rows
        sel_row_df = pd.DataFrame(sel_row)

        # Configure the GridOptionsBuilder for the selected rows dataframe
        gd_selected_rows = GridOptionsBuilder.from_dataframe(sel_row_df)
        gd_selected_rows.configure_default_column(editable=True, groupable=True)
        
        # Define the cell style for data bars
        cellstyle_jscode_selected_rows = JsCode("""
        function(params) {
            var value = params.value;
            var dataBarWidth = value + "%";
            var style = {
                display: 'flex',
                alignItems: 'center',
                padding: '2px',
                width: '100%'
            };
            var barStyle = {
                width: dataBarWidth,
                height: '10px',
                backgroundColor: '#5cb85c'  // Green color for data bar
            };
            var valueStyle = {
                marginLeft: '4px',
                fontWeight: 'bold'
            };
            return {
                'div': {
                    style: style,
                    children: [
                        {
                            style: barStyle
                        },
                        {
                            style: valueStyle,
                            children: value
                        }
                    ]
                }
            };
        }
        """)

        # Configure the column with data bar for selected rows dataframe
        gd_selected_rows.configure_column('Loan Decision (Pred Confid)', cellStyle=cellstyle_jscode_selected_rows)

        # Build the grid options for selected rows dataframe
        grid_options_selected_rows = gd_selected_rows.build()

        # Display the AgGrid table for selected rows dataframe
        #grid_table_selected_rows = AgGrid(
        #    sel_row_df,
        #    gridOptions=grid_options_selected_rows,
        #    update_mode='SELECTION_CHANGED',
        #    height=100,
        #    allow_unsafe_jscode=True,
        #    theme='streamlit'
        #)
    with gap:
        st.write("")
    with AIbox:
        sel_row = grid_table["selected_rows"]
        # Convert sel_row to a dataframe
        sel_row_df = pd.DataFrame(sel_row)
        if not sel_row_df.empty:
            # Fill missing values in "Loan Decision (Predicted)" column with an empty string
            sel_row_df["Loan Decision (Predicted)"].fillna("", inplace=True)
            Appids = ", ".join(sel_row_df["Application ID"].astype(str).tolist())
            product_names = ", ".join(sel_row_df["Product Name"].tolist())
            loan_decisions = ", ".join(sel_row_df["Loan Decision (Predicted)"].tolist())
            confidences = ", ".join(sel_row_df["Loan Decision (Pred Confid)"].astype(str).tolist())
            Reason1s = ", ".join(sel_row_df["Reason1"].astype(str).tolist())
            Reason2s = ", ".join(sel_row_df["Reason2"].astype(str).tolist())
            Reason3s = ", ".join(sel_row_df["Reason3"].astype(str).tolist())
            
            with st.container():
                #st.write("")  # Empty line for line break
                #st.write("")  # Empty line for line break
                #st.write("")  # Empty line for line break
                st.markdown("<h1 style='text-align: center; font-size: 20px; margin-top: 5px; margin-bottom: 0px;'>AI driven insights</h1>", unsafe_allow_html=True)
                #st.markdown("---")
                st.write("Selected Application: {}".format(Appids))
                st.write("Product: {}".format(product_names))
                
                # Calculate SLA Breach Likelihood
                sel_row_df["SLA Breach (Likelihood)"] = ((30 - sel_row_df["Time to Decide (days)"]) / 30)

                # Retrieve the SLA Breach Likelihood value
                SLA_breach_likelihood = sel_row_df["SLA Breach (Likelihood)"].iloc[0]

                # Display SLA Breach Likelihood
                SLA_breach_pct = round(SLA_breach_likelihood*100)
                st.write("SLA Breach (Likelihood): {}%".format(SLA_breach_pct))
                st.progress(SLA_breach_likelihood)



                if "Approved" in loan_decisions:
                    st.markdown("<span style='color:lightgreen'>Predicted Decision: {}</span>".format(loan_decisions), unsafe_allow_html=True)
                elif "Conditional Approval" in loan_decisions:
                    st.markdown("<span style='color:yellow'>Predicted Decision: {}</span>".format(loan_decisions), unsafe_allow_html=True)
                elif "Rejected" in loan_decisions:
                    st.markdown("<span style='color:pink'>Predicted Decision: {}</span>".format(loan_decisions), unsafe_allow_html=True)
                else:
                    st.write("Predicted Decision: {}".format(loan_decisions))
                
                
                for confidence in confidences.split(","):
                    confidence = float(confidence)
                    normalized_confidence = confidence / 100.0  # Normalize confidence between 0 and 1
                    if confidence > 95:
                        #st.markdown("<span style='color:green'>Prediction confidence: {}%</span>".format(confidence), unsafe_allow_html=True)
                        st.write("Prediction confidence: {}%".format(confidence))
                        st.progress(normalized_confidence)

                    elif 85 <= confidence <= 95:
                        #st.markdown("<span style='color:yellow'>Prediction confidence: {}%</span>".format(confidence), unsafe_allow_html=True)
                        st.write("Prediction confidence: {}%".format(confidence))
                        st.progress(normalized_confidence)
                    else:
                        #st.markdown("<span style='color:red'>Prediction confidence: {}%</span>".format(confidence), unsafe_allow_html=True)
                        st.write("Prediction confidence: {}%".format(confidence))
                        st.progress(normalized_confidence)

                st.markdown("<h1 style='text-align: left; font-size: 18px; margin-top: 0px; margin-bottom: 0px;'>Influencing Reasons on Loan Decision</h1>", unsafe_allow_html=True)
                st.markdown("     - R1: {}".format(Reason1s))
                st.markdown("     - R2: {}".format(Reason2s))
                st.markdown("     - R3: {}".format(Reason3s))
        else:
            #st.write("Select an Application for AI insights.")
           st.write("")  # Empty line for line break
           st.write("")  # Empty line for line break
           st.write("")  # Empty line for line break
           st.write("")  # Empty line for line break
           st.write("")  # Empty line for line break
           st.write("")  # Empty line for line break
           st.markdown("<h1 style='text-align: center; font-size: 25px; margin-top: 5px; margin-bottom: 0px;'>Select a case for AI insights</h1>", unsafe_allow_html=True)
           #st.markdown("---")

        
with tab5:
    # Predefined list of questions and corresponding answers
    questions = ["top 3 cities with high rejection rates", 
                 "What age group have good approval rates"]
    answers = ["Sure! Here are the top 3 cities with high rejection rates.    \n\t 1)New York - 27%    \n\t 2)California - 32%    \n\t 3)Texas - 17%", 
               "Below is the distribution of approval rates by age group.    \n\t 1)Age group = 25-35, Approval Rate = 72%    \n\t 2)Age group = 36-50, Approval Rate = 60%    \n\t 3)Age group = 50+, Approval Rate = 17%" 
               ]

    # Display the text input and submit button
    user_text = st.text_input("Enter your question:")
    submit_button = st.button("Submit")

    if submit_button:
        if user_text in questions:
            index = questions.index(user_text)
            st.success(answers[index])
        else:
            st.error("Sorry, I don't have an answer to that question.")
