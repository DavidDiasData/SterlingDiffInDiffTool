import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from io import StringIO
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import plotly.express as px

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Diff-in-Diff Tool'
)

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
st.header('Difference-in-Differences Analysis Tool', divider='gray')


st.caption("Tool to use Difference-in-Differences based on Matheus Facure Alves Causal Inference for the brave and true handbook.")

st.caption("Diff-in-diff definition")
st.caption("Use case")
st.caption("Diff-in-diff definition")
st.caption("Approaches and things to keep an eye on")


st.subheader("Guidance", divider="gray")

# Add some spacing


#st.caption('Y  = β0 + β1 TREAT + β2 POST + β3 TREAT*POST + e')
st.latex(r'''Y_dt  = β_0 + β_1 TREAT_d + β_2 POST_t + β_3 TREAT_d*POST_t + e_dt  ''')

st.markdown('**How to use the app?**')
st.caption('1. Upload a file (CSV files only*)')
st.caption('2. Choose the columns you want for the analysis')
st.caption('3. Run the analysis and get the insights')


st.markdown('**Your data should have at least 4 columns:**')
st.caption('1. Event Date: Daily, weekly, monthly event date')
st.caption('2. Metric: Numeric / float column with the metric you want to observe. Purchases, Clicks, actions, etc.')
st.caption('3. Treatment: A boolean column with the followings data values: (0 = No treatment; 1 = Treatment)')
st.caption('4. Post: A boolean column related with the event date column to indicate the post treatment periods with the followings data values: (0 = Before treatment; 1 = After Treatment)')

#@st.cache_data
#def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
#    return df.to_csv().encode("utf-8")

#sample_csv = convert_df("/Causal inference - diff-in-diff - Raw Data - Example (Sterling example).csv")

st.download_button(
    label="Download sample data",
    data="/Causal inference - diff-in-diff - Raw Data - Example (Sterling example).csv",
    file_name="diff-in-diff-sample-data.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("Upload a file (CSV files only*)")
if uploaded_file is not None:

    # Can be used wherever a "file-like" object is accepted:
   dataframe = pd.read_csv(uploaded_file)
   col1, col2 = st.columns(2)
   with col1:
         st.write("Take a look od your data")
         st.write(dataframe.head(10))


   with col2:
    
         event_date_name = st.selectbox(
    "Select your event date column",
    (list(dataframe.columns)))
    #st.write("You selected:", dataframe[event_date_values])
         metric_name = st.selectbox(
    "Select your metric column",
    (list(dataframe.columns)))
         groups_name = st.selectbox(
    "Select your groups column",
    (list(dataframe.columns)))
         intervention_date_name = st.selectbox(
    "Select your intervention date column",
    (list(dataframe.columns)))
   st.write("Your model:", str(metric_name) + '~' + str(groups_name) + '*' +  str(intervention_date_name) )
   st.latex(r'''Y_dt  = β_0 + β_1 TREAT_d + β_2 POST_t + β_3 TREAT_d*POST_t + e_dt  ''')
   model_string = str(metric_name) + '~' + str(groups_name) + '*' +  str(intervention_date_name)
   target_group_before_string = str(groups_name) + '==1' + ' & '  + str(intervention_date_name) + '==0'
   target_group_after_string = str(groups_name) + '==1' + ' & '  + str(intervention_date_name) + '==1'
   control_group_intervention_before_string = str(groups_name) + '==0' + ' & '  + str(intervention_date_name) + '==0'
   control_group_intervention_after_string = str(groups_name) + '==0' + ' & '  + str(intervention_date_name) + '==1'
   if st.button("Run analysis"):
        target_group_before = dataframe.query(target_group_before_string)[metric_name].mean()

        target_group_after = dataframe.query(target_group_after_string)[metric_name].mean()
        target_group_diff = target_group_after - target_group_before



        control_group_intervention_after = dataframe.query(control_group_intervention_after_string)[metric_name].mean()
        #target_group_after - control_group_intervention_after


        control_group_intervention_before = dataframe.query(control_group_intervention_before_string)[metric_name].mean()

        diff_in_diff = (target_group_after-target_group_before)-(control_group_intervention_after-control_group_intervention_before)
        summary_data = {'event_data': ['before intervention', 'after intervention'],
                         'target_data': [target_group_before, target_group_after],
                           'control_data': [control_group_intervention_before, control_group_intervention_after],
                           'counterfactual_data': [target_group_before, target_group_before+(control_group_intervention_after-control_group_intervention_before)]}
        df_summary_data = pd.DataFrame(data=summary_data)
        table_results = smf.ols(model_string, data=dataframe).fit().summary().tables[1]

        tab1, tab2, tab3,tab4  = st.tabs(["Actual Data", "Diff-in-Diff","Regression Model / Explanation"])

        with tab1:

         fig = px.line(dataframe, x=event_date_name, y=metric_name, color=groups_name)
         st.plotly_chart(fig, theme="streamlit")
         with tab2:
            st.header('GDP over time', divider='gray')
      
            x = list(df_summary_data['event_data'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
            x=x,
            y=list(df_summary_data['control_data']),
            name = 'Control group', # Style name/legend entry with html tags
            connectgaps=False # override default to connect the gaps
))
            fig.add_trace(go.Scatter(
            x=x,
            y=list(df_summary_data['target_data']),
            name='Target group',
))
            fig.add_trace(go.Scatter(
            x=x,
            y=[df_summary_data['target_data'][0], df_summary_data['target_data'][0]+(control_group_intervention_after-control_group_intervention_before)],
            name='Counterfactual',
))

            st.plotly_chart(fig, theme="streamlit")
            st.table(summary_data)

         with tab3:
            st.header('Regression Model', divider='gray')
            p_value_list = list([table_results[1][4], table_results[2][4],table_results[3][4], table_results[4][4]])
            #results_variables = np.where(table_results[4][0:4] >= 0.05, 'is not statistically significant', 'is statistically significant')
            st.table(table_results)
            list_variables = list([str(table_results[1][0]), str(table_results[2][0]), str(table_results[3][0]), str(table_results[4][0])])
            
            #index_name = str(explanation_variable_name)
            #st.write(table_results[list_variables.index(str(explanation_variable_name))][1])
            st.write(table_results[1][0], ': The base value when all other variables are zero is', table_results[1][1])
            st.write('The standard error for the intercept is ', table_results[1][2], ' which means the estimated value could vary by approximately ',  round(float(str(table_results[1][2]))) , ' units.')
            st.write('The p-value is ',p_value_list[0])
            st.write('We are 95 per cent confident that the true value of the intercept falls between ', table_results[1][5], ' and ',  table_results[1][6])

            st.write(table_results[2][0], ': The base value when all other variables are zero is', table_results[2][1])
            st.write('The standard error is ', table_results[2][2], ' which means the estimated value could vary by approximately ',  round(float(str(table_results[2][2]))) , ' units.')
            st.write('The p-value is ',p_value_list[1])
            st.write('We are 95 per cent confident that the true value of the intercept falls between ', table_results[2][5], ' and ',  table_results[2][6])

            st.write(table_results[3][0], ': The base value when all other variables are zero is', table_results[3][1])
            st.write('The standard error is ',  table_results[3][2], ' which means the estimated value could vary by approximately ',  round(float(str(table_results[3][2]))) , ' units.')
            st.write('The p-value is ',p_value_list[2])
            st.write('We are 95 per cent confident that the true value of the intercept falls between ', table_results[3][5], ' and ',  table_results[3][6])

            st.write(table_results[4][0], ': The base value when all other variables are zero is', table_results[4][1])
            st.write('The standard error is ', table_results[4][2], ' which means the estimated value could vary by approximately ',  round(float(str(table_results[4][2]))) , ' units.')
            st.write('The p-value is ',p_value_list[3])
            st.write('We are 95 per cent confident that the true value of the intercept falls between ', table_results[4][5], ' and ',  table_results[4][6])



st.caption("Do you wanna contribute to this project?")
st.link_button("Donate", "https://donate.stripe.com/5kAbLQ0Nk1v85nqfZ0")


st.caption("Do you need an experimentation program?")
st.link_button("Contact us", "https://sterlingdata.webflow.io/company/contact")
st.link_button("David Dias' LinkedIn", "https://www.linkedin.com/in/daviddiasrodriguez/")


st.header('References', divider='gray')
st.link_button("Difference-in-Differences - Causal Inference for the Brave and True (Matheus Facure Alves)", "https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html")
st.link_button("Mastering 'Metrics: The Path from Cause to Effect", "https://www.amazon.com/Mastering-Metrics-Path-Cause-Effect/dp/0691152845/ref=sr_1_2?dib=eyJ2IjoiMSJ9.miIdw25tOmutaXlv500au_MXflKZpw6srAJY25Ntai-tpQ5o1nqclJns1Dlpe8H5muFiIr4MRkmiCFyngrRoeGXH85_-hblJSn4zH_JGquw.wtdl9PQzd6Ii-hi9mSLLSWrS7zzPACImEI2A-H7PoVU&dib_tag=se&qid=1726502392&refinements=p_27%3AJoshua+D+Angrist&s=books&sr=1-2&text=Joshua+D+Angrist")


st.caption('Sterling @ 2024')





