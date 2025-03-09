import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import plotly.express as px

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Difference-in-Differences Analysis Tool',
    page_icon="📊",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/daviddiasrodriguez/',
        'Report a bug': "https://www.linkedin.com/in/daviddiasrodriguez/",
        'About': "Made by David Dias Rodríguez. Sterling @ 2025"
    }
)

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
st.header('Difference-in-Differences Analysis Tool', divider='gray')

st.caption('Tool based on :blue[Matheus Facure Alves Causal Inference for the brave and true handbook.] https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html')
st.caption('This tool is useful when you can not run an A/B Test.')
st.caption('Made by :blue[Sterling]')
#st.caption('This technique falls in the category of quasi-experiments.')


tab_sample_data, tab_own_analysis = st.tabs(["Sample Data", "Build Your Own Analysis"])



with tab_sample_data:
     st.markdown('**Learn how to use this app from sample data**')
     st.image("sterling_diff-in-diff-tool-sharp-causality-week-1.jpg", caption='Real world examples. Just one metric per case study.')

     list_examples = ["Causal inference - diff-in-diff - Raw Data - Example (Sterling example).csv","Diff-In-Diff Examples v1.1 - Dogecoin vs Bitcoin - Transactions.csv", "Diff-In-Diff Examples v1.1 - Diff-in-Diff- Dogecoin vs litecoin.csv", "Diff-In-Diff Examples v1.1 - Diff-in-Diff- Dogecoin vs Ethereum.csv","Diff-In-Diff Examples v1.1 - India vs China - Fertility rate.csv", "Diff-In-Diff Examples v1.1 - Aryma Labs - Market A vs Market B - MMM.csv"  ]
     sample_data_examples = st.selectbox(
               "Select the data example",
               (list_examples), index=0, key='sample_data_examples')
     dataframe = pd.read_csv(sample_data_examples)
     data_as_csv= dataframe.to_csv(index=False).encode("utf-8")


     st.download_button(
     label="Download sample data",
     data=data_as_csv,
     file_name="diff-in-diff-sample-data.csv",
     mime="text/csv"
     )



     col1, col2 = st.columns(2)
     with col1:
          st.write(dataframe)


     with col2:
               
          event_date_name = st.selectbox(
               "Select your event date column",
               (list(dataframe.columns)), index=0, key='event_date_name_sample_data')
          
          metric_name = st.selectbox(
               "Select your metric column",
               (list(dataframe.columns)), index=1, key='metric_name_sample_data')
          groups_name = st.selectbox(
               "Select your groups column",
               (list(dataframe.columns)), index=2, key='groups_name_sample_data')
          intervention_date_name = st.selectbox(
               "Select your intervention date column",
               (list(dataframe.columns)), index=3, key='intervention_date_name_sample_data')
          
          model_string = str(metric_name) + '~' + str(groups_name) + '*' +  str(intervention_date_name)
          target_group_before_string = str(groups_name) + '==1' + ' & '  + str(intervention_date_name) + '==0'
          target_group_after_string = str(groups_name) + '==1' + ' & '  + str(intervention_date_name) + '==1'
          control_group_intervention_before_string = str(groups_name) + '==0' + ' & '  + str(intervention_date_name) + '==0'
          control_group_intervention_after_string = str(groups_name) + '==0' + ' & '  + str(intervention_date_name) + '==1'
     st.latex(r'''Y_dt  = β_0 + β_1 TREAT_d + β_2 POST_t + β_3 TREAT_d*POST_t + e_dt  ''')

     st.caption("You will run a lineal regression model based on the one above:")
     st.caption("Treat variable: This will be your groups column")
     st.caption("Post variable: This will be your intervention column")
     st.caption("Treat*Post variable: This will be the combined effect of the group and intervention column")
     if st.button("Run the sample analysis", key='run_analysis_sample_data'):
                    target_group_before = dataframe.query(target_group_before_string)[metric_name].mean()

                    target_group_after = dataframe.query(target_group_after_string)[metric_name].mean()
                    target_group_diff = target_group_after - target_group_before



                    control_group_intervention_after = dataframe.query(control_group_intervention_after_string)[metric_name].mean()
                    #target_group_after - control_group_intervention_after


                    control_group_intervention_before = dataframe.query(control_group_intervention_before_string)[metric_name].mean()

                    diff_in_diff = (target_group_after-target_group_before)-(control_group_intervention_after-control_group_intervention_before)
                    sample_summary_data_table = {'event_data': ['before intervention', 'after intervention', 'variation (%)'],
                                    'control_data': [control_group_intervention_before, control_group_intervention_after, ((control_group_intervention_after-control_group_intervention_before)/control_group_intervention_before)*100],
                                        'target_data': [target_group_before, target_group_after, ((target_group_after-target_group_before)/target_group_before)*100],
                                        'counterfactual_data': [target_group_before, target_group_before+(control_group_intervention_after-control_group_intervention_before), (((target_group_before+(control_group_intervention_after-control_group_intervention_before))-target_group_before) / target_group_before)*100]}
                    summary_data = {'event_data': ['before intervention', 'after intervention'],
                                    'control_data': [control_group_intervention_before, control_group_intervention_after],
                                        'target_data': [target_group_before, target_group_after],
                                        'counterfactual_data': [target_group_before, target_group_before+(control_group_intervention_after-control_group_intervention_before)]}
                    df_summary_data = pd.DataFrame(data=summary_data)
                    table_results = smf.ols(model_string, data=dataframe).fit().summary().tables[1]

                    tab_actual_data_sample_data, tab_diff_data_sample_data, tab_regression_sample_data = st.tabs(["Current Data", "Diff-in-Diff Analysis","Regression Model / Explanation"])

                    with tab_actual_data_sample_data:

                         fig = px.line(dataframe, x=event_date_name, y=metric_name, color=groups_name)
                         st.plotly_chart(fig, theme="streamlit")
                    with tab_diff_data_sample_data:
                    
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
                         st.table(sample_summary_data_table)
                         st.markdown('**Control group**')
                         st.caption("This is the group with no intervention at all, you will use it as for comparing with the target group which will receive the intervention / change planned.")
                         st.markdown('**Target group**')
                         st.caption("This is the group which will get the intervention / change planned, you will use it as for comparing with the control group and check how the metrics change over time.")
                         st.markdown('**Counterfactual group**')
                         st.caption("This is a group similar to the target group in a scenario when the intervention doesn’t happen at all, the variation will be similar to the control group.")


                    with tab_regression_sample_data:
                         p_value_list = list([table_results[1][4], table_results[2][4],table_results[3][4], table_results[4][4]])
                         #results_variables = np.where(table_results[4][0:4] >= 0.05, 'is not statistically significant', 'is statistically significant')
                         st.table(table_results)
                         list_variables = list([str(table_results[1][0]), str(table_results[2][0]), str(table_results[3][0]), str(table_results[4][0])])
                         
                    
                         st.write(table_results[1][0], ': This is the metric of your control group before the intervention. The base value is', table_results[1][1])
                         st.write('The standard error for the intercept is ', table_results[1][2], ' which means the estimated value could vary by approximately ',  table_results[1][2] , ' units from the base value.')
                         st.write('The p-value is ', p_value_list[0], '. The closer to 0, the higher the metric.')
                         st.write('We are 95 per cent confident that the true value of the control group metric falls between ', table_results[1][5], ' and ',  table_results[1][6])

                         
                         st.write(table_results[2][0], ': The value is the difference between the control group (', round(sample_summary_data_table['control_data'][0], 2) , ') and the target group metric (', round(sample_summary_data_table['target_data'][0], 2), ') . The value is ', table_results[2][1])
                         st.write('The standard error is ', table_results[2][2], ' which means the estimated value could vary by approximately ',  table_results[2][2] , ' units from the base value.')
                         st.write('The p-value is ', p_value_list[1], '. The closer to 0, the higher the metric')
                         st.write('We are 95 per cent confident that the true ', table_results[2][0],  ' value falls between ', table_results[2][5], ' and ',  table_results[2][6])

                         st.write(table_results[3][0], ': This is the value of the intervention effect or the difference between the metric before (', round(sample_summary_data_table['control_data'][0], 2),') and after the intervention (', round(sample_summary_data_table['control_data'][1], 2),') in the control group. The value is ', table_results[3][1])
                         st.write('The standard error is ',  table_results[3][2], ' which means the estimated value could vary by approximately ',  table_results[3][2], ' units from the base value.')
                         st.write('The p-value is ', p_value_list[2], '. The closer to 0, the higher the base value.')
                         st.write('We are 95 per cent confident that the true ', table_results[3][0], 'value falls between ', table_results[3][5], ' and ',  table_results[3][6])

                         st.write(table_results[4][0], ': The value is the difference between the counterfactual group (', round(sample_summary_data_table['counterfactual_data'][1], 2), ') and the target group (', round(sample_summary_data_table['target_data'][1], 2), ') after the intervention. The value is ', table_results[4][1])
                         st.write('The standard error is ', table_results[4][2], ' which means the estimated value could vary by approximately ',  table_results[4][2] , ' units from the base value.')
                         st.write('The p-value is ', p_value_list[3], '. The closer to 0, the higher the base value.')
                         st.write('We are 95 per cent confident that the true ', table_results[4][0],' value falls between ', table_results[4][5], ' and ',  table_results[4][6])









     with tab_own_analysis:




          st.markdown('**How to use the app?**')
          st.caption('1. Upload a file (CSV files only*)')
          st.caption('2. Choose the columns you want for the analysis')
          st.caption('3. Run the analysis and get the insights')


          st.markdown('**Your data should have at least 4 columns:**')
          st.caption('1. Event Date: Daily, weekly, monthly event date')
          st.caption('2. Metric: Numeric / float column with the metric you want to observe. Purchases, Clicks, actions, etc.')
          st.caption('3. Treatment: A boolean column with the followings data values: (0 = No treatment; 1 = Treatment)')
          st.caption('4. Post: A boolean column related with the event date column to indicate the post treatment periods with the followings data values: (0 = Before treatment; 1 = After Treatment)')



          
          st.markdown('**Learn how to use this app from sample data**')
          st.download_button(
     label="Download and learn from sample data ",
     data=data_as_csv,
     file_name="diff-in-diff-sample-data.csv",
     mime="text/csv"
     )


          uploaded_file = st.file_uploader("Upload a file (CSV files only*)")
          if uploaded_file is not None:

          # Can be used wherever a "file-like" object is accepted:
               dataframe = pd.read_csv(uploaded_file)

               #if dataframe:
               
               col1, col2 = st.columns(2)
               with col1:
                    st.write("Take a look of your data")
                    st.write(dataframe)


               with col2:
               
                    event_date_name = st.selectbox(
               "Select your event date column",
               (list(dataframe.columns)), key='event_date_name_own_analysis', index=0)
               #st.write("You selected:", dataframe[event_date_values])
                    metric_name = st.selectbox(
               "Select your metric column",
               (list(dataframe.columns)), key='metric_name_own_analysis', index=1)
                    groups_name = st.selectbox(
               "Select your groups column",
               (list(dataframe.columns)), key='groups_name_own_analysis', index=2)
                    intervention_date_name = st.selectbox(
               "Select your intervention date column",
               (list(dataframe.columns)), key='intervention_date_name_own_analysis', index=3)
               st.latex(r'''Y_dt  = β_0 + β_1 TREAT_d + β_2 POST_t + β_3 TREAT_d*POST_t + e_dt  ''')
               st.write("You will run this model based on the one above:", str(metric_name) + '~' + str(groups_name) + '*' +  str(intervention_date_name) )
               model_string = str(metric_name) + '~' + str(groups_name) + '*' +  str(intervention_date_name)
               target_group_before_string = str(groups_name) + '==1' + ' & '  + str(intervention_date_name) + '==0'
               target_group_after_string = str(groups_name) + '==1' + ' & '  + str(intervention_date_name) + '==1'
               control_group_intervention_before_string = str(groups_name) + '==0' + ' & '  + str(intervention_date_name) + '==0'
               control_group_intervention_after_string = str(groups_name) + '==0' + ' & '  + str(intervention_date_name) + '==1'
               if st.button("Run analysis", key='run_analysis_own_analysis'):
                    target_group_before = dataframe.query(target_group_before_string)[metric_name].mean()

                    target_group_after = dataframe.query(target_group_after_string)[metric_name].mean()
                    target_group_diff = target_group_after - target_group_before



                    control_group_intervention_after = dataframe.query(control_group_intervention_after_string)[metric_name].mean()
                    #target_group_after - control_group_intervention_after


                    control_group_intervention_before = dataframe.query(control_group_intervention_before_string)[metric_name].mean()

                    diff_in_diff = (target_group_after-target_group_before)-(control_group_intervention_after-control_group_intervention_before)
                    sample_summary_data_table = {'event_data': ['before intervention', 'after intervention', 'variation (%)'],
                                    'control_data': [control_group_intervention_before, control_group_intervention_after, ((control_group_intervention_after-control_group_intervention_before)/control_group_intervention_before)*100],
                                        'target_data': [target_group_before, target_group_after, ((target_group_after-target_group_before)/target_group_before)*100],
                                        'counterfactual_data': [target_group_before, target_group_before+(control_group_intervention_after-control_group_intervention_before), (((target_group_before+(control_group_intervention_after-control_group_intervention_before))-target_group_before) / target_group_before)*100]}
                    summary_data = {'event_data': ['before intervention', 'after intervention'],
                                    'control_data': [control_group_intervention_before, control_group_intervention_after],
                                        'target_data': [target_group_before, target_group_after],
                                        'counterfactual_data': [target_group_before, target_group_before+(control_group_intervention_after-control_group_intervention_before)]}
                    df_summary_data = pd.DataFrame(data=sample_summary_data_table)
                    table_results = smf.ols(model_string, data=dataframe).fit().summary().tables[1]

                    tab1, tab2, tab3 = st.tabs(["Current Data", "Diff-in-Diff Analysis","Regression Model / Explanation"])

                    with tab1:

                         fig = px.line(dataframe, x=event_date_name, y=metric_name, color=groups_name)
                         st.plotly_chart(fig, theme="streamlit")
                    with tab2:
                    
                         x = list(df_summary_data['event_data'][0:2])
                         fig = go.Figure()
                         fig.add_trace(go.Scatter(
                         x=x,
                         y=list(df_summary_data['control_data'][0:2]),
                         name = 'Control group', # Style name/legend entry with html tags
                         connectgaps=False # override default to connect the gaps
               ))
                         fig.add_trace(go.Scatter(
                         x=x,
                         y=list(df_summary_data['target_data'][0:2]),
                         name='Target group',
               ))
                         fig.add_trace(go.Scatter(
                         x=x,
                         y=[df_summary_data['target_data'][0], df_summary_data['target_data'][0]+(control_group_intervention_after-control_group_intervention_before)],
                         name='Counterfactual',
               ))

                         st.plotly_chart(fig, theme="streamlit")
                         st.table(df_summary_data)
                         st.markdown('**Control group**')
                         st.caption("This is the group with no intervention at all, you will use it as for comparing with the target group which will receive the intervention / change planned.")
                         st.markdown('**Target group**')
                         st.caption("This is the group which will get the intervention / change planned, you will use it as for comparing with the control group and check how the metrics change over time.")
                         st.markdown('**Counterfactual group**')
                         st.caption("This is a group similar to the target group in a scenario when the intervention doesn’t happen at all, the variation will be similar to the control group.")


                    with tab3:
                         p_value_list = list([table_results[1][4], table_results[2][4],table_results[3][4], table_results[4][4]])
                         #results_variables = np.where(table_results[4][0:4] >= 0.05, 'is not statistically significant', 'is statistically significant')
                         st.table(table_results)
                         list_variables = list([str(table_results[1][0]), str(table_results[2][0]), str(table_results[3][0]), str(table_results[4][0])])
                         
                         st.write(table_results[1][0], ': This is the metric of your control group before the intervention. The base value is', table_results[1][1])
                         st.write('The standard error for the intercept is ', table_results[1][2], ' which means the estimated value could vary by approximately ',  table_results[1][2] , ' units from the base value.')
                         st.write('The p-value is ', p_value_list[0], '. The closer to 0, the higher the metric.')
                         st.write('We are 95 per cent confident that the true value of the control group metric falls between ', table_results[1][5], ' and ',  table_results[1][6])

                         
                         st.write(table_results[2][0], ': The value is the difference between the control group (', round(sample_summary_data_table['control_data'][0], 2) , ') and the target group metric (', round(sample_summary_data_table['target_data'][0], 2), ') . The value is ', table_results[2][1])
                         st.write('The standard error is ', table_results[2][2], ' which means the estimated value could vary by approximately ',  table_results[2][2] , ' units from the base value.')
                         st.write('The p-value is ', p_value_list[1], '. The closer to 0, the higher the metric')
                         st.write('We are 95 per cent confident that the true ', table_results[2][0],  ' value falls between ', table_results[2][5], ' and ',  table_results[2][6])

                         st.write(table_results[3][0], ': This is the value of the intervention effect or the difference between the metric before (', round(sample_summary_data_table['control_data'][0], 2),') and after the intervention (', round(sample_summary_data_table['control_data'][1], 2),') in the control group. The value is ', table_results[3][1])
                         st.write('The standard error is ',  table_results[3][2], ' which means the estimated value could vary by approximately ',  table_results[3][2], ' units from the base value.')
                         st.write('The p-value is ', p_value_list[2], '. The closer to 0, the higher the base value.')
                         st.write('We are 95 per cent confident that the true ', table_results[3][0], 'value falls between ', table_results[3][5], ' and ',  table_results[3][6])

                         st.write(table_results[4][0], ': The value is the difference between the counterfactual group (', round(sample_summary_data_table['counterfactual_data'][1], 2), ') and the target group (', round(sample_summary_data_table['target_data'][1], 2), ') after the intervention. The value is ', table_results[4][1])
                         st.write('The standard error is ', table_results[4][2], ' which means the estimated value could vary by approximately ',  table_results[4][2] , ' units from the base value.')
                         st.write('The p-value is ', p_value_list[3], '. The closer to 0, the higher the base value.')
                         st.write('We are 95 per cent confident that the true ', table_results[4][0],' value falls between ', table_results[4][5], ' and ',  table_results[4][6])
                    
                    







st.subheader("Collaboration and contact links", divider="gray")

st.caption("Do you wanna contribute to this project?")
st.link_button("Donate", "https://donate.stripe.com/6oEaFe4TeegDcF25kl")


st.caption("Do you need an experimentation program?")
st.link_button("Contact us", "https://sterlingdata.webflow.io/company/contact?tool=diff-in-diff-streamlit")
st.link_button("David Dias' LinkedIn", "https://www.linkedin.com/in/daviddiasrodriguez/")


st.header('References & Credits', divider='gray')
st.link_button("Difference-in-Differences - Causal Inference for the Brave and True (Matheus Facure Alves)", "https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html")
st.link_button("Mastering 'Metrics: The Path from Cause to Effect", "https://www.amazon.com/Mastering-Metrics-Path-Cause-Effect/dp/0691152845/ref=sr_1_2?dib=eyJ2IjoiMSJ9.miIdw25tOmutaXlv500au_MXflKZpw6srAJY25Ntai-tpQ5o1nqclJns1Dlpe8H5muFiIr4MRkmiCFyngrRoeGXH85_-hblJSn4zH_JGquw.wtdl9PQzd6Ii-hi9mSLLSWrS7zzPACImEI2A-H7PoVU&dib_tag=se&qid=1726502392&refinements=p_27%3AJoshua+D+Angrist&s=books&sr=1-2&text=Joshua+D+Angrist")
st.link_button("OLS Summary: P-values and Confidence Intervals", "https://albertum.medium.com/ols-summary-p-values-and-confidence-intervals-abd4e3e968cd")
st.link_button("Streamlit / Snowflake Employee: Jakub Kmiotek", "https://www.linkedin.com/in/jakub-kmiotek-18070897/")
st.link_button("Streamlit / Snowflake Employee: Antoni Kędracki", "https://www.linkedin.com/in/akedracki/")
st.link_button("Pedro H. C. Sant'Anna - Difference-in-Differences: A brief guide to practice", "https://psantanna.com/DiD/Instacart_202409.pdf")
st.link_button("Card & Krueger (1994) - The classic DiD study on minimum wage and employment", "https://www.jstor.org/stable/2677856")
st.link_button("Callaway & Sant'Anna (2021) -  Advances in DiD methods for staggered adoption", "https://www.sciencedirect.com/science/article/abs/pii/S0304407620303948")
st.link_button("goodman-bacon 2021 - decomposing DiD estimates in multi-period settings", "https://www.sciencedirect.com/science/article/abs/pii/S0304407621001445")

st.link_button("Blockchain ETL for crypto currency analysis", "https://github.com/blockchain-etl/bitcoin-etl")
st.link_button("Aryma Labs - Proving Efficacy of Marketing Mix Model through the Difference in Difference (DID) Technique", "https://www.techrxiv.org/users/778033/articles/912681-proving-efficacy-of-marketing-mix-modeling-mmm-through-the-difference-in-difference-did-technique")
st.link_button("Meridian - The lastest Marketing Mix Modelling Framework", "https://github.com/google/meridian")
st.link_button("Fertility rate, total (births per woman) per country", "https://api.worldbank.org/v2/en/indicator/SP.DYN.TFRT.IN?downloadformat=csv")



st.header('Python Packages for DiD Analysis', divider='gray')
st.link_button("Differences", "https://github.com/bernardodionisi/differences")
st.link_button("Causal Impact", "https://github.com/google/tfp-causalimpact")


st.caption('Sterling @ 2025')
st.caption('Updated: 09/02/25')

st.markdown(
    """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-Q1JZHKMZE0"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-Q1JZHKMZE0');
</script>
    """, unsafe_allow_html=True)




