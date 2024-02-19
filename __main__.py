from decimal import Decimal
import streamlit as st
import altair as alt
import nl2sql

st.set_page_config(
    page_title='CDP Demo',
    page_icon='✅'
)

st.title("Customer Data Platform - Segment Explorer")

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    if message["role"] == "user":
      st.markdown(message["content"])
    else:
      if message["status"] == "Success":
        if message['is_chart']:
          st.line_chart(message["content"])
        else:
          st.dataframe(message["content"], hide_index=message["is_audience"])
      else:
        st.markdown(message["content"])

if prompt := st.chat_input("How many users did not purchase anything during the last 3 months?"):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    message_placeholder = st.empty()
    is_audience = False
    is_chart = False
    status = st.status("Processing Request...", expanded=True)
    generated_query = nl2sql.call_gen_sql(prompt, status)
    if generated_query['status'] == 'Success':
      response = generated_query['sql_result']
      if len(response.index) == 1: is_audience = True
      date_col = response.select_dtypes(include=['dbdate'])
      if len(date_col.columns) == 1:
        is_chart = True
        chart_columns = []
        values_columns = []

        # Identify columns which hold categorical values (brands, categories, etc.) from the ones holding numerical values
        for col in response.columns:
          if col != date_col.columns[0]:
            if response[col].dtype == 'float64' or response[col].dtype == 'int64':
              response[col].fillna(0, inplace=True)
              values_columns.append(col)
            else:
              chart_columns.append(col)
        
        # If there is exactly one numerical column, it can be displayed as a chart, otherwise it is too complex, fall-back
        # to tabular chart
        if len(values_columns) == 1:
          response = response.pivot_table(index=date_col.columns[0], columns=chart_columns, values=values_columns[0])
        else:
          response = response.pivot_table(index=date_col.columns[0], columns=chart_columns, values=values_columns)
          is_chart = False
        print(response)
      
      if is_chart:
        message_placeholder.line_chart(response)
      else:
        message_placeholder.dataframe(response, hide_index=is_audience)
        #message_placeholder.markdown(f"""The generated SQL Query answers the question:  
# ***{generated_query['reversed_question']}***""")
      status.update(label="Request Successfully Processed", state="complete", expanded=False)
    else:
      response = generated_query['error_message']
      message_placeholder.markdown(response)
      status.update(label="Error While Processing Request", state="error", expanded=False)
  st.session_state.messages.append({"role": "assistant", "content": response, "status": generated_query['status'], "is_audience": is_audience, "is_chart": is_chart, "reversed_question": generated_query['reversed_question']})