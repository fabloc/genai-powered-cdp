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
    status = st.status("Processing Request...", expanded=True)
    generated_query = nl2sql.call_gen_sql(prompt, status)
    if generated_query['status'] == 'Success':
      response = generated_query['sql_result']
      is_audience = True if len(response.index) == 1 else False
      date_col = response.select_dtypes(include=['dbdate'])
      if len(date_col.columns) == 1:
        is_chart = True
        chart_columns = []
        for col in response.columns:
          if col != date_col.columns[0]:
            chart_columns.append(col)
        grouped_df = response.groupby([date_col.columns[0]]).sum()
        print("Grouped Dataframe: \n" + str(grouped_df))
        print("dtypes of columns: " + str(grouped_df.dtypes))
        # reshaped_df = grouped_df.unstack(level=date_col.columns[0])
        message_placeholder.line_chart(grouped_df)
      else:
        message_placeholder.dataframe(response, hide_index=is_audience)
        #message_placeholder.markdown(f"""The generated SQL Query answers the question:  
# ***{generated_query['reversed_question']}***""")
      status.update(label="Request Successfully Processed", state="complete", expanded=False)
    else:
      response = generated_query['error_message']
      message_placeholder.markdown(response)
      status.update(label="Error While Processing Request", state="error", expanded=False)
  st.session_state.messages.append({"role": "assistant", "content": response, "status": generated_query['status'], "is_audience": is_audience, "reversed_question": generated_query['reversed_question']})