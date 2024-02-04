import streamlit as st
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
      chart_columns = []
      is_chart = False
      for col in response.columns:
        if col == 'month':
          is_chart = True
        else:
          chart_columns.append(col)
      if is_chart:
        message_placeholder.chart(response, x='month', y=chart_columns)
      else:
        message_placeholder.dataframe(response, hide_index=is_audience)
      status.update(label="Request Successfully Processed", state="complete", expanded=False)
    else:
      response = generated_query['error_message']
      message_placeholder.markdown(response)
      status.update(label="Error While Processing Request", state="error", expanded=False)
  st.session_state.messages.append({"role": "assistant", "content": response, "status": generated_query['status'], "is_audience": is_audience})