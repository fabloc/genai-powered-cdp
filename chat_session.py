import logging, cfg
from vertexai.preview.generative_models import Content, Part

class SessionChat:

  def __init__(self, model, context):
    # create logger
    logger = logging.getLogger(self.__class__.__name__)
    self.logger = logger

    self.chat = model.start_chat(history=[
      Content(role="user", parts=[Part.from_text(context)]),
      Content(role="model", parts=[Part.from_text('ok')]),
    ])
      
  def get_chat_response(self, prompt: str) -> str:
      self.logger.info("Sending chat prompt...")
      response = self.chat.send_message(prompt)
      self.logger.info("Chat prompt received: " + response.text)
      return response.text


def clean_sql(result):
  result = result.replace("```sql", "").replace("```", "")
  return result


def clean_json(result):
  result = result.replace("```json", "").replace("```", "")
  return result


class SQLCorrectionChat(SessionChat):

  sql_correction_context = f"""

      You are a BigQuery SQL guru. This session is trying to troubleshoot a Google BigQuery SQL query.
      As the user provides versions of the query and the errors returned by BigQuery,
      return a never seen alternative SQL query that fixes the errors.
      It is important that the query still answer the original question.

      Guidelines:
      {cfg.prompt_guidelines}
    """
   
  def __init__(self, model):
      super().__init__(model, self.sql_correction_context)

  def get_chat_response(self, question, generated_sql, table_schema, error_msg, similar_questions):

    context_prompt = f"""
      What is an alternative SQL statement to address the error mentioned below?
      Present a different SQL from previous ones. It is important that the query still answer the original question.
      Do not repeat suggestions.

    Question:
    {question}

    Previously Generated (bad) SQL Query:
    {generated_sql}

    Error Message:
    {error_msg}

    Table Schema:
    {table_schema}

    Good SQL Examples:
    {similar_questions}
    """

    response = super().get_chat_response(context_prompt)

    return clean_sql(str(response))


class ExplanationCorrectionChat(SessionChat):

  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'

  explanation_correction_context = f"""
        TODO
    """
   
  def __init__(self, model):
      super().__init__(model, self.explanation_correction_context)

  def get_chat_response(self, question, generated_sql, table_schema, error_msg, similar_questions):

    context_prompt = f"""
      What is an alternative SQL statement to address the error mentioned below?
      Present a different SQL from previous ones. It is important that the query still answer the original question.
      Do not repeat suggestions.

    Question:
    {question}

    Previously Generated (bad) SQL Query:
    {generated_sql}

    Error Message:
    {error_msg}

    Table Schema:
    {table_schema}

    Good SQL Examples:
    {similar_questions}
    """

    response = super().get_chat_response(context_prompt)

    return clean_sql(str(response))