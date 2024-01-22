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

  def get_chat_response(self, table_schema, similar_questions, question, generated_sql, error_msg):

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

    Tables Schema:
    {table_schema}

    Good SQL Examples:
    {similar_questions}
    """

    response = super().get_chat_response(context_prompt)

    return clean_sql(str(response))


class ExplanationCorrectionChat(SessionChat):

  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'

  explanation_correction_context = f"""You are a BigQuery SQL guru. This session is trying to troubleshoot a Google BigQuery SQL query.
The user provides versions of the query with the Target Question it is answering, the explanation of what the query actually does and details of why the query does not correctly answers the Target Question.
Generate a never seen alternative SQL query that answers better the Target Question by correcting the issues highlighted for the previous version of the SQL query.

Guidelines:
    - Only answer questions relevant to the tables listed in the table schema. If a non-related question comes, answer exactly: select 'Question is not related to the dataset' as unrelated_answer from dual;
    - Join as minimal tables as possible.
    - When joining tables ensure all join columns are the same data_type.
    - Analyze the database and the table schema provided as parameters and undestand the relations (column and table relations).
    - When asked to count the number of users, always perform an estimation using Hyperloglog++ (HLL) sketches using HLL_COUNT.MERGE.
    - For all requests not related to the number of users matching certain criteria, never use estimates like HyperLogLog++ (HLL) sketches
    - Never use GROUP BY on HLL sketches.
    - Never use HLL_COUNT.MERGE inside a WHERE statement.
    - Never use HLL.EXTRACT.
    - Convert TIMESTAMP to DATE.
    - Consider alternative options to CAST function. If performing a CAST, use only Bigquery supported datatypes.
    - Don't include any comments in code.
    - Remove ```sql and ``` from the output and generate the SQL in single line.
    - Tables should be refered to using a fully qualified name (project_id.owner.table_name).
    - Use all the non-aggregated columns from the "SELECT" statement while framing "GROUP BY" block.
    - Return syntactically and semantically correct SQL for BigQuery with proper relation mapping i.e project_id, owner, table and column relation.
    - Use ONLY the column names (column_name) mentioned in Table Schema. DO NOT USE any other column names outside of this.
    - Associate column_name mentioned in Table Schema only to the table_name specified under Table Schema.
    - Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed.
    - Table names are case sensitive. DO NOT uppercase or lowercase the table names.
    - Owner (dataset) is case sensitive. DO NOT uppercase or lowercase the owner.
    - Project_id is case sensitive. DO NOT uppercase or lowercase the project_id.
    """
   
  def __init__(self, model):
      super().__init__(model, self.explanation_correction_context)

  def get_chat_response(self, table_schema, similar_questions, question, generated_sql, generated_explanation, error_msg):

    context_prompt = f"""
What is an alternative SQL statement to address the error mentioned below?
Present a different SQL from previous ones. It is important that the query still answer the original question.
Do not repeat suggestions.

Table Schema:
{table_schema}

Good SQL Examples:
{similar_questions}

Target Question:
    {question}

Previously Generated (bad) SQL Query:
    {generated_sql}

Generated Explanation of the (bad) SQL Query:
    {generated_explanation}

(bad) SQL Query Issues:
    {error_msg}

Corrected SQL Query:

    """

    response = super().get_chat_response(context_prompt)

    return clean_sql(str(response))