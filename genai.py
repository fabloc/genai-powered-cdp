import logging, cfg
from vertexai.preview.generative_models import Content, Part
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import CodeGenerationModel, ChatModel, CodeChatModel, TextGenerationModel
import json
from json import JSONDecodeError

def init():

  # create logger
  global logger
  logger = logging.getLogger('nl2sql')

  #vertexai.init(project=PROJECT_ID, location="us-central1")
  global sql_generation_model
  sql_generation_model = createModel(cfg.project_id, "us-central1", cfg.sql_generation_model_id)

  #vertexai.init(project=PROJECT_ID, location="us-central1")
  global sql_correction_model
  sql_correction_model = createModel(cfg.project_id, "us-central1", cfg.sql_correction_model_id)

  global validation_model
  validation_model = createModel(cfg.project_id, "us-central1", cfg.validation_model_id)

# Initialize Palm Models to be used
def createModel(PROJECT_ID, REGION, model_id):

  if model_id == 'code-bison-32k':
    model = CodeGenerationModel.from_pretrained(model_id)
  elif model_id == 'gemini-pro':
    model = GenerativeModel(model_id)
  elif model_id == 'codechat-bison-32k':
    model = CodeChatModel.from_pretrained(model_id)
  elif model_id == 'chat-bison-32k':
    model = ChatModel.from_pretrained(model_id)
  elif model_id == 'text-unicorn':
    model = TextGenerationModel.from_pretrained(model_id)
  else:
    logger.error("Requested model '" + model_id + "' not supported. Please review the config.ini file.")
    raise ValueError
  return model


def generate_sql(model, context_prompt):
  if isinstance(model, GenerativeModel):
    generated_sql_json = model.generate_content(
      context_prompt,
      generation_config={
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_p": 1
    })
    generated_sql = generated_sql_json.candidates[0].content.parts[0].text
  elif isinstance(model, TextGenerationModel):
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_k": 40
    }
    generated_sql_json = model.predict(
      context_prompt,
      **parameters)
    generated_sql = generated_sql_json.text
  return clean_json(generated_sql)


def gen_dyn_rag_sql(question,table_result_joined, similar_questions):

  similar_questions_str = question_to_query_examples(similar_questions)

  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'
  context_prompt = f"""
You are a BigQuery SQL guru. Write a SQL conformant query for Bigquery that answers the following question while using the provided context to correctly refer to the BigQuery tables and the needed column names.

Guidelines:
{cfg.prompt_guidelines}

Tables Schema:
{table_result_joined}

{similar_questions_str}

[Question]:
{question}

[SQL Generated]:

    """

    #Column Descriptions:
    #{column_result_joined}

  logger.debug('LLM GEN SQL Prompt: \n' + context_prompt)

  context_query = generate_sql(sql_generation_model, context_prompt)

  return context_query

def sql_explain(question, generated_sql, table_schema):

  logger.info("Starting SQL explanation...")

  context_prompt = f"""
You are a BigQuery SQL guru. Generate a high-level question to which the [SQL Query] answers.

Guidelines:
  - Analyze the database and the table schema provided as parameters and understand the relations (column and table relations) and the column descriptions.
  - In the generated question, stay as concise as possible while not missing any filtering and time range specified by the [SQL query].
  - In the generated question, if no time range is specified for a specific filter, consider that it is global, or total.

[Tables Schema]:
{table_schema}

[SQL Query]:
{generated_sql}
"""

  logger.debug('Validation - Question Generation from SQL Prompt: \n' + context_prompt)

  generated_question = generate_sql(validation_model, context_prompt)

  response_json = {
    "is_matching": "{Answer with 'True' or 'False' depending on the outcome of the comparison between the provided Query and the Reference Question}",
    "mismatch_details": "{Write all identified missing or incorrect filters from the Query. If not, return an empty string. Be specific when highlighting a difference, and provide ways to modify the Query to match the Reference Question.}"
  }

  logger.info("Completed SQL explanation.")

  logger.info("Starting SQL validation...")

  context_prompt = f"""
Compare a Query to a Reference Question and assess whether they are equivalent or not and how the Query should be modified to match the Reference Question.

[Guidelines]:
- Answer using the following json format:
{response_json}
- Remove ```json prefix and ``` suffix from the outputs.
- Use double quotes "" for json property names and values in the returned json object.

[Reference Question]:
{question}

[Query]:
{generated_question}
"""

  logger.debug('Validation - Question Comparison Prompt: \n' + context_prompt)

  try:

    sql_explanation = generate_sql(validation_model, context_prompt)
    logger.info("Validation completed with status: \n" + sql_explanation)
    validation_json = json.loads(sql_explanation, strict=False)

  except JSONDecodeError as e:
    logger.error("Error while deconding JSON response:: " + str(e))
    sql_explanation['is_matching'] = 'False'
    sql_explanation['mismatch_details'] = 'Returned JSON malformed'
  except Exception as e:
    logger.error("Exception: " + str(e))
    sql_explanation['is_matching'] = 'False'
    sql_explanation['mismatch_details'] = 'Undefined error. Retry'


  validation_json['generated_question'] = generated_question

  return validation_json


class SessionChat:

  def __init__(self, model, context):

    self.chat = model.start_chat(history=[
      Content(role="user", parts=[Part.from_text(context)]),
      Content(role="model", parts=[Part.from_text('ok')]),
    ])
      
  def get_chat_response(self, prompt: str) -> str:
      logger.info("Sending chat prompt...")
      response = self.chat.send_message(prompt)
      logger.info("Chat prompt received: " + response.text)
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
      As the user provides versions of the query and the errors with the SQL Query, return a never seen alternative SQL query that fixes the errors.
      It is important that the query still answers the original question.

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

    {question_to_query_examples(similar_questions)}
    """

    logger.debug('LLM GEN SQL Prompt: \n' + context_prompt)

    response = super().get_chat_response(context_prompt)

    return clean_sql(str(response))


def question_to_query_examples(similar_questions):
  similar_questions_str = ''
  if len(similar_questions) > 0:
    similar_questions_str = "Good SQL Examples:\n\n"
    for similar_question in similar_questions:
      similar_questions_str += "    [Question]:\n" + similar_question['question'] + "\n    [SQL Query]:\n" + similar_question['sql_query'] + "\n\n"
  return similar_questions_str


class ExplanationCorrectionChat(SessionChat):

  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'

  explanation_correction_context = f"""You are a BigQuery SQL guru. This session is trying to troubleshoot a Google BigQuery SQL query.
The user provides versions of the query with the Original Question it is answering, the question that the query actually answers to and details of why the query does not correctly answers the Original Question.
Generate a never seen alternative SQL query that answers better the Original Question by correcting the issues highlighted for the previous version of the SQL query.

Guidelines:
{cfg.prompt_guidelines}
    """
   
  def __init__(self, model):
      super().__init__(model, self.explanation_correction_context)

  def get_chat_response(self, table_schema, similar_questions, question, generated_sql, generated_explanation, error_msg):

    similar_questions_str = question_to_query_examples(similar_questions)

    context_prompt = f"""
What is an alternative SQL statement to address the error mentioned below?
Present a different SQL from previous ones. It is important that the query be corrected to answer the Original Question.
Do not repeat suggestions.

[Table Schema]:
{table_schema}

{similar_questions_str}

[Target Question]:
    {question}

[Previously Generated (bad) SQL Query]:
    {generated_sql}

[Question answered by the Generated (bad) SQL Query]:
    {generated_explanation}

[(bad) SQL Query Issues]:
    {error_msg}

[Corrected SQL Query]:

    """

    response = super().get_chat_response(context_prompt)

    return clean_sql(str(response))