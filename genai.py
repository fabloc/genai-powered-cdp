import logging, cfg
from vertexai.preview.generative_models import Content, Part
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import CodeGenerationModel, ChatModel, CodeChatModel, TextGenerationModel
import json
from json import JSONDecodeError
from concurrent.futures import ThreadPoolExecutor

def init():

  # create logger
  global logger
  logger = logging.getLogger('nl2sql')

  #vertexai.init(project=PROJECT_ID, location="us-central1")
  global sql_generation_model
  sql_generation_model = createModel(cfg.project_id, "us-central1", cfg.sql_generation_model)

  #vertexai.init(project=PROJECT_ID, location="us-central1")
  global sql_correction_model
  sql_correction_model = createModel(cfg.project_id, "us-central1", cfg.sql_correction_model_id)

  global validation_model
  validation_model = createModel(cfg.project_id, "us-central1", cfg.validation_model_id)

  global executor
  executor = ThreadPoolExecutor(5)

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


def generate_sql(model, context_prompt, temperature = 0.0):
  if isinstance(model, GenerativeModel):
    generated_sql_json = model.generate_content(
      context_prompt,
      generation_config={
        "max_output_tokens": 1024,
        "temperature": temperature,
        "top_p": 1
    })
    generated_sql = generated_sql_json.candidates[0].content.parts[0].text
  elif isinstance(model, TextGenerationModel):
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": temperature,
        "top_k": 40
    }
    generated_sql_json = model.predict(
      context_prompt,
      **parameters)
    generated_sql = generated_sql_json.text
  return clean_json(generated_sql)



def question_to_query_examples(similar_questions):
  similar_questions_str = ''
  if len(similar_questions) > 0:
    similar_questions_str = "Good SQL Examples:\n\n"
    for similar_question in similar_questions:
      similar_questions_str += "    - Question:\n" + similar_question['question'] + "\n    -SQL Query:\n" + similar_question['sql_query'] + "\n\n"
  return similar_questions_str


def gen_dyn_rag_sql(question,table_result_joined, similar_questions, fast: bool = False):

  similar_questions_str = question_to_query_examples(similar_questions)

  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'
  context_prompt = f"""
You are a BigQuery SQL guru. Write a SQL conformant query for Bigquery that answers the following question while using the provided context to correctly refer to the BigQuery tables and the needed column names.

Guidelines:
{cfg.prompt_guidelines}

Tables Schema:
{table_result_joined}

{similar_questions_str}

Question:
  {question}

SQL Generated:

    """

    #Column Descriptions:
    #{column_result_joined}

  logger.debug('LLM GEN SQL Prompt: \n' + context_prompt)

  context_query = generate_sql(sql_generation_model, context_prompt)

  return context_query

def sql_explain(question, generated_sql, table_schema, similar_questions):

  logger.info("Starting SQL explanation...")

  response_json = {
    "is_matching": "Answer with 'True' or 'False'",
    "mismatch_details": "All identified errors. Be specific in the description. Don't propose a corrected SQL query. If no errors were found, return an empty string"
  }

  example_json = {
    "is_matching": "False",
    "mismatch_details": "'who purchased at least 5 'Roxy' products before january 2022' should be implemented using 'daily_purchased_products_by_brands.purchased_items' aggregated using SUM and filtered with a time condition on 'session_day'"
  }

  context_prompt = f"""
You are an AI for SQL validation. Your mission is to classify a SQL [SQL Query] as valid or invalid by performing a semantic comparison with the [Question].
- Analyze the [Table Schema] provided below, understand the relations (column and table relations). 
- Make sure to understand the Scope of each column, which can be 'daily' or 'global', as specified in their description in the [Table Schema].

[Table Schema]:
{table_schema}

[Validation Steps]:
1. Scan the [Question] and extract all the properties with their associated filters. In particular, understand specific time filters. Classify every property: 'global' (if there is an explicit 'global' or 'in total' in the [Question] or by default no time filter) or 'time-bound' otherwise.
2. For each classified property, identify how it is implemented in the [SQL Query] and make sure that the right columns with the correct scope are used: 'global' columns for 'global' properties and aggregated 'daily' columns.
3. If a column with 'daily' scope is present in the [SQL Query], it should be aggregated using SUM, COUNT, etc., and filtered using 'session_day' with time conditions in a 'WHERE' block. 
4. Answer using the following json format: {json.dumps(response_json)}
5. Always use double quotes "" for json property names and values in the returned json object.
6. Remove ```json prefix and ``` suffix from the outputs.

Here is an example of an SQL Query with a Question and the Evaluation of whether the SQL Query matches the Question:

Remember that before you answer a question, you must check to see if it complies with your mission above.

[SQL Query]:
{generated_sql}

[Question]:
{question}

[Evaluation]:

"""

  logger.debug('Validation - Question Generation from SQL Prompt: \n' + context_prompt)

  try:

    sql_explanation = generate_sql(validation_model, context_prompt, temperature = 0.8)
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

  return validation_json


class SessionChat:

  def __init__(self, model, context):

    self.chat = model.start_chat(history=[
      Content(role="user", parts=[Part.from_text(context)]),
      Content(role="model", parts=[Part.from_text('YES')]),
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
It is important that the query still answers the original question and follows the following guidelines:

Guidelines:
{cfg.prompt_guidelines}

Please reply 'YES' if you understand.
    """
   
  def __init__(self):
      super().__init__(sql_correction_model, self.sql_correction_context)

  def get_chat_response(self, table_schema, similar_questions, question, generated_sql, bq_error_msg, validation_error_msg):


    error_msg = ('- ' + bq_error_msg if bq_error_msg != None else '') + ('\n- ' + validation_error_msg if validation_error_msg != None else '')

    context_prompt = f"""
What is an alternative SQL statement to address the Error Messages mentioned below?
Present a different SQL from previous ones. It is important that the query still answer the original question.
Do not repeat suggestions.

Question:
{question}

Previously Generated (bad) SQL Query:
{generated_sql}

Error Messages:
{error_msg}

Tables Schema:
{table_schema}

{question_to_query_examples(similar_questions)}
    """

    logger.debug('SQL Correction Chat Prompt: \n' + context_prompt)

    response = super().get_chat_response(context_prompt)

    return clean_sql(str(response))