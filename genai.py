import logging, cfg
from vertexai.preview.generative_models import Content, Part
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import CodeGenerationModel, ChatModel, CodeChatModel, TextGenerationModel
import json
import jsonschema
from json import JSONDecodeError
from concurrent.futures import ThreadPoolExecutor

def init():

  # create logger
  global logger
  logger = logging.getLogger('nl2sql')

  global fast_sql_generation_model
  fast_sql_generation_model = createModel(cfg.project_id, "us-central1", cfg.fast_sql_generation_model)

  global fine_sql_generation_model
  fine_sql_generation_model = createModel(cfg.project_id, "us-central1", cfg.fine_sql_generation_model)

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


def gen_dyn_rag_sql(question,table_result_joined, similar_questions):

  similar_questions_str = question_to_query_examples(similar_questions)

  # If no similar question found, use more performant model to generate SQL Query (yet much slower)
  if fast := (len(similar_questions) > 0):
    logger.info("Similar question found, using fast model to generate SQL Query")
  else:
    logger.info("No similar question found, using fine model to generate SQL Query")

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

  context_query = generate_sql(fast_sql_generation_model if fast is True else fine_sql_generation_model, context_prompt)

  return context_query

def sql_explain(question, generated_sql, table_schema, similar_questions):

  logger.info("Starting SQL explanation...")

  response_json = {
    "filters": [{
        "name": "Name of each filter",
        "classification": "classification of the filter",
        "columns": [{
          "name": "name of each column used to implement the filter in \[SQL Query\]",
          "scope": "Scope of the column",
          "is_relevant": "True if the column is appropriate to implement the filter, False otherwise"
        }]
    }],
    "unneeded_filter": [
      "Names of each filter in natural language that are present in \[SQL Query\] and are not matching any classified filter in \[Question\]"
    ]
  }

  schema = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "type": "object",
    "properties": {
      "filters": {
        "type": "array",
        "items": {
          "name": {"type": "string"},
          "classification": {"type": "string"},
          "columns": {
            "type": "array",
            "items": {
              "name": {"type": "string"},
              "scope": {"type": "string"},
              "is_relevant": {"type": "boolean"},
            }
          },
          "required": ["name", "classification", "columns"]
        }
      },
      "unneeded_filter": {
        "type": "array",
        "items": {"type": "string"}
      }
    },
    "required": ["filters"]
  }

  sql_explanation = {
    "is_matching": True,
    "mismatch_details": ''
  }

  context_prompt = f"""
You are an AI for SQL Analysis. Your mission is to analyse a SQL [SQL Query] and identify how its different parts answer a [Question].
You will reply only with a json object as described in the [Analysis Steps].

[Table Schema]:
{table_schema}

[Analysis Steps]:
Let's work this out step by step to make sure we have the right answer:
1. Analyze the tables in [Table Schema], and understand the relations (column and table relations).
2. For each column in each table of [Table Schema], find its scope in its description (format: 'Scope: (time-dependent|global)').
3. Parse the [Question] and identify all the filters required by the question. Make sure to identify all time constraints related to each filter.
4. Classify each filter: 'time-dependent' if time constraints are explicitly given in the [Question] or 'global' otherwise. Some examples: the filter 'purchased for more than $55 in total' has classification 'global' - the filter 'purchased at least 1 Decathlon product in the last 3 months' has classification 'time-dependent'.
5. For each filter, trace back all the columns only part of [Table Schema] used to implement it in [SQL Query]. It's important to stick to the columns described in [Table Schema]. Make sure to prefix the column with the appropriate structure name if needed. For example, the column 'purchased_items' can belong to the structure 'total_products_purchased_by_brands' which has a 'global' scope and should be noted 'total_products_purchased_by_brands.purchased_items', or belong to the structure 'daily_products_purchased_by_brands' which has 'time-dependent' scope and should be noted 'daily_products_purchased_by_brands.purchased_items'.
6. Answer using only the following json format: {json.dumps(response_json)}
7. Always use double quotes "" for json property names and values in the returned json object.
8. Remove ```json prefix and ``` suffix from the outputs. Don't add any comment around the returned json.

Remember that before you answer a question, you must check to see if it complies with your mission above.

[SQL Query]:
{generated_sql}

[Question]:
{question}

[Evaluation]:

"""

  logger.debug('Validation - Question Generation from SQL Prompt: \n' + context_prompt)

  try:

    raw_json = generate_sql(validation_model, context_prompt)
    logger.info("Validation completed with status: \n" + raw_json)
    validation_json = json.loads(raw_json, strict=False)

    # Validate JSON against JSON schema
    jsonschema.validate(validation_json, schema = schema)

    # Analyze results for possible mismatch in SQL Query implementation
    for filter in validation_json['filters']:
      for column in filter['columns']:
        if column['scope'] != filter['classification']:
          sql_explanation['is_matching'] = 'False'
          sql_explanation['mismatch_details'] += "- \"" + filter['name'] + "\" is implemented using column '" + column['name'] + "' with scope '" + column['scope'] + "'. Use a column with scope '" + ('time-dependent' if column['scope'] == 'global' else 'global') + "' instead.\n"

    if len(validation_json['unneeded_filter']) > 0:
      sql_explanation["is_matching"] = False
      sql_explanation["mismatch_details"] += "- The SQL Query implements the following logic which is not requested by the question: \"" + ("\", \"".join(validation_json['unneeded_filter']) + "\"\n")

  except JSONDecodeError as e:
    logger.error("Error while deconding JSON response: " + str(e))
    sql_explanation['is_matching'] = 'False'
    sql_explanation['mismatch_details'] = 'Returned JSON malformed'
  except jsonschema.exceptions.ValidationError as e:
    logger.error("JSON Response does not match expected JSON schema: " + str(e))
    sql_explanation['is_matching'] = 'False'
    sql_explanation['mismatch_details'] = 'Returned JSON malformed'
  except jsonschema.exceptions.SchemaError as e:
    logger.error("Invalid JSON Schema !")
  except Exception as e:
    logger.error("Exception: " + str(e))
    sql_explanation['is_matching'] = 'False'
    sql_explanation['mismatch_details'] = 'Undefined error. Retry'

  logger.info("Validation response analysis: \n" + json.dumps(sql_explanation))

  return sql_explanation


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


    error_msg = ('- Syntax error returned from BigQuery: ' + bq_error_msg if bq_error_msg != None else '') + ('\n' + validation_error_msg if validation_error_msg != None else '')

    context_prompt = f"""
What is an alternative SQL statement to address the errors mentioned below?
Present a different SQL from previous ones. It is important that the query still answer the original question.
Do not repeat suggestions.

Question:
{question}

Previously Generated (bad) SQL Query:
{generated_sql}

Error Messages:
{error_msg}

Table Schema:
{table_schema}

{question_to_query_examples(similar_questions)}

New Generated SQL Query:

    """

    logger.debug('SQL Correction Chat Prompt: \n' + context_prompt)

    response = super().get_chat_response(context_prompt)

    return clean_sql(str(response))