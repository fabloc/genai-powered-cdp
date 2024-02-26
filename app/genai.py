import logging, cfg
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import CodeGenerationModel, ChatModel, CodeChatModel, TextGenerationModel
import json, re
import jsonschema
from json import JSONDecodeError
from concurrent.futures import ThreadPoolExecutor

def init():

  vertexai.init(project=cfg.project_id, location="us-central1")

  # create logger
  global logger
  logger = logging.getLogger('nl2sql')

  global fast_sql_generation_model
  fast_sql_generation_model = createModel(cfg.fast_sql_generation_model)

  global fine_sql_generation_model
  fine_sql_generation_model = createModel(cfg.fine_sql_generation_model)

  global validation_model
  validation_model = createModel(cfg.validation_model_id)

  global sql_correction_model
  validation_model = createModel(cfg.sql_correction_model_id)

  global executor
  executor = ThreadPoolExecutor(5)

# Initialize Palm Models to be used
def createModel(model_id):

  if model_id == 'code-bison-32k':
    model = CodeGenerationModel.from_pretrained(model_id)
  elif 'gemini-pro' in model_id:
    model = GenerativeModel(model_id)
  elif model_id == 'codechat-bison-32k':
    model = CodeChatModel.from_pretrained(model_id)
  elif 'chat-bison-32k' in model_id:
    model = ChatModel.from_pretrained(model_id)
  elif model_id == 'text-unicorn':
    model = TextGenerationModel.from_pretrained('text-unicorn@001')
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
        "temperature": temperature
    })
    generated_sql = generated_sql_json.candidates[0].content.parts[0].text
  elif isinstance(model, TextGenerationModel):
    generated_sql_json = model.predict(
      str(context_prompt),
      max_output_tokens = 1024,
      temperature = temperature
    )
    generated_sql = generated_sql_json.text
  return clean(generated_sql)

def question_to_query_examples(similar_questions):
  similar_questions_str = ''
  if len(similar_questions) > 0:
    similar_questions_str = "[Good SQL Examples]:\n"
    for similar_question in similar_questions:
      similar_questions_str += "- Question:\n" + similar_question['question'] + "\n-SQL Query:\n" + full_clean(similar_question['sql_query']) + '\n\n'
  return similar_questions_str


def gen_dyn_rag_sql(question,table_result_joined, similar_questions):

  similar_questions_str = question_to_query_examples(similar_questions)

  # If no similar question found, use more performant model to generate SQL Query (yet much slower)
  if fast := (len(similar_questions) > 0):
    logger.info("Similar question found, using fast model to generate SQL Query")
  else:
    logger.info("No similar question found, using fine model to generate SQL Query")

  context_prompt = f"""You are a BigQuery SQL guru. Write a SQL conformant query for Bigquery that answers the [Question] while using the provided context to correctly refer to the BigQuery tables and the needed column names.

[Guidelines]:
{cfg.prompt_guidelines}

[Table Schema]:
{table_result_joined}
{similar_questions_str}

[Question]:
{question}

[SQL Generated]:
"""

    #Column Descriptions:
    #{column_result_joined}

  logger.debug('LLM GEN SQL Prompt: \n' + context_prompt)

  context_query = generate_sql((fast_sql_generation_model if fast is True else fine_sql_generation_model), context_prompt)

  logger.info("SQL query generated:\n" + context_query)

  return full_clean(context_query)

def sql_explain(question, generated_sql, table_schema, similar_questions):

  logger.info("Starting SQL explanation...")

  response_json = {
    "filters": [{
        "name": "List all [Filters]",
        "classification": "classification of the filter",
        "columns": [{
          "name": "Name of every column from [SQL Query] implementing the filter, from [Table Schema]",
          "scope": "Scope of the column. It can have the values 'global', 'time-dependent' or 'both'",
          "is_relevant": "True if the column is correctly implementing the filter, False otherwise"
        }]
    }],
    "reversed_question": "Write in natural language the question that the [SQL Query] is responding to"
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
      "reversed_question": {"type": "string"}
    },
    "required": ["filters", "reversed_question"]
  }

  sql_explanation = {
    "is_matching": True,
    "mismatch_details": ''
  }

  context_prompt = f"""You are an AI for SQL Analysis. Your mission is to analyse a SQL [SQL Query] and identify how its different parts answer a [Question].
You will reply only with a json object as described in the [Analysis Steps].

[Table Schema]:
{table_schema}

{question_to_query_examples(similar_questions)}[Analysis Steps]:
Let's work this out step by step to make sure we have the right answer:
1. Analyze the tables in [Table Schema], and understand the relations (column and table relations).
2. For each column in each table of [Table Schema], find its scope in its description (format: 'Scope: (time-dependent|global)').
3. Analyze the [Question] and break it down into a list of [Filters]. Make sure to associate the correct time constraints to each filter. For example the question "Give me 5 brands at random from the top 100 selling brands" has [Filters] = ["random selection of 5 brands", "top 100 selling brands"] - the question "Number of users with age under 35, who have already bought 'Decathlon' brand products, but not after '2023-08-01'" has the [Filters] = ["age under 35", "purchased at least 1 'Decathlon' brand product", "has not purchased a 'Decathlon' brand product after '2023-08-01'"] - question "Number of users with age under 35, located in France, and who have already bought 'Decathlon' brand products, before August 2023" has [Filters] = ["age under 35", "located in France", "has purchased 'Decathlon' brand product before August 2023"].
4. Classify each filter in [Filters]: 'time-dependent' or 'global'. For example: the filter 'purchased for more than $55 in total' has classification 'global' - the filter 'purchased at least 1 Decathlon product in the last 3 months' has classification 'time-dependent'.
5. For each filter in [Filters], trace back all the matching columns present in [SQL Query] and also part of [Table Schema]. Make sure to prefix the column with the appropriate structure if needed. For example, the column 'purchased_items' can belong to the structure 'total_products_purchased_by_brands' which has a 'global' scope and should be noted 'total_products_purchased_by_brands.purchased_items', or belong to the structure 'daily_products_purchased_by_brands' which has 'time-dependent' scope and should be noted 'daily_products_purchased_by_brands.purchased_items'.
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
        if column['scope'] != 'both' and column['scope'] != filter['classification']:
          sql_explanation['is_matching'] = 'False'
          sql_explanation['mismatch_details'] += "- \"" + filter['name'] + "\" is implemented using column '" + column['name'] + "' with scope '" + column['scope'] + "'. Use a column with scope '" + ('time-dependent' if column['scope'] == 'global' else 'global') + "' instead.\n"
        if column['is_relevant'] == False:
          sql_explanation['is_matching'] = 'False'
          sql_explanation["mismatch_details"] += "- The column '" + column['name'] + "' is either not relevant or not used correctly to implement the filter '" + filter['name'] + "'.\n"

    sql_explanation["reversed_question"] = validation_json['reversed_question']

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
    raise e

  logger.info("Validation response analysis: \n" + json.dumps(sql_explanation))

  return sql_explanation


class CorrectionSession:

  def __init__(self, table_schema: str, question: str, similar_questions: str):

    self.table_schema = table_schema
    self.question = question
    self.similar_questions = similar_questions
    
    self.model = createModel(cfg.sql_correction_model_id)
    self.iterations_history = []

  def add_iteration(self, sql, errors):
    self.iterations_history.append({
      'sql': sql.replace('\n', ' '),
      'errors': errors.replace('\n', '')
    })

  def format_history(self) -> str:
    history_str = ''
    for iteration in self.iterations_history:
      history_str += f"""{iteration['sql']}

"""
    return history_str
  
  def format_last_query(self) -> str:

    last_query = self.iterations_history[-1]
    return f"""SQL Query:
{last_query['sql']}

Errors:
{last_query['errors']}"""


  def get_corrected_sql(self, sql: str, bq_error_msg: str, validation_error_msg: str) -> str:
    logger.info("Sending prompt...")
    error_msg = ('- Syntax error returned from BigQuery: ' + bq_error_msg if bq_error_msg != None else '') + ('\n  ' + validation_error_msg if validation_error_msg != None else '')
    self.add_iteration(sql, error_msg)

    context_prompt = f"""You are a BigQuery SQL guru. This session is trying to troubleshoot a Google BigQuery SQL Query.
As the user provides the [Last Generated SQL Queries with Errors], return a correct [New Generated SQL Query] that fixes the errors.
It is important that the query still answers the original question and follows the following [Guidelines]:

[Guidelines]:
{cfg.prompt_guidelines}

[Table Schema]:
{self.table_schema}
{question_to_query_examples(self.similar_questions)}

[Correction Steps]:
Let's work this out step by step to make sure we have the right corrected SQL Query:
1. Analyze the tables in [Table Schema], and understand the relations (column and table relations).
2. Analyze the SQL Query in [Last Generated SQL Queries with Errors] and understand why it has errors.
3. Propose a SQL Query that corrects the [Last Generated SQL Queries with Errors] while matching the [Question].
4. The [New Generated SQL Query] must not be present in [Forbidden SQL Queries] 
5. Always use double quotes "" for json property names and values in the returned json object.
6. Remove ```json prefix and ``` suffix from the outputs. Don't add any comment around the returned json.

Remember that before you answer a question, you must check to see if it complies with your mission above.

[Question]:
{self.question}

[Last Generated SQL Queries with Errors]:
{self.format_last_query()}

[Forbidden SQL Queries]:
{self.format_history()}
[New Generated SQL Query]:
"""

    logger.debug('SQL Correction Prompt: \n' + context_prompt)

    response = generate_sql(self.model, context_prompt)
    logger.info("Received corrected SQL Query: \n" + response)

    return full_clean(response)


def full_clean(str):
  # Remove unwanted 'sql', 'json', and additional spaces
  return re.sub(' +', ' ', str.replace('\n', ' '))

def clean(str):
  return clean_json(clean_sql(str))

def clean_sql(result):
  result = result.replace("sql", "").replace("```", "")
  return result


def clean_json(result):
  result = result.replace("json", "").replace("```", "")
  return result