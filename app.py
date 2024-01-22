import time
import gradio as gr
import logging, yaml, sys
import nl2sql, pgvector_handler

def slow_echo(message, history):
    generated_query = nl2sql.call_gen_sql(message)
    if generated_query['status'] == 'success':
        response = '<center>' + generated_query['sql_result'] + '</center><br/>Generated SQL:<br/>' + generated_query['generated_sql']
    else:
        response = generated_query['error_message']
        if 'generated_sql' in generated_query and generated_query['generated_sql'] is not None:
            response += '\nBest generated SQL query:\n' + generated_query['generated_sql']
    return response

demo = gr.ChatInterface(slow_echo)

def main():

    # Load the log config file
    with open('logging_config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    # Configure the logging module with the config file
    logging.config.dictConfig(config)

    # create logger
    global logger
    logger = logging.getLogger('nl2sql')

    # Override default uncaught exception handler to log all exceptions using the custom logger
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    logger.info("-------------------------------------------------------------------------------")
    logger.info("-------------------------------------------------------------------------------")

    pgvector_handler.init()

    nl2sql.init()


    # response = call_gen_sql("Number of users who purchased products from the brand with the most purchases in the last year?")
    # logger.info('Answer:\n' + json.dumps(response, indent=2))

    demo.launch()

if __name__ == "__main__":
    main()