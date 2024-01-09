import yaml, logging, sys
from google.cloud import bigquery
import cdp_server
import nl2sql
import pgvector_handler

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

    cdp_server.init()

    # response = call_gen_sql("Number of users who purchased products from the brand with the most purchases in the last year?")
    # logger.info('Answer:\n' + json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
