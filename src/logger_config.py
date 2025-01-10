import logging

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the default logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)

# Create a logger object
logger = logging.getLogger('app_logger') 