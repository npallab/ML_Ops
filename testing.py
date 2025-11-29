from src.logger import get_logger
from src.customexception import CustomException
import sys  

logger = get_logger(__name__)

def test_divide_numbers(a, b):
    try:
        result=a/b 
        logger.info(f"Division result: {result}")
    except ZeroDivisionError as e:
        logger.error("Attempted to divide by zero.")
        raise CustomException("Division by zero is not allowed.", sys)
if __name__ == "__main__":
    try:
        test_divide_numbers(2,0)
    except CustomException as ce:
        logger.error(str(ce))
    