import sys  # 1. Fixed import name

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # 2. We actually call the function to generate the custom message
        self.error_message = CustomException.get_detailed_msg(error_message, error_detail=error_detail)

    @staticmethod
    def get_detailed_msg(error_msg, error_detail: sys):
        # This extracts the traceback info
        _, _, exc_tb = error_detail.exc_info()
        
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        return f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{error_msg}]"
    
    def __str__(self):
        # 3. Now this works because self.error_message was defined in __init__
        return self.error_message