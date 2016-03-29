def return_dict(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, dict):
            return result
        else:
            return {
                function.__name__: result
            }
    return wrapper