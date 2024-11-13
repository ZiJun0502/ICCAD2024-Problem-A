class Config:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    def __init__(self, params={}):
        if not self._initialized:
            self.params = params
            self._initialized = True