class ExpInfo():
    def __init__(self, dirname):

        _arr = dirname.split("_")
        self.cut_x = float(_arr[2])
        self.cut_y = float(_arr[3])
        self.cut_z = float(_arr[4])
        self.tickness = float(_arr[5])
        self.no_gap_between_slices = bool(_arr[6])
        self.num_of_missing_slices = int(_arr[7])
        self.start_data_id = int(_arr[8])
        self.is_curvature = bool(_arr[9])
        self.end_data_id = int(_arr[10])
        self.brain_region = _arr[11]
        self.num_of_slices = int(_arr[12])
    
  # This method allows you to easily see the state of the object when you print it
    def __repr__(self):
        # vars()를 사용하여 모든 변수와 값을 딕셔너리로 가져오기
        variables = vars(self)

        _str = ""
        for var_name, var_value in variables.items():
            _str += f"{var_name}:{var_value}, "
        return _str
