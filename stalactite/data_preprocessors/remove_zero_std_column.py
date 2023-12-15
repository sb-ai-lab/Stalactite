from .base_preprocessor import DataPreprocessor


class RemoveZeroStdColumns(DataPreprocessor):
        
    def fit_transform(self, inp_data):
        
        self.nonzero_std_columns = (inp_data.std(axis=0) != 0)
        out_data = inp_data[:, self.nonzero_std_columns]
        
        return out_data
    
    def transform(self, inp_data):
        
        out_data = inp_data[:, self.nonzero_std_columns]
        return out_data