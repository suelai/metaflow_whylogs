from metaflow import FlowSpec, step, Parameter

class CaliforniaHousingML(FlowSpec):
    
    test_size = Parameter("test_size", default=0.3)
    random_state = Parameter("random_state", default=42)
    sample_fraction = Parameter("sample_fraction", default=0.5)

    @step
    def start(self):
        from sklearn.datasets import fetch_california_housing
        dd = fetch_california_housing(as_frame=True)
        sample = dd.frame.sample(frac=self.sample_fraction)
        self.data = sample[sample.columns.drop('MedHouseVal')]
        self.target = sample['MedHouseVal']
        self.next(self.scaler_data)

    @step
    def scaler_data(self):
        from sklearn.preprocessing import StandardScaler
        scale = StandardScaler()
        self.X = scale.fit_transform(self.data)
        self.next(self.split_data)

    @step
    def split_data(self):
        from sklearn.model_selection import train_test_split
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X, self.target, 
                                                                            test_size=self.test_size, 
                                                                            random_state=self.random_state)
        self.next(self.train_data)

    @step
    def train_data(self):
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(self.X_train,self.y_train)
        self.training_score = lr.score(self.X_train,self.y_train)
        self.testing_score = lr.score(self.X_test,self.y_test)
        self.next(self.end)

    @step
    def end(self):
        import numpy as np
        msg = "The training score is : {} and the testing score is {}%"
        print(msg.format(round(self.training_score, 3), round(self.testing_score, 3)))

if __name__ == "__main__":
    CaliforniaHousingML()