class PrivacyGuard:
    method: str

    def __init__(self, method):
        assert method in ['DP', 'MPC', 'HE', 'TEE', 'GD']
        self.method = method

    def encode_intermediate_results(self):
        ...

    def encode_gradients(self):
        ...