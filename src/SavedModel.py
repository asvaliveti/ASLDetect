class SavedModel:
    def __init__(self, state_dict, val_acc):
        self.state_dict = state_dict
        self.val_acc = val_acc

    def get_state_dict(self):
        return self.state_dict