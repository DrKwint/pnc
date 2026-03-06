from flax import nnx

class TransitionModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.l3 = nnx.Linear(64, out_features, rngs=rngs)

    def __call__(self, x):
        h1 = nnx.relu(self.l1(x))
        h2 = nnx.relu(self.l2(h1))
        out = self.l3(h2)
        return out

class MCDropoutTransitionModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs, dropout_rate: float = 0.1):
        self.l1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l3 = nnx.Linear(64, out_features, rngs=rngs)

    def __call__(self, x, deterministic: bool = False):
        h1 = nnx.relu(self.l1(x))
        h1 = self.dropout1(h1, deterministic=deterministic)
        h2 = nnx.relu(self.l2(h1))
        h2 = self.dropout2(h2, deterministic=deterministic)
        out = self.l3(h2)
        return out

class ClassificationModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(in_features, 200, rngs=rngs)
        self.l2 = nnx.Linear(200, 200, rngs=rngs)
        self.l3 = nnx.Linear(200, 200, rngs=rngs)
        self.l4 = nnx.Linear(200, out_features, rngs=rngs)

    def __call__(self, x):
        h1 = nnx.relu(self.l1(x))
        h2 = nnx.relu(self.l2(h1))
        h3 = nnx.relu(self.l3(h2))
        out = self.l4(h3)
        return out

class MCDropoutClassificationModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs, dropout_rate: float = 0.5):
        self.l1 = nnx.Linear(in_features, 200, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l2 = nnx.Linear(200, 200, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l3 = nnx.Linear(200, 200, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l4 = nnx.Linear(200, out_features, rngs=rngs)

    def __call__(self, x, deterministic: bool = False):
        h1 = nnx.relu(self.l1(x))
        h1 = self.dropout1(h1, deterministic=deterministic)
        h2 = nnx.relu(self.l2(h1))
        h2 = self.dropout2(h2, deterministic=deterministic)
        h3 = nnx.relu(self.l3(h2))
        h3 = self.dropout3(h3, deterministic=deterministic)
        out = self.l4(h3)
        return out
