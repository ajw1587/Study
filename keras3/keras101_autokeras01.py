# https://autokeras.com/tutorial/image_classification/
import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# One Hot 해도 되고 안해도 된다.
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = ak.ImageClassifier(
    # overwrite = True,
    max_trials = 2
)

model.fit(x_train, y_train, epochs = 1)

results = model.evaluate(x_test, y_test)

print(results)

# ※Parameters※
# inputs Union[autokeras.Input, List[autokeras.Input]]: A list of Node instances. The input node(s) of the AutoModel.
# outputs Union[autokeras.Head, autokeras.Node, list]: A list of Node or Head instances. The output node(s) or head(s) of the AutoModel.
# project_name str: String. The name of the AutoModel. Defaults to 'auto_model'.
# max_trials int: Int. The maximum number of different Keras Models to try. The search may finish before reaching the max_trials. Defaults to 100.
# directory Optional[Union[str, pathlib.Path]]: String. The path to a directory for storing the search outputs. Defaults to None, which would create a folder with the name of the AutoModel in the current directory.
# objective str: String. Name of model metric to minimize or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
# tuner Union[str, Type[autokeras.engine.tuner.AutoTuner]]: String or subclass of AutoTuner. If string, it should be one of 'greedy', 'bayesian', 'hyperband' or 'random'. It can also be a subclass of AutoTuner. Defaults to 'greedy'.
# overwrite bool: Boolean. Defaults to False. If False, reloads an existing project of the same name if one is found. Otherwise, overwrites the project.
# seed Optional[int]: Int. Random seed.
# max_model_size Optional[int]: Int. Maximum number of scalars in the parameters of a model. Models larger than this are rejected.
# **kwargs: Any arguments supported by kerastuner.Tuner.
