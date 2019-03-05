from keras.models import model_from_json
import numpy as np 

test_file = '../test.csv'
model_file = 'layer_3_model.json'
model_weights_file = 'layer_3_model_weights.hdf5'
output_file = 'predictions_3.csv'

json_file = open(model_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_weights_file)

def load_eval_data(file_name):
    raw_data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
    raw_sample_image = raw_data[0]
    image_size = int(np.sqrt(len(raw_sample_image)))
    X_shape = (len(raw_data), image_size, image_size, 1)
    X_data = np.zeros(X_shape)
    for index, datum in enumerate(raw_data):
        X_data[index] = np.array(datum/255).reshape(image_size, image_size, 1)
    return X_data
X_eval = load_eval_data(test_file)
print("model layers")
#for layer in model.layers:
    #print(layer.get_output_at(0).get_shape().as_list())

with open(output_file, 'w') as f:
    f.write('ImageId,Label\n')
    y_eval = model.predict(X_eval)
    #print("y_evaluation {}".format(y_eval))
    for index, y_hat in enumerate(y_eval):
        prediction = np.argmax(y_hat)
        f.write(str(index+1)+','+str(prediction)+'\n')
f.close()


""" Reference :https://github.com/arunkv/kaggle-digit-recognizer/tree/master/script"""