from absl import app, flags
from load_model import convert_keras_model
from compress import comp_group, comp_global
from evaluate import evaluate
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.python.platform import gfile
import os

FLAGS = flags.FLAGS

# 환경 변수 처리
flags.DEFINE_string('model_name', 'resnet101', 'Model name')
flags.DEFINE_integer('base_bits', 8, 'Number of base bits')
flags.DEFINE_enum('clustering_unit', 'global', ['layer', 'group', 'global','l_same','l_ratio','g_same','g_ratio'], 'Clustering unit')
flags.DEFINE_integer('rand_num', 4561, 'random_state_KMeans')

def main(argv):
    # 환경 변수 읽기
    model_name = FLAGS.model_name
    base_bits = FLAGS.base_bits
    clustering_unit = FLAGS.clustering_unit

    #load model
    model_path = 'origin_model/' + model_name + '.tflite'
    if not os.path.isfile(model_path):
        tflite_model = convert_keras_model(model_name)
        open(model_path, 'wb').write(tflite_model)

    with gfile.Open(model_path, 'rb') as model_file:
        model_data = model_file.read()

    model_obj = schema_fb.Model.GetRootAsModel(model_data, 0)
    model = schema_fb.ModelT.InitFromObj(model_obj)
    subgraph = model.subgraphs[0]
    rand_num = FLAGS.rand_num
    #compress model
    if clustering_unit == 'group':
        model_data = comp_group(model, base_bits, rand_num)
    elif clustering_unit == 'global':
        model_data = comp_global(model, base_bits, rand_num)

    #save model
    output_path = f"comp_model/{model_name}_{clustering_unit}_{base_bits}_{rand_num}.tflite"
    with open(output_path, "wb") as out_file:
        out_file.write(model_data)
    print(f'{model_name}_{clustering_unit}_{base_bits}_{rand_num} :: ')
    #evaluate model
    evaluate(output_path)

if __name__ == "__main__":
    app.run(main)
