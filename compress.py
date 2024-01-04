from sklearn.cluster import KMeans
import flatbuffers
import numpy as np
import tensorflow as tf

def comp_group(model, BASE_BITS, rand_num):
    subgraph = model.subgraphs[0]
    MAX = 127
    MIN = -128
    #define block_name
    block_name = ['conv2', 'conv3', 'conv4', 'conv5']
    #compress group-wise
    for name in block_name:
        filter_list = []
        filter_offset = []
        print(f'compress...{name}')
        for tensor in subgraph.tensors:
            if name not in tensor.name.decode('utf-8'):
                continue
            if tensor.shape.size < 4:
                continue
            if (tensor.shape[1], tensor.shape[2]) != (3, 3):
                continue

            filters = model.buffers[tensor.buffer].data
            filters = np.reshape(filters, tensor.shape).astype(np.int8)

            cin = tensor.shape[0]
            cout = tensor.shape[3]

            filter_list += [tuple(filters[i, :, :, j].flatten())
                            for i in range(cin) for j in range(cout)]
            filter_offset += [(tensor.buffer, i * np.product(tensor.shape[1:]) + j, cout)
                              for i in range(cin) for j in range(cout)]

        num_filters = len(filter_list)
        print("Total number of filters:", num_filters)

        filter_array = np.array(filter_list, np.int8)
        print('filter_array', filter_array[0,:])
        print('num of clusters =', 2**BASE_BITS)
        result = KMeans(n_clusters=2 ** BASE_BITS, n_init=1, random_state=rand_num).fit(filter_array)

        for i in range(2 ** BASE_BITS):
            indices = np.where(result.labels_ == i)[0]
            cluster = filter_array[indices, :]
            base = np.round(np.mean(cluster, axis=0))

            offset = (cluster - base).astype(np.int8)
            positive_mask = offset > 0
            offset_ = offset.copy()
            for k in range(offset.shape[0]):
                for j in range(offset.shape[1]):
                    if positive_mask[k, j]:
                        low_bits = offset[k, j] & 0b1111
                        rounded_increment = np.where(low_bits >= 8, 16, 0).astype(np.int8)
                        offset_[k, j] = (offset[k, j] & ~0b1111) + rounded_increment
                    else:
                        offset_[k, j] = np.abs(offset[k, j])
                        low_bits = offset_[k, j] & 0b1111
                        rounded_increment = np.where(low_bits >= 8, 16, 0).astype(np.int8)
                        offset_[k, j] = -((offset_[k, j] & ~0b1111) + rounded_increment)

            buffer_map = dict()
            for u in range(indices.size):
                buffer_id, buffer_offset, d = filter_offset[indices[u]]
                if buffer_id not in buffer_map:
                    buffer_map[buffer_id] = []
                buffer_map[buffer_id].append((u, buffer_offset, d))

            for buffer_id in buffer_map:
                data = model.buffers[buffer_id].data.copy()
                for t, buffer_offset, d in buffer_map[buffer_id]:
                    for r in range(offset.shape[1]):
                        offsets = buffer_offset + d * r
                        value = max(MIN, min(MAX, base[r] + offset_[t, r]))
                        data[offsets] = np.array(value, dtype=np.int8)

                model.buffers[buffer_id].data = data
    # save data
    builder = flatbuffers.Builder(1024)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    model_data = builder.Output()

    return model_data

def comp_global(model, BASE_BITS, rand_num):
    subgraph = model.subgraphs[0]
    MAX = 127
    MIN = -128

    filter_list = []
    filter_offset = []

    for tensor in subgraph.tensors:
        if tensor.shape.size < 4:
            continue
        if (tensor.shape[1], tensor.shape[2]) != (3, 3):
            continue

        filters = model.buffers[tensor.buffer].data
        filters = np.reshape(filters, tensor.shape).astype(np.int8)

        cin = tensor.shape[0]
        cout = tensor.shape[3]

        filter_list += [tuple(filters[i, :, :, j].flatten())
                        for i in range(cin) for j in range(cout)]
        filter_offset += [(tensor.buffer, i * np.product(tensor.shape[1:]) + j, cout)
                          for i in range(cin) for j in range(cout)]

    num_filters = len(filter_list)
    print("Total number of filters:", num_filters)

    filter_array = np.array(filter_list, np.int8)
    result = KMeans(n_clusters=2 ** BASE_BITS, n_init=1, random_state=rand_num).fit(filter_array)

    for i in range(2 ** BASE_BITS):
        indices = np.where(result.labels_ == i)[0]
        cluster = filter_array[indices, :]
        base = np.round(np.mean(cluster, axis=0))

        offset = (cluster - base).astype(np.int8)
        positive_mask = offset > 0
        offset_ = offset.copy()
        for k in range(offset.shape[0]):
            for j in range(offset.shape[1]):
                if positive_mask[k, j]:
                    low_bits = offset[k, j] & 0b1111
                    rounded_increment = np.where(low_bits >= 8, 16, 0).astype(np.int8)
                    offset_[k, j] = (offset[k, j] & ~0b1111) + rounded_increment
                else:
                    offset_[k, j] = np.abs(offset[k, j])
                    low_bits = offset_[k, j] & 0b1111
                    rounded_increment = np.where(low_bits >= 8, 16, 0).astype(np.int8)
                    offset_[k, j] = -((offset_[k, j] & ~0b1111) + rounded_increment)

        buffer_map = dict()
        for u in range(indices.size):
            buffer_id, buffer_offset, d = filter_offset[indices[u]]
            if buffer_id not in buffer_map:
                buffer_map[buffer_id] = []
            buffer_map[buffer_id].append((u, buffer_offset, d))

        for buffer_id in buffer_map:
            data = model.buffers[buffer_id].data.copy()
            for t, buffer_offset, d in buffer_map[buffer_id]:
                for r in range(offset.shape[1]):
                    offsets = buffer_offset + d * r
                    value = max(MIN, min(MAX, base[r] + offset_[t, r]))
                    data[offsets] = np.array(value, dtype=np.int8)

            model.buffers[buffer_id].data = data

    # save data
    builder = flatbuffers.Builder(1024)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    model_data = builder.Output()

    return model_data

