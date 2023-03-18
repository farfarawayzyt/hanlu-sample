import numpy as np

from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import experimental
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import nvidia.dali.fn as fn
import nvidia.dali.math as dmath
import nvidia.dali.types as types

from dalic.rand_augment import rand_augment

image_dir = 'dummy_data'
max_batch_size = 8

def my_random_crop(images, osize=224, padding=int(224 * 0.125)):
    '''
        images: [256, 256, 3]
        padded: [312, 312, 3]
    '''

    # pad, reflect
    horizontal_flip = fn.flip(images, horizontal=1)
    images = fn.cat(horizontal_flip[:, -padding:, :], images, horizontal_flip[:, :padding, :], axis=1) # W
    vertical_flip = fn.flip(images, horizontal=0, vertical=1)
    images = fn.cat(vertical_flip[-padding:, :, :], images, vertical_flip[:padding, :, :], axis=0) # H

    # images: [312, 312, 3]
    images = fn.crop(images, crop_h=osize, crop_w=osize, crop_pos_x=fn.random.uniform(range=(0,1)), crop_pos_y=fn.random.uniform(range=(0,1)))
    return images

@experimental.pipeline_def(enable_conditionals=True)
def strong_pipeline(file_list: str, file_root: str, initial_fill):
    jpegs, labels = fn.readers.file(
        file_list=file_list, file_root=file_root,
        random_shuffle=True, initial_fill=initial_fill)
    images = fn.decoders.image(jpegs, device='mixed')

    images = fn.resize(images, size=(256, 256))
    coin = fn.random.coin_flip()
    images = fn.flip(images, horizontal=coin)
    weak = my_random_crop(images)


    images = rand_augment(weak, n=2, m=9, num_magnitude_bins=10)

    
    images = fn.erase(images, 
        anchor=fn.random.uniform(range=(0,1), shape=(2,)),
        shape=(16, 16), 
        normalized_anchor=True,
        normalized_shape=False,
        fill_value=(127, 127, 127)
    )

    return weak, images, labels.gpu()

if __name__ == '__main__':
    dali_loader = DALIGenericIterator(
        [strong_pipeline(
            batch_size=32, num_threads=8, device_id=0,
            file_list='filelist/imagenet_train_labeled.txt',
            file_root='data',
        )], 
        ['data', 'label'],
        last_batch_policy=LastBatchPolicy.DROP
    )

    from time import perf_counter


    all_du = []
    time_point = perf_counter()
    for i, data in enumerate(dali_loader):
        x0, y0 = data[0]['data'], data[0]['label']

        print(x0.shape, x0.dtype)

        duration = perf_counter() - time_point
        all_du.append(duration)
        print(f'{i} => {duration}')
        time_point = perf_counter()
        if i > 200:
            input()
            break

    import numpy as np
    mean = np.mean(all_du)
    print(f'mean batch: {mean}')