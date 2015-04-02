''' Version 1.000

 Code provided by Daniel Jiwoong Im

 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''


import cPickle, gzip, numpy
import theano
import theano.tensor as T
import numpy as np 
import math
import matplotlib as mp
import matplotlib.pyplot as plt

def save_the_weight(x,fname):
    f = file(fname+'.save', 'wb')
    cPickle.dump(x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def separate_data_into_classes(train_set, num_classes, flag=1):
   
    sep_train_set = []
    num_cases_per_class = []

    for class_i in xrange(num_classes):
        train_data = train_set[0][train_set[1]==class_i,:]
        Nc = train_data.shape[0]
        num_cases_per_class.append(Nc) 

        if flag:
            sep_train_set.append(shared_dataset([train_data, class_i *np.ones((Nc,1),dtype='float32')]))
        else:
            sep_train_set.append([train_data, class_i *np.ones((Nc,1),dtype='float32')])
    return sep_train_set, num_cases_per_class



def load_dataset(path):
    # Load the dataset
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return [train_set[0], train_set[1]], \
            [valid_set[0],valid_set[1]], \
            [test_set [0],test_set [1]]


def normalize(data, vdata=None, tdata=None):
    mu   = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    data = ( data - mu ) / std

    if vdata == None and tdata != None:
        tdata = (tdata - mu ) /std
        return data, tdata

    if vdata != None and tdata != None:
        vdata = (vdata - mu ) /std
        tdata = (tdata - mu ) /std
        return data, vdata, tdata
    return data


def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data 

def share_input(x):
    return theano.shared(np.asarray(x, dtype=theano.config.floatX))

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    #When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue

    return shared_x, T.cast(shared_y, 'int32')


'''Given tiles of raw data, this function will return training, validation, and test sets.
r_train - ratio of train set
r_valid - ratio of valid set
r_test  - ratio of test set'''
def gen_train_valid_test(raw_data, raw_target, r_train, r_valid, r_test):
    N = raw_data.shape[0]
    perms = np.random.permutation(N)
    raw_data   = raw_data[perms,:]
    raw_target = raw_target[perms]

    tot = float(r_train + r_valid + r_test)  #Denominator
    p_train = r_train / tot  #train data ratio
    p_valid = r_valid / tot  #valid data ratio
    p_test  = r_test / tot	 #test data ratio
    
    n_raw = raw_data.shape[0] #total number of data		
    n_train =int( math.floor(n_raw * p_train)) # number of train
    n_valid =int( math.floor(n_raw * p_valid)) # number of valid
    n_test  =int( math.floor(n_raw * p_test) ) # number of test

    
    train = raw_data[0:n_train, :]
    valid = raw_data[n_train:n_train+n_valid, :]
    test  = raw_data[n_train+n_valid: n_train+n_valid+n_test,:]
    
    train_target = raw_target[0:n_train]
    valid_target = raw_target[n_train:n_train+n_valid]
    test_target  = raw_target[n_train+n_valid: n_train+n_valid+n_test]
    
    print 'Among ', n_raw, 'raw data, we generated: '
    print train.shape[0], ' training data'
    print valid.shape[0], ' validation data'
    print test.shape[0],  ' test data\n'
    
    train_set = [train, train_target]
    valid_set = [valid, valid_target]
    test_set  = [test, test_target]
    return [train_set, valid_set, test_set]


'''decaying learning rate'''
def get_epsilon(epsilon, n, i):
    return epsilon / ( 1 + i/float(n))


def get_thrd(epoch, tot_epoch):
    return (1.0 - 0.5) * epoch / tot_epoch  + 0.5


'''Display dataset as a tiles'''
def display_dataset(data, patch_sz, tile_shape, scale_rows_to_unit_interval=False, \
                                            binary=False, i=1, fname='dataset'):

    x = tile_raster_images(data, img_shape=patch_sz, \
    						tile_shape=tile_shape, tile_spacing=(1,1), output_pixel_vals=False, scale_rows_to_unit_interval=scale_rows_to_unit_interval)
    
    if binary:
    	x[x==1] = 255		

    ## For MNIST
    if fname != None:
        plt.figure()
        plt.imshow(x,cmap='gray')
        plt.axis('off')
        plt.savefig(fname+'.png')
    else:
        plt.figure()
        plt.imshow(x,cmap='gray')
        plt.axis('off')
        #image = PIL.Image.fromarray(numpy.uint8(x))#.convert('RGB')
        #image.show()

    # For CIFAR10 images
    #plt.imshow(x)
    #image = PIL.Image.fromarray(x).convert('RGB')
    #image.show()
     
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=False,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    
    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).
    
    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.
    
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats
    
    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not
    
    
    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.
    
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    # tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    # tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X[0].dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
		#print tile_row, tile_shape[1], tile_col, X.shape[0]
		#print tile_row * tile_shape[1] + tile_col < X.shape[0]
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
		    #print this_x
		    #print scale_rows_to_unit_interval
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
			#print this_x.shape
			#print this_img
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
			#print this_x.shape
			#print this_img

                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array


if __name__ == '__main__':

    train_set, valid_set, test_set = load_dataset('./mnist.pkl.gz')
    print 123, train_set[1].shape

    train_set_x, train_set_y = shared_dataset(train_set);    
    test_set_x, test_set_y   = shared_dataset(test_set);    
    valid_set_x, valid_set_y = shared_dataset(valid_set); 

    print type(train_set_x)
    print dir(train_set_x)

    data  = train_set_x[2 * 500: 3 * 500]
    print data


